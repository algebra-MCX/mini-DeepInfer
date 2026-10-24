"""Sparse attention that reads its keys through row indices, preserving the reference's BF16 probability rounding.

Like FlashMLA's sparse kernels, keys are not gathered into a [T, keys, dim] buffer first: each query
row lists the rows of a KV table it attends to (-1 marks an empty slot), and the kernel loads them in
place. A query reads two such tables, the sliding window and the compressed cache, the same split as
FlashMLA's `key_cache` / `key_cache_extra`.
"""

import torch
import triton
import triton.language as tl

QUERY_CHUNK = 2048  # Queries per launch; bounds the fp32 score scratch at QUERY_CHUNK * heads * keys.


@triton.jit
def _load_keys(KV, Idx, n, row, start, d, cols, dim, TILE: tl.constexpr):
    """Key tile [TILE, WIDTH] at columns start.. of the row's index list, zero where the slot is empty."""
    j = start + tl.arange(0, TILE)
    idx = tl.load(Idx + row * n + j, j < n, other=-1)
    valid = idx >= 0
    kv = tl.load(KV + idx.to(tl.int64)[:, None] * dim + d[None, :], valid[:, None] & cols[None, :], other=0.0)
    return kv, valid, j < n


@triton.jit
def _score_pass(q, KV, Idx, n, row, scores, maximum, d, cols, head_ok, dim,
                SCALE: tl.constexpr, TILE: tl.constexpr):
    """Write the scores of one key table and fold them into the running maximum."""
    for start in range(0, n, TILE):
        kv, valid, inside = _load_keys(KV, Idx, n, row, start, d, cols, dim, TILE)
        s = tl.where(valid[None, :], tl.dot(q, tl.trans(kv)) * SCALE, -float("inf"))
        maximum = tl.maximum(maximum, tl.max(s, 1))
        tl.store(scores + start + tl.arange(0, TILE)[None, :], s, head_ok[:, None] & inside[None, :])
    return maximum


@triton.jit
def _value_pass(KV, Idx, n, row, scores, maximum, acc, denominator, d, cols, head_ok, dim, TILE: tl.constexpr):
    """Add one key table's BF16-rounded probabilities times its values (keys double as values)."""
    for start in range(0, n, TILE):
        kv, _, inside = _load_keys(KV, Idx, n, row, start, d, cols, dim, TILE)
        s = tl.load(scores + start + tl.arange(0, TILE)[None, :], head_ok[:, None] & inside[None, :],
                    other=-float("inf"))
        p = tl.exp(s - maximum[:, None])
        denominator += tl.sum(p, 1)
        acc = tl.dot(p.to(tl.bfloat16), kv, acc)
    return acc, denominator


@triton.jit
def _sparse_attention(Q, KV1, Idx1, n1, KV2, Idx2, n2, Sink, Scores, Out, heads, dim,
                      SCALE: tl.constexpr, WIDTH: tl.constexpr, HEADS: tl.constexpr, TILE: tl.constexpr):
    """One query row and a block of HEADS heads, which share the row's keys (multi-query attention).

    The heads form the M dimension of tensor-core products, so each key tile is loaded once per
    block rather than once per head. A first pass writes the scores and finds the global key maximum;
    the second rounds the unnormalized probabilities to BF16 against that maximum, as the reference does.
    """
    row, block = tl.program_id(0), tl.program_id(1)
    h = block * HEADS + tl.arange(0, HEADS)
    d = tl.arange(0, WIDTH)
    head_ok, cols = h < heads, d < dim
    q = tl.load(Q + (row * heads + h[:, None]) * dim + d[None, :], head_ok[:, None] & cols[None, :], other=0.0)
    scores1 = Scores + (row * heads + h[:, None]) * (n1 + n2)  # [table 1 keys | table 2 keys]
    scores2 = scores1 + n1

    maximum = tl.full((HEADS,), -1e30, tl.float32)
    maximum = _score_pass(q, KV1, Idx1, n1, row, scores1, maximum, d, cols, head_ok, dim, SCALE, TILE)
    maximum = _score_pass(q, KV2, Idx2, n2, row, scores2, maximum, d, cols, head_ok, dim, SCALE, TILE)
    tl.debug_barrier()  # the second pass reads scores written by other threads of this program

    denominator = tl.exp(tl.load(Sink + h, head_ok, other=0.0) - maximum)
    acc = tl.zeros((HEADS, WIDTH), tl.float32)
    acc, denominator = _value_pass(KV1, Idx1, n1, row, scores1, maximum, acc, denominator,
                                   d, cols, head_ok, dim, TILE)
    acc, denominator = _value_pass(KV2, Idx2, n2, row, scores2, maximum, acc, denominator,
                                   d, cols, head_ok, dim, TILE)
    out = Out + (row * heads + h[:, None]) * dim + d[None, :]
    tl.store(out, tl.div_rn(acc, denominator[:, None]), head_ok[:, None] & cols[None, :])


def sparse_attention_cuda(q: torch.Tensor, sources: list[tuple[torch.Tensor, torch.Tensor]],
                          sink: torch.Tensor, scale: float) -> torch.Tensor:
    """q [T, H, D]; one or two (kv [N, D], idx [T, K] int32) key tables."""
    kv1, idx1 = sources[0]
    kv2, idx2 = sources[1] if len(sources) > 1 else (kv1, idx1[:, :0])  # an empty second table
    rows, heads, dim = q.shape
    n1, n2 = idx1.shape[1], idx2.shape[1]
    out = torch.empty_like(q)
    block = 16 if heads <= 16 else 32
    scores = torch.empty(min(rows, QUERY_CHUNK), heads, n1 + n2, dtype=torch.float32, device=q.device)
    for start in range(0, rows, QUERY_CHUNK):
        r = slice(start, start + QUERY_CHUNK)
        _sparse_attention[(q[r].shape[0], triton.cdiv(heads, block))](
            q[r], kv1, idx1[r], n1, kv2, idx2[r], n2, sink, scores, out[r], heads, dim,
            SCALE=scale, WIDTH=max(16, triton.next_power_of_2(dim)), HEADS=block, TILE=32,
            num_warps=4, num_stages=3, enable_fp_fusion=False,
        )
    return out
