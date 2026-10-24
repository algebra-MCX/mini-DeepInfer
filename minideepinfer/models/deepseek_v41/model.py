"""DeepSeek-V4.1 text model for a paged, continuously batched runtime.

The math follows the official reference (`inference/model.py`) and is validated against it, but the
data layout is an engine's: a step processes a flat batch of tokens from many sequences, each at its
own position, and all recurrent state lives in caches owned by the runner (`V41Cache`):

    window ring     per sequence slot, the last `window_size` sliding-window KV of every layer
    compressor tail per sequence slot, the tokens of a not-yet-complete compression group
    compressed KV   paged, one entry per `ratio` tokens, written only by KV-source layers
    indexer keys    paged, alongside the compressed KV

Attention for a query attends over its sliding window plus up to `index_topk` compressed entries
chosen by the indexer. Layers between two KV sources read the source's cache, and layers between two
index sources reuse the source's top-k choice; a forward-local `StepContext` carries both down.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from minideepinfer.models.deepseek_v41.activation import swiglu_cuda
from minideepinfer.models.deepseek_v41.attention import sparse_attention_cuda
from minideepinfer.models.deepseek_v41.config import V41Config
from minideepinfer.models.deepseek_v41.engram import EngramHasher
from minideepinfer.models.deepseek_v41.hc import collapse_cuda, expand_cuda, split_sinkhorn_cuda
from minideepinfer.models.deepseek_v41.indexer import index_scores_cuda
from minideepinfer.models.deepseek_v41.moe import combine_cuda
from minideepinfer.models.deepseek_v41.norm import rms_norm_cuda
from minideepinfer.models.deepseek_v41.parallel import all_gather_rows, reduce_scatter_rows, world
from minideepinfer.models.deepseek_v41.quant import fp4_fake_quant, fp8_fake_quant
from minideepinfer.models.deepseek_v41.weights import BLOCK, dequant_fp4, dequant_fp8

PAGE_TOKENS = 128  # Positions covered by one KV cache block (the reference's 128-token store granularity).
ATTN_QUERY_CHUNK = 256  # Queries per gathered attention slice; bounds the [T, keys, head_dim] buffer.
INDEX_QUERY_CHUNK = 512  # Queries per indexer scoring slice; bounds the [T, heads, entries] buffer.
DEQUANT_CHUNK_VALUES = 2**28  # Weights dequantized at once (1 GiB in fp32); bounds the temporaries of an expert stack.


# ---------------------------------------------------------------------------------------------
# Runtime inputs and caches.
# ---------------------------------------------------------------------------------------------


@dataclass
class V41Batch:
    """One step's tokens. Token-level tensors are [T]; sequence-level ones are [B]."""

    input_ids: torch.Tensor
    positions: torch.Tensor
    token_seq: torch.Tensor  # Batch row of each token.
    slots: torch.Tensor  # State slot of each token's sequence.
    chunk_start: torch.Tensor  # First position computed this step, per token.
    chunk_end: torch.Tensor  # One past the last position computed this step, per token.
    block_tables: torch.Tensor  # [B, max_blocks] int64, 0-padded.
    ngram_ids: torch.Tensor | None  # [T, max_ngram_size] raw token ids, -1 before the sequence start.
    cu_seqlens: list[int]  # Host-side token offsets of each sequence, [B + 1].
    max_end: int  # Host-side max of chunk_end.


@dataclass
class V41Cache:
    window: torch.Tensor  # [n_layers, n_slots, window, head_dim] bf16
    comp_kv_state: dict[int, torch.Tensor]  # layer -> [n_slots, ratio, head_dim] fp32
    comp_score_state: dict[int, torch.Tensor]
    comp_kv: dict[int, torch.Tensor]  # kv-source layer -> [n_blocks * entries, head_dim] bf16
    index_k: dict[int, torch.Tensor]  # kv-source layer -> [n_blocks * entries, index_head_dim] bf16

    @staticmethod
    def bytes_per_block(cfg: V41Config) -> int:
        total = 0
        for layer in cfg.kv_source_layers:
            entries = PAGE_TOKENS // cfg.compress_ratios[layer]
            total += entries * (cfg.head_dim + cfg.index_head_dim) * 2
        return total

    @staticmethod
    def bytes_per_slot(cfg: V41Config) -> int:
        total = cfg.n_layers * cfg.window_size * cfg.head_dim * 2
        for layer in cfg.kv_source_layers:
            total += 2 * cfg.compress_ratios[layer] * cfg.head_dim * 4
        return total

    @classmethod
    def allocate(cls, cfg: V41Config, num_blocks: int, num_slots: int, device: str = "cuda") -> V41Cache:
        def entries(layer: int) -> int:
            return num_blocks * (PAGE_TOKENS // cfg.compress_ratios[layer])

        sources = cfg.kv_source_layers
        multi = [s for s in sources if cfg.compress_ratios[s] > 1]
        state_shape = lambda s: (num_slots, cfg.compress_ratios[s], cfg.head_dim)  # noqa: E731
        return cls(
            window=torch.zeros(
                cfg.n_layers, num_slots, cfg.window_size, cfg.head_dim, dtype=torch.bfloat16, device=device
            ),
            comp_kv_state={s: torch.zeros(state_shape(s), device=device) for s in multi},
            comp_score_state={s: torch.zeros(state_shape(s), device=device) for s in multi},
            comp_kv={s: torch.zeros(entries(s), cfg.head_dim, dtype=torch.bfloat16, device=device) for s in sources},
            index_k={
                s: torch.zeros(entries(s), cfg.index_head_dim, dtype=torch.bfloat16, device=device) for s in sources
            },
        )

    def copy_slot(self, src: int, dst: int) -> None:
        """Copies a sequence's recurrent state (sliding window, compressor tails) to another slot."""
        self.window[:, dst] = self.window[:, src]
        for state in (*self.comp_kv_state.values(), *self.comp_score_state.values()):
            state[dst] = state[src]


@dataclass
class StepContext:
    """Per-forward values shared across layers."""

    win_chunk_idx: torch.Tensor  # [T, window] flat token index of each window key in this step's batch
    win_ring_idx: torch.Tensor  # [T, window] flat ring index of each window key in the cache
    win_from_chunk: torch.Tensor  # [T, window] bool: key computed this step (else read from the ring)
    win_valid: torch.Tensor  # [T, window] bool: key position >= 0
    ring_write_idx: torch.Tensor  # [T] flat ring index each token is written to; slot 0 if it leaves the window
    topk: dict[int, torch.Tensor] = field(default_factory=dict)  # index-source layer -> [T, k] entry ids, -1 = none
    candidates: torch.Tensor | None = None  # [T, n] bool, from the candidate-source layer's indexer


def paged_entry(block_tables: torch.Tensor, rows, entry: torch.Tensor, ratio: int) -> torch.Tensor:
    """Flat index into a paged compressed cache of entry `entry` of the sequences at batch `rows`."""
    entries = PAGE_TOKENS // ratio
    return block_tables[rows, entry * ratio // PAGE_TOKENS] * entries + entry % entries


def build_step_context(batch: V41Batch, window: int) -> StepContext:
    pos = batch.positions
    t = torch.arange(pos.numel(), device=pos.device)
    j = torch.arange(window, device=pos.device)
    key_pos = pos[:, None] - (window - 1) + j
    # Only the last `window` tokens of each chunk survive in the ring; the others are written to the
    # reserved slot 0, which keeps the shapes fixed (no boolean indexing, so no host sync).
    keep = pos >= batch.chunk_end - window
    return StepContext(
        win_chunk_idx=(t[:, None] - (window - 1) + j).clamp_min(0),
        win_ring_idx=batch.slots[:, None] * window + key_pos.remainder(window),
        win_from_chunk=key_pos >= batch.chunk_start[:, None],
        win_valid=key_pos >= 0,
        ring_write_idx=torch.where(keep, batch.slots, 0) * window + pos.remainder(window),
    )


# ---------------------------------------------------------------------------------------------
# Building blocks.
# ---------------------------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda and x.is_contiguous() and self.weight.is_contiguous() and 0 < x.shape[-1] <= 8192:
            return rms_norm_cuda(x, self.weight, self.eps)
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


def precompute_freqs_cis(
    dim: int,
    seqlen: int,
    *,
    base: float,
    original_seq_len: int = 0,
    factor: float = 1.0,
    beta_fast: int = 32,
    beta_slow: int = 1,
) -> torch.Tensor:
    """Rotary frequencies as complex exponentials, with YaRN when original_seq_len > 0."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:

        def corrected_dim(rotations: float) -> float:
            return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    freqs = torch.outer(torch.arange(seqlen), freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def rope_tail(x: torch.Tensor, freqs: torch.Tensor, rope_dim: int, inverse: bool = False) -> torch.Tensor:
    """Rotates the last `rope_dim` features (adjacent pairs as complex numbers) by per-token freqs [T, rope_dim / 2]."""
    head, tail = x[..., :-rope_dim], x[..., -rope_dim:]
    c = torch.view_as_complex(tail.float().unflatten(-1, (-1, 2)))
    f = freqs.conj() if inverse else freqs
    if c.ndim == 3:
        f = f[:, None]
    tail = torch.view_as_real(c * f).flatten(-2).to(x.dtype)
    return torch.cat([head, tail], dim=-1)


def sparse_attention(
    q: torch.Tensor, kv: torch.Tensor, valid: torch.Tensor, sink: torch.Tensor, scale: float
) -> torch.Tensor:
    """Multi-query attention over gathered keys, with a per-head sink in the denominator.

    q [T, H, D], kv [T, K, D] (keys double as values), valid [T, K]. A row with no valid key yields 0.
    """
    if (q.is_cuda and all(x.is_contiguous() for x in (q, kv, valid, sink))
            and q.dtype == kv.dtype == torch.bfloat16 and valid.dtype == torch.bool
            and sink.dtype == torch.float32 and 0 < q.shape[-1] <= 512 and kv.shape[1] > 0):
        return sparse_attention_cuda(q, kv, valid, sink, scale)
    s = torch.einsum("thd,tkd->thk", q.float(), kv.float()) * scale
    s = s.masked_fill(~valid[:, None, :], float("-inf"))
    m = s.amax(-1).clamp_min(-1e30)
    p = torch.exp(s - m[..., None])
    denom = p.sum(-1) + torch.exp(sink[None] - m)
    o = torch.einsum("thk,tkd->thd", p.to(torch.bfloat16).float(), kv.float()) / denom[..., None]
    return o.to(torch.bfloat16)


def index_scores(q: torch.Tensor, keys: torch.Tensor, weights: torch.Tensor, cu_seqlens: list[int],
                 token_seq: torch.Tensor | None = None) -> torch.Tensor:
    """Indexer scores sum_h weights[t, h] * relu(q[t, h] . keys[seq(t), n]) -> [T, n].

    q [T, H, D], keys [B, n, D] (one row per sequence), weights [T, H]. With one query per sequence
    (decode) all sequences are scored in one batched GEMM. Otherwise one fused kernel scores the batch
    on CUDA bf16 given each row's sequence (`token_seq`), else each sequence in query slices.
    """
    if q.size(0) == keys.size(0):
        return (torch.einsum("bhd,bnd->bhn", q, keys).relu_() * weights.unsqueeze(-1)).sum(dim=1)
    d = q.size(-1)
    if (token_seq is not None and q.is_cuda and q.dtype == keys.dtype == weights.dtype == torch.bfloat16
            and d >= 16 and d & (d - 1) == 0 and all(x.is_contiguous() for x in (q, keys, weights, token_seq))):
        return index_scores_cuda(q, keys, weights, token_seq)
    score = torch.empty(q.size(0), keys.size(1), dtype=q.dtype, device=q.device)
    for b in range(len(cu_seqlens) - 1):
        for start in range(cu_seqlens[b], cu_seqlens[b + 1], INDEX_QUERY_CHUNK):
            rows = slice(start, min(start + INDEX_QUERY_CHUNK, cu_seqlens[b + 1]))
            s = torch.einsum("shd,nd->shn", q[rows], keys[b])
            score[rows] = (s.relu_() * weights[rows].unsqueeze(-1)).sum(dim=1)
    return score


def select_candidate_blocks(
    scores: torch.Tensor, visible: torch.Tensor, topk_blocks: int, block_size: int
) -> torch.Tensor:
    """Level one of the indexer's two-level top-k: keep the `topk_blocks` best blocks of entries per query.

    scores [T, n] with unreachable entries at -inf; visible [T] counts each query's reachable entries.
    The block holding a query's newest entry is always kept: it is only partly filled, so an older full
    block could otherwise outscore it. Returns a [T, n] bool mask.
    """
    width = scores.size(-1)
    blocks = F.pad(scores, (0, -width % block_size), value=float("-inf")).unflatten(-1, (-1, block_size)).amax(-1)
    last = (visible - 1).div(block_size, rounding_mode="floor")
    blocks = blocks.masked_fill(torch.arange(blocks.size(-1), device=scores.device) == last[:, None], float("inf"))
    top = blocks.topk(min(topk_blocks, blocks.size(-1)), dim=-1)
    keep = torch.zeros_like(blocks, dtype=torch.bool).scatter_(-1, top.indices, top.values > float("-inf"))
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


def hc_split_sinkhorn(
    mixes: torch.Tensor, scale: torch.Tensor, base: torch.Tensor, hc: int, iters: int, eps: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Splits hyper-connection mixes into pre / post weights and a doubly stochastic comb matrix."""
    if mixes.is_cuda and iters >= 1 and all(
        t.is_contiguous() and t.dtype == torch.float32 for t in (mixes, scale, base)
    ):
        return split_sinkhorn_cuda(mixes, scale, base, hc, iters, eps)
    pre = torch.sigmoid(mixes[:, :hc] * scale[0] + base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[:, hc : 2 * hc] * scale[1] + base[hc : 2 * hc])
    comb = (mixes[:, 2 * hc :] * scale[2] + base[2 * hc :]).view(-1, hc, hc)
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def linear(in_features: int, out_features: int, dtype: torch.dtype | None = None) -> nn.Linear:
    return nn.Linear(in_features, out_features, bias=False, dtype=dtype)


class QuantLinear(nn.Module):
    """A checkpoint-format linear layer, with an optional bf16 cache for frozen inference weights.

    FP8 weights carry one E8M0 scale per 32x32 block; FP4 weights pack two E2M1 values per byte with
    one E8M0 scale per 32 inputs. The dequantized weight is exactly what loading in bf16 would give.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        fp4: bool = False,
        storage: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        """`storage` optionally provides preallocated (weight, scale) tensors, e.g. views into stacked experts."""
        super().__init__()
        if storage is not None:
            weight, scale = storage
        elif fp4:
            weight, scale = self.empty_fp4(in_features, out_features)
        else:
            weight = torch.empty(out_features, in_features, dtype=torch.float8_e4m3fn)
            scale = torch.empty(-(-out_features // BLOCK), -(-in_features // BLOCK), dtype=torch.float8_e8m0fnu)
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.scale = nn.Parameter(scale, requires_grad=False)
        self.register_buffer("_cached_weight", None, persistent=False)

    @staticmethod
    def empty_fp4(in_features: int, out_features: int, *batch: int) -> tuple[torch.Tensor, torch.Tensor]:
        weight = torch.empty(*batch, out_features, in_features // 2, dtype=torch.int8)
        scale = torch.empty(*batch, out_features, in_features // BLOCK, dtype=torch.float8_e8m0fnu)
        return weight, scale

    def _dequant(self) -> torch.Tensor:
        if self.weight.dtype == torch.int8:
            return dequant_fp4(self.weight, self.scale)
        return dequant_fp8(self.weight, self.scale)

    def dequant(self) -> torch.Tensor:
        return self._cached_weight if self._cached_weight is not None else self._dequant()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.dequant())


class EngramTable(nn.Module):
    """An n-gram hash table in FP8 (one scale per 32 values of a row), sharded by rows across ranks."""

    def __init__(self, num_embeddings: int, dim: int) -> None:
        super().__init__()
        world_size, rank = world()
        self.part = -(-num_embeddings // world_size)
        self.start = rank * self.part
        self.weight = nn.Parameter(torch.empty(self.part, dim, dtype=torch.float8_e4m3fn), requires_grad=False)
        self.scale = nn.Parameter(torch.empty(self.part, dim // BLOCK, dtype=torch.float8_e8m0fnu), requires_grad=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        num_rows, sharded = ids.size(0), world()[0] > 1
        if sharded:
            ids = all_gather_rows(ids)
        local = ids - self.start
        miss = (local < 0) | (local >= self.part)
        local = local.masked_fill(miss, 0)
        rows = self.weight[local].float().unflatten(-1, (-1, BLOCK)) * self.scale[local].float().unsqueeze(-1)
        rows = rows.flatten(-2).to(torch.bfloat16).masked_fill(miss.unsqueeze(-1), 0)
        return reduce_scatter_rows(rows, num_rows) if sharded else rows


# ---------------------------------------------------------------------------------------------
# Attention.
# ---------------------------------------------------------------------------------------------


class Compressor(nn.Module):
    """Pools each group of `ratio` consecutive tokens into one KV latent (ratio 1: a plain projection)."""

    def __init__(self, cfg: V41Config, ratio: int) -> None:
        super().__init__()
        self.ratio = ratio
        self.norm = RMSNorm(cfg.head_dim, cfg.norm_eps)
        # The pooling of ratio > 1 runs in fp32, so those projections are kept in fp32.
        dtype = torch.float32 if ratio > 1 else torch.bfloat16
        self.wkv = linear(cfg.dim, cfg.head_dim, dtype)
        if ratio > 1:
            self.wgate = linear(cfg.dim, cfg.head_dim, dtype)

    def forward(
        self, x: torch.Tensor, batch: V41Batch, cache: V41Cache, layer_id: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Returns (latents before RoPE, token index of each latent's last member, validity or None).

        The number of groups a step completes is data dependent; to keep shapes fixed, latents are
        produced for an upper bound of groups and the padding ones are marked invalid.
        """
        pos, r = batch.positions, self.ratio
        if r == 1:
            return self.norm(self.wkv(x)), torch.arange(x.size(0), device=x.device), None

        xf = x.float()
        kv, score = self.wkv(xf), self.wgate(xf)
        state_kv = cache.comp_kv_state[layer_id].view(-1, kv.size(-1))
        state_score = cache.comp_score_state[layer_id].view(-1, kv.size(-1))

        # Groups completed by a token of this step; members before the chunk come from the state. A chunk
        # of L tokens completes at most L // r + 1 groups, so T // r + B bounds the whole batch.
        max_done = x.size(0) // r + len(batch.cu_seqlens) - 1
        done = torch.nonzero_static((pos + 1) % r == 0, size=max_done, fill_value=-1).squeeze(-1)
        valid = done >= 0
        done = done.clamp_min(0)
        k = torch.arange(r, device=x.device)
        member_pos = pos[done, None] - (r - 1) + k
        from_chunk = (member_pos >= batch.chunk_start[done, None])[..., None]
        chunk_idx = (done[:, None] - (r - 1) + k).clamp_min(0)
        state_idx = batch.slots[done, None] * r + member_pos.remainder(r)
        g_kv = torch.where(from_chunk, kv[chunk_idx], state_kv[state_idx])
        g_score = torch.where(from_chunk, score[chunk_idx], state_score[state_idx])
        pooled = (g_kv * g_score.softmax(dim=1)).sum(dim=1)

        # Tokens of a group left incomplete at the end of the chunk wait in the state; the others are
        # written to the reserved slot 0.
        tail = pos >= (batch.chunk_end // r) * r
        tail_idx = torch.where(tail, batch.slots, 0) * r + pos.remainder(r)
        state_kv[tail_idx] = kv
        state_score[tail_idx] = score
        return self.norm(pooled.to(x.dtype)), done, valid


class Indexer(nn.Module):
    """Scores compressed entries per query and keeps the `index_topk` best (a small side attention)."""

    def __init__(self, cfg: V41Config, layer_id: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.ratio = cfg.compress_ratios[layer_id]
        self.source = cfg.kv_source_of(layer_id)
        self.owns_k = layer_id in cfg.kv_source_layers
        self.is_candidate_source = layer_id == cfg.candidate_source_layer
        self.uses_candidates = 0 <= cfg.candidate_source_layer < layer_id
        self.n_heads, self.head_dim = cfg.index_n_heads, cfg.index_head_dim
        self.wq_b = QuantLinear(cfg.q_lora_rank, self.n_heads * self.head_dim)
        self.weights_proj = linear(cfg.dim, self.n_heads)
        if self.owns_k:
            self.wk = linear(cfg.head_dim, self.head_dim)
            self.k_norm = RMSNorm(self.head_dim, cfg.norm_eps)

    def make_keys(self, latent: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        k = rope_tail(self.k_norm(self.wk(latent)), freqs, self.cfg.rope_head_dim)
        return fp4_fake_quant(k, 32, e4m3_scale=False)

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        freqs: torch.Tensor,
        batch: V41Batch,
        cache: V41Cache,
        ctx: StepContext,
    ) -> torch.Tensor:
        """Returns [T, k] compressed entry ids per query, -1 where fewer entries are visible."""
        T, r = x.size(0), self.ratio
        n_max = batch.max_end // r
        if n_max == 0:
            return torch.full((T, 0), -1, dtype=torch.long, device=x.device)
        q = self.wq_b(qr).unflatten(-1, (self.n_heads, self.head_dim))
        q = fp4_fake_quant(rope_tail(q, freqs, self.cfg.rope_head_dim), 32, e4m3_scale=False)
        weights = self.weights_proj(x) * (self.head_dim**-0.5 * self.n_heads**-0.5)

        n = torch.arange(n_max, device=x.device)
        keys = cache.index_k[self.source][paged_entry(batch.block_tables, slice(None), n, r)]  # [B, n_max, head_dim]
        score = index_scores(q, keys, weights, batch.cu_seqlens, batch.token_seq)
        visible = (batch.positions + 1) // r  # entries whose whole group precedes or ends at the query
        score = score.masked_fill(n[None] >= visible[:, None], float("-inf"))
        if self.is_candidate_source:
            ctx.candidates = select_candidate_blocks(
                score, visible, self.cfg.candidate_topk_blocks, self.cfg.candidate_block_size
            )
        elif self.uses_candidates:  # level two: our own scores, restricted to the source's candidate blocks
            assert ctx.candidates is not None and ctx.candidates.shape == score.shape
            score = score.masked_fill(~ctx.candidates, float("-inf"))
        idx = score.topk(min(self.cfg.index_topk, n_max), dim=-1).indices
        return torch.where(idx < visible[:, None], idx, -1)


class Attention(nn.Module):
    def __init__(self, cfg: V41Config, layer_id: int, freqs: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_id = layer_id
        self.n_heads, self.head_dim, self.rope_dim = cfg.n_heads, cfg.head_dim, cfg.rope_head_dim
        self.n_groups, self.o_lora_rank = cfg.o_groups, cfg.o_lora_rank
        self.ratio = cfg.compress_ratios[layer_id]
        self.scale = self.head_dim**-0.5

        self.attn_sink = nn.Parameter(torch.empty(self.n_heads, dtype=torch.float32))
        self.wq_a = QuantLinear(cfg.dim, cfg.q_lora_rank)
        self.q_norm = RMSNorm(cfg.q_lora_rank, cfg.norm_eps)
        self.wq_b = QuantLinear(cfg.q_lora_rank, self.n_heads * self.head_dim)
        self.wkv = QuantLinear(cfg.dim, self.head_dim)
        self.kv_norm = RMSNorm(self.head_dim, cfg.norm_eps)
        self.wo_a = QuantLinear(self.n_heads * self.head_dim // self.n_groups, self.n_groups * self.o_lora_rank)
        self.wo_b = QuantLinear(self.n_groups * self.o_lora_rank, cfg.dim)

        self.is_kv_source = layer_id in cfg.kv_source_layers
        self.is_index_source = layer_id in cfg.index_source_layers
        if self.is_kv_source:
            self.compressor = Compressor(cfg, self.ratio)
        if self.is_index_source:
            self.indexer = Indexer(cfg, layer_id)
        # Compressed layers rotate at their own theta (with YaRN); pure sliding-window layers do not.
        self.freqs = freqs["compress" if self.ratio else "window"]

    def _compressed(
        self, x: torch.Tensor, qr: torch.Tensor, batch: V41Batch, cache: V41Cache, ctx: StepContext
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Flat cache ids [T, k] of the selected compressed keys and their validity.

        A KV-source layer first writes this step's compressed entries (and indexer keys) to the cache.
        """
        cfg, r = self.cfg, self.ratio
        source = cfg.kv_source_of(self.layer_id)
        if self.is_kv_source:
            latent, last_tok, valid = self.compressor(x, batch, cache, self.layer_id)
            entry = (batch.positions[last_tok] + 1) // r - 1
            flat = paged_entry(batch.block_tables, batch.token_seq[last_tok], entry, r)
            if valid is not None:  # padding latents go to the reserved block 0
                flat = torch.where(valid, flat, 0)
            freqs = self.freqs[entry * r]  # a latent takes the position of its group's first token
            if self.is_index_source:  # the indexer keys use the latent before RoPE
                cache.index_k[source][flat] = self.indexer.make_keys(latent, freqs)
            latent = fp4_fake_quant(rope_tail(latent, freqs, self.rope_dim), 16, e4m3_scale=True)
            cache.comp_kv[source][flat] = latent

        if self.is_index_source:
            ctx.topk[self.layer_id] = self.indexer(x, qr, self.freqs[batch.positions], batch, cache, ctx)
        idx = ctx.topk[cfg.index_source_of(self.layer_id)]
        return paged_entry(batch.block_tables, batch.token_seq[:, None], idx.clamp_min(0), r), idx >= 0

    def forward(self, x: torch.Tensor, batch: V41Batch, cache: V41Cache, ctx: StepContext) -> torch.Tensor:
        T = x.size(0)
        freqs = self.freqs[batch.positions]
        qr = self.q_norm(self.wq_a(x))
        q = rope_tail(self.wq_b(qr).unflatten(-1, (self.n_heads, self.head_dim)), freqs, self.rope_dim)
        kv = fp8_fake_quant(rope_tail(self.kv_norm(self.wkv(x)), freqs, self.rope_dim))

        ring = cache.window[self.layer_id].view(-1, self.head_dim)
        if self.ratio:
            comp_flat, comp_valid = self._compressed(x, qr, batch, cache, ctx)
            comp_kv = cache.comp_kv[self.cfg.kv_source_of(self.layer_id)]

        # Keys are gathered per query slice: the full [T, keys, head_dim] buffer would not fit for long chunks.
        o = torch.empty_like(q)
        for start in range(0, T, ATTN_QUERY_CHUNK):
            rows = slice(start, start + ATTN_QUERY_CHUNK)
            keys = torch.where(
                ctx.win_from_chunk[rows, :, None], kv[ctx.win_chunk_idx[rows]], ring[ctx.win_ring_idx[rows]]
            )
            valid = ctx.win_valid[rows]
            if self.ratio:
                keys = torch.cat([keys, comp_kv[comp_flat[rows]]], dim=1)
                valid = torch.cat([valid, comp_valid[rows]], dim=1)
            o[rows] = sparse_attention(q[rows], keys, valid, self.attn_sink, self.scale)
        ring[ctx.ring_write_idx] = kv

        o = rope_tail(o, freqs, self.rope_dim, inverse=True)
        o = fp8_fake_quant(o.view(T, self.n_groups, -1))
        # wo_a is block-diagonal over groups: each group projects only its own heads.
        o = torch.einsum("tgd,grd->tgr", o, self.wo_a.dequant().view(self.n_groups, self.o_lora_rank, -1))
        return self.wo_b(o.flatten(1))


# ---------------------------------------------------------------------------------------------
# MoE, Engram and the hyper-connection block.
# ---------------------------------------------------------------------------------------------


def swiglu(gate: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
    gate, up = gate.float(), up.float()
    if limit > 0:
        up = up.clamp(-limit, limit)
        gate = gate.clamp(max=limit)
    return F.silu(gate) * up


def swiglu_bf16(gate: torch.Tensor, up: torch.Tensor, limit: float,
                weights: torch.Tensor | None = None) -> torch.Tensor:
    """Apply routing weights in FP32 before the one BF16 cast consumed by the output projection."""
    if (gate.is_cuda and gate.device == up.device and gate.shape == up.shape
            and gate.ndim == 2 and gate.dtype == up.dtype == torch.bfloat16
            and gate.is_contiguous() and up.is_contiguous()
            and (weights is None or (weights.shape == (gate.size(0), 1)
                                     and weights.device == gate.device and weights.dtype == torch.float32
                                     and weights.is_contiguous()))):
        return swiglu_cuda(gate, up, limit, weights)
    h = swiglu(gate, up, limit)
    return (h if weights is None else weights * h).to(torch.bfloat16)


class Expert(nn.Module):
    def __init__(
        self,
        dim: int,
        inter_dim: int,
        swiglu_limit: float,
        fp4: bool,
        storage: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> None:
        super().__init__()
        storage = storage or {}
        self.w1 = QuantLinear(dim, inter_dim, fp4, storage.get("w1"))
        self.w2 = QuantLinear(inter_dim, dim, fp4, storage.get("w2"))
        self.w3 = QuantLinear(dim, inter_dim, fp4, storage.get("w3"))
        self.swiglu_limit = swiglu_limit

    def forward(self, x: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
        return self.w2(swiglu_bf16(self.w1(x), self.w3(x), self.swiglu_limit, weights))


class GroupedExperts(nn.Module):
    """This rank's routed experts as stacked FP4 weights, run with one grouped GEMM per projection.

    The stacks are buffers; each `Expert` in `experts` holds views into them, so the checkpoint still
    loads expert by expert. The forward has no host sync: tokens are sorted by expert on the GPU and the
    group sizes are passed to the GEMM as device offsets.
    """

    def __init__(self, cfg: V41Config, local: range) -> None:
        super().__init__()
        self.local, self.swiglu_limit = local, cfg.swiglu_limit
        dim, inter = cfg.dim, cfg.moe_inter_dim
        shapes = {"w1": (dim, inter), "w2": (inter, dim), "w3": (dim, inter)}
        for name, (in_features, out_features) in shapes.items():
            weight, scale = QuantLinear.empty_fp4(in_features, out_features, len(local))
            self.register_buffer(f"{name}_weight", weight, persistent=False)
            self.register_buffer(f"{name}_scale", scale, persistent=False)
        self.experts = [
            Expert(dim, inter, cfg.swiglu_limit, fp4=True, storage={n: self._stack(n, i) for n in shapes})
            for i in range(len(local))
        ]

    def _stack(self, name: str, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return getattr(self, f"{name}_weight")[i], getattr(self, f"{name}_scale")[i]

    def _weight(self, name: str) -> torch.Tensor:
        """[n_local, in, out] bf16, the layout `torch._grouped_mm` expects for its second operand.

        Dequantized a few experts at a time so the fp32 temporaries stay small next to the bf16 stack.
        """
        packed, scale = getattr(self, f"{name}_weight"), getattr(self, f"{name}_scale")
        out = torch.empty(*packed.shape[:-1], packed.size(-1) * 2, dtype=torch.bfloat16, device=packed.device)
        chunk = max(1, DEQUANT_CHUNK_VALUES // out[0].numel())
        for start in range(0, packed.size(0), chunk):
            rows = slice(start, start + chunk)
            dequant_fp4(packed[rows], scale[rows], out[rows])
        return out.transpose(-2, -1)

    def forward(self, x: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Sum over each token's local experts of weight * expert(x); [T, dim] fp32, 0 for non-local experts."""
        T, k = indices.shape
        n_local = len(self.local)
        expert = (indices - self.local.start).flatten()
        is_local = (expert >= 0) & (expert < n_local)
        expert = expert.masked_fill(~is_local, n_local)  # non-local choices sort past every group
        order = expert.argsort(stable=True)
        counts = torch.zeros(n_local + 1, dtype=torch.int32, device=x.device)
        counts.scatter_add_(0, expert, torch.ones_like(expert, dtype=torch.int32))
        offs = counts[:n_local].cumsum(0, dtype=torch.int32)

        xg = x[order // k]
        gate = torch._grouped_mm(xg, self._weight("w1"), offs=offs)
        up = torch._grouped_mm(xg, self._weight("w3"), offs=offs)
        h = swiglu_bf16(gate, up, self.swiglu_limit, weights.flatten()[order, None])
        y = torch._grouped_mm(h, self._weight("w2"), offs=offs)
        # Un-sort and add each token's choices in slot order: deterministic and independent of the batch.
        # Rows past the last group (non-local choices) are not written by the GEMM and must not be read.
        if y.is_cuda:
            return combine_cuda(y, order, is_local, k)
        y = torch.where(is_local[order, None], y.float(), 0.0)
        return torch.empty_like(y).index_copy_(0, order, y).view(T, k, -1).sum(1)


class Gate(nn.Module):
    def __init__(self, cfg: V41Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.weight = nn.Parameter(torch.empty(cfg.n_routed_experts, cfg.dim))
        self.bias = nn.Parameter(torch.empty(cfg.n_routed_experts, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        if x.is_cuda and x.dtype == self.weight.dtype == torch.bfloat16:
            # Tensor-core bf16 GEMM with fp32 output, as in DeepInfer; the fp32 GEMM runs on SIMT cores.
            scores = torch.mm(x, self.weight.t(), out_dtype=torch.float32)
        else:
            scores = F.linear(x.float(), self.weight.float())
        if cfg.score_func == "softmax":
            scores = scores.softmax(-1)
        elif cfg.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = F.softplus(scores).sqrt()
        indices = (scores + self.bias).topk(cfg.n_activated_experts, dim=-1).indices  # bias selects, never scales
        weights = scores.gather(1, indices)
        if cfg.n_activated_experts > 1:
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
        return weights * cfg.route_scale, indices


class MoE(nn.Module):
    """Routed experts are split evenly across ranks (None marks another rank's expert); each rank routes
    the tokens of all ranks through its own experts. The shared expert is replicated."""

    def __init__(self, cfg: V41Config) -> None:
        super().__init__()
        world_size, rank = world()
        assert cfg.n_routed_experts % world_size == 0
        n_local = cfg.n_routed_experts // world_size
        self.local = range(rank * n_local, (rank + 1) * n_local)
        self.gate = Gate(cfg)
        self.routed = GroupedExperts(cfg, self.local)
        self.experts = nn.ModuleList(
            self.routed.experts[i - self.local.start] if i in self.local else None for i in range(cfg.n_routed_experts)
        )
        self.shared_experts = Expert(cfg.dim, cfg.moe_inter_dim, cfg.swiglu_limit, fp4=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sharded = world()[0] > 1
        xs = all_gather_rows(x) if sharded else x
        weights, indices = self.gate(xs)
        y = self.routed(xs, weights, indices)
        if sharded:
            y = reduce_scatter_rows(y, x.size(0))
        y += self.shared_experts(x)
        return y.type_as(x)


class Engram(nn.Module):
    """Adds an n-gram table lookup into every hyper-connection copy, gated by its match with that copy."""

    def __init__(self, cfg: V41Config, layer_id: int) -> None:
        super().__init__()
        self.hash_index = cfg.engram_layer_ids.index(layer_id)
        self.dim, self.hc, self.eps = cfg.dim, cfg.hc_mult, cfg.norm_eps
        n_cols = (cfg.engram_max_ngram_size - 1) * cfg.engram_n_heads
        self.embed = EngramTable(cfg.engram_num_embeddings[self.hash_index], cfg.engram_head_dim)
        self.wkv = QuantLinear(n_cols * cfg.engram_head_dim, cfg.dim * (cfg.hc_mult + 1))
        self.q_weight = nn.Parameter(torch.ones(cfg.hc_mult, cfg.dim))
        self.k_weight = nn.Parameter(torch.ones(cfg.hc_mult, cfg.dim))

    def forward(self, x: torch.Tensor, hashes: torch.Tensor) -> torch.Tensor:
        kv = self.wkv(self.embed(hashes[:, self.hash_index]).flatten(-2))
        key, value = kv.split([self.hc * self.dim, self.dim], dim=-1)
        key = key.float().unflatten(-1, (self.hc, self.dim))
        h = x.float()
        rstd = torch.rsqrt(h.square().mean(-1) + self.eps) * torch.rsqrt(key.square().mean(-1) + self.eps)
        dot = (h * (self.q_weight.float() * self.k_weight.float()) * key).sum(-1) * rstd * self.dim**-0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))  # signed sqrt, as in training
        return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)


class Block(nn.Module):
    """Attention + MoE on a residual stream carried as `hc_mult` copies (Hyper-Connections).

    Each sublayer's `hc_mixes` produce the collapse weights used by the *next* sublayer.
    """

    def __init__(self, cfg: V41Config, layer_id: int, freqs: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.cfg = cfg
        self.attn = Attention(cfg, layer_id, freqs)
        self.ffn = MoE(cfg)
        self.engram = Engram(cfg, layer_id) if layer_id in cfg.engram_layer_ids else None
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        mix = (2 + cfg.hc_mult) * cfg.hc_mult
        for part in ("attn", "ffn"):
            setattr(self, f"hc_{part}_fn", nn.Parameter(torch.empty(mix, cfg.hc_mult * cfg.dim, dtype=torch.float32)))
            setattr(self, f"hc_{part}_base", nn.Parameter(torch.empty(mix, dtype=torch.float32)))
            setattr(self, f"hc_{part}_scale", nn.Parameter(torch.empty(3, dtype=torch.float32)))

    def hc_mixes(
        self, x: torch.Tensor, fn: torch.Tensor, scale: torch.Tensor, base: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        xf = x.flatten(1).float()
        mixes = F.linear(xf, fn) * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.cfg.norm_eps)
        return hc_split_sinkhorn(mixes, scale, base, self.cfg.hc_mult, self.cfg.hc_sinkhorn_iters, self.cfg.hc_eps)

    @staticmethod
    def hc_pre(x: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
        if (x.is_cuda and x.is_contiguous() and pre.is_contiguous()
                and x.dtype == torch.bfloat16 and pre.dtype == torch.float32 and 0 < x.shape[1] <= 8):
            return collapse_cuda(x, pre)
        return (pre.unsqueeze(-1) * x.float()).sum(1).to(x.dtype)

    @staticmethod
    def hc_post(x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
        if (x.is_cuda and all(t.is_contiguous() for t in (x, residual, post, comb))
                and x.dtype == residual.dtype == torch.bfloat16 and post.dtype == comb.dtype == torch.float32
                and 0 < residual.shape[1] <= 8):
            return expand_cuda(x, residual, post, comb)
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + (comb.unsqueeze(-1) * residual.unsqueeze(-2)).sum(1)
        return y.type_as(x)

    def forward(
        self, x: torch.Tensor, pre_mix: torch.Tensor, batch: V41Batch, cache: V41Cache, ctx: StepContext
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """x [T, hc, dim] -> (x, collapse weights [T, hc] for the next block's attention)."""
        residual = x
        attn_pre, attn_post, attn_comb = self.hc_mixes(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        h = self.attn(self.attn_norm(self.hc_pre(x, pre_mix)), batch, cache, ctx)
        x = self.hc_post(h, residual, attn_post, attn_comb)

        residual = x
        ffn_pre, ffn_post, ffn_comb = self.hc_mixes(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        h = self.ffn(self.ffn_norm(self.hc_pre(x, attn_pre)))
        return self.hc_post(h, residual, ffn_post, ffn_comb), ffn_pre


class DeepseekV41(nn.Module):
    """embed -> expand to hc_mult copies -> blocks -> collapse -> norm. Logits via `compute_logits`."""

    def __init__(self, cfg: V41Config, max_positions: int, tokenizer_path: str) -> None:
        super().__init__()
        self.cfg = cfg
        freqs = {
            # Sliding-window layers use plain RoPE; YaRN (original_seq_len > 0) applies to compressed layers only.
            "window": precompute_freqs_cis(cfg.rope_head_dim, max_positions, base=cfg.rope_theta),
            "compress": precompute_freqs_cis(
                cfg.rope_head_dim,
                max_positions,
                base=cfg.compress_rope_theta,
                original_seq_len=cfg.original_seq_len,
                factor=cfg.rope_factor,
                beta_fast=cfg.beta_fast,
                beta_slow=cfg.beta_slow,
            ),
        }
        freqs = {k: v.to(torch.get_default_device()) for k, v in freqs.items()}
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList(Block(cfg, i, freqs) for i in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.head = nn.Parameter(torch.empty(cfg.vocab_size, cfg.dim, dtype=torch.float32))  # fp32 logits directly
        self.hasher = EngramHasher(cfg, tokenizer_path) if cfg.engram_layer_ids else None
        self.cache: V41Cache | None = None

    def forward(self, batch: V41Batch) -> torch.Tensor:
        assert self.cache is not None, "KV cache not allocated"
        ctx = build_step_context(batch, self.cfg.window_size)
        hashes = self.hasher(batch.ngram_ids) if self.hasher is not None else None
        h = self.embed(batch.input_ids).unsqueeze(1).repeat(1, self.cfg.hc_mult, 1)
        pre_mix = torch.zeros(h.size(0), self.cfg.hc_mult, device=h.device)
        pre_mix[:, 0] = 1.0
        for layer in self.layers:
            if layer.engram is not None:
                h = layer.engram(h, hashes)
            h, pre_mix = layer(h, pre_mix, batch, self.cache, ctx)
        return self.norm(Block.hc_pre(h, pre_mix))

    def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden.float(), self.head)

    def checkpoint_select(self, name: str) -> bool | slice:
        """Which part of a checkpoint tensor this rank holds (see `weights.iterate_raw`)."""
        parts = name.split(".")
        if parts[0] == "layers" and parts[2:4] == ["ffn", "experts"]:
            return int(parts[4]) in self.layers[int(parts[1])].ffn.local
        if parts[0] == "layers" and parts[2:4] == ["engram", "embed"]:
            table = self.layers[int(parts[1])].engram.embed
            return slice(table.start, table.start + table.part)
        return True

    @torch.no_grad()
    def cache_non_expert_weights(self, budget_bytes: int) -> int:
        """Materialize FP8 linear weights in module order within a hard budget, before graph capture.

        Routed FP4 experts and FP8 embedding tables keep their original storage/lookup paths.
        """
        if budget_bytes < 0:
            raise ValueError("weight cache budget must be nonnegative")
        used = 0
        for module in self.modules():
            if isinstance(module, QuantLinear):
                module._cached_weight = None
                size = module.weight.numel() * 2
                if module.weight.dtype == torch.float8_e4m3fn and used + size <= budget_bytes:
                    module._cached_weight = module._dequant()
                    used += size
        return used

    @torch.no_grad()
    def load_weights(self, weights) -> None:
        params = dict(self.named_parameters())
        loaded = set()
        for name, tensor in weights:
            name = {"head.weight": "head"}.get(name, name)
            if name not in params:
                raise KeyError(f"unexpected checkpoint tensor {name}")
            params[name][: tensor.size(0)].copy_(tensor)  # the last shard of a row-sharded table may be short
            loaded.add(name)
        missing = set(params) - loaded
        if missing:
            raise ValueError(f"missing weights: {sorted(missing)[:8]}")
        for module in self.modules():
            if isinstance(module, QuantLinear) and module._cached_weight is not None:
                # Preserve graph input addresses when refreshing previously cached weights.
                module._cached_weight.copy_(module._dequant())
