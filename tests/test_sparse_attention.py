"""Indexed attention keeps masked rows, BF16 rounding, and graph replay semantics."""

import pytest
import torch

from minideepinfer.models.deepseek_v41 import attention
from minideepinfer.models.deepseek_v41.model import sparse_attention

pytestmark = pytest.mark.gpu


def table(kv: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-query keys kv [T, K, D] as a KV table and its row indices, -1 where not valid."""
    T, K = valid.shape
    idx = torch.arange(T * K, dtype=torch.int32, device=kv.device).view(T, K)
    return kv.flatten(0, 1), torch.where(valid, idx, -1)


# Head blocks of 16 or 32 and key tiles of 32: partial head blocks (33, 40), partial and single
# key tiles (1, 31, 33), and the real V4.1 shape (64 heads over 128 window + 512 compressed keys).
@pytest.mark.parametrize("heads,keys,dim", [(1, 1, 33), (3, 17, 33), (4, 129, 512), (33, 31, 64),
                                            (40, 33, 128), (64, 640, 512)])
def test_sparse_attention_matches_cpu(heads, keys, dim):
    gen = torch.Generator().manual_seed(47)
    q = torch.randn(7, heads, dim, generator=gen).bfloat16()
    kv = torch.randn(7, keys, dim, generator=gen).bfloat16()
    valid = torch.rand(7, keys, generator=gen) > 0.2
    valid[0] = False
    sink = torch.randn(heads, generator=gen)
    expected = sparse_attention(q, [table(kv, valid)], sink, dim**-0.5)
    actual = sparse_attention(q.cuda(), [table(kv.cuda(), valid.cuda())], sink.cuda(), dim**-0.5)
    torch.testing.assert_close(actual.cpu().float(), expected.float(), rtol=2e-2, atol=2e-2)
    assert actual[0].count_nonzero() == 0


# A window table and a compressed table, as a compressed layer reads them. Windows of 24 and 7 end
# mid-tile; 0 compressed keys is a pure sliding-window layer; 64 heads over 128 + 512 is V4.1.
@pytest.mark.parametrize("T,W,K,H", [(37, 128, 96, 4), (20, 24, 40, 4), (20, 7, 40, 33), (30, 16, 0, 4),
                                     (9, 128, 512, 64)])
def test_two_tables_match_cpu(T, W, K, H):
    gen = torch.Generator().manual_seed(T)
    q = torch.randn(T, H, 512, generator=gen).bfloat16()
    window, comp = torch.randn(T + 50, 512, generator=gen).bfloat16(), torch.randn(70, 512, generator=gen).bfloat16()
    win_idx = torch.randint(-1, T + 50, (T, W), generator=gen, dtype=torch.int32)
    comp_idx = torch.randint(-1, 70, (T, K), generator=gen, dtype=torch.int32)
    win_idx[0], comp_idx[0] = -1, -1  # query 0 sees no key at all: output 0
    sink = torch.randn(H, generator=gen)
    sources = [(window, win_idx), (comp, comp_idx)]
    expected = sparse_attention(q, sources, sink, 512**-0.5)
    actual = sparse_attention(q.cuda(), [(kv.cuda(), idx.cuda()) for kv, idx in sources], sink.cuda(), 512**-0.5)
    torch.testing.assert_close(actual.cpu().float(), expected.float(), rtol=2e-2, atol=2e-2)
    assert actual[0].count_nonzero() == 0


def test_query_chunks_match_one_launch(monkeypatch):
    gen = torch.Generator(device="cuda").manual_seed(3)
    q = torch.randn(11, 4, 64, device="cuda", generator=gen).bfloat16()
    kv = torch.randn(30, 64, device="cuda", generator=gen).bfloat16()
    idx = torch.randint(-1, 30, (11, 40), device="cuda", generator=gen, dtype=torch.int32)
    sink = torch.randn(4, device="cuda", generator=gen)
    expected = sparse_attention(q, [(kv, idx)], sink, 0.2)
    monkeypatch.setattr(attention, "QUERY_CHUNK", 4)  # 4 + 4 + 3 rows
    torch.testing.assert_close(sparse_attention(q, [(kv, idx)], sink, 0.2), expected, rtol=0, atol=0)


def test_sparse_attention_rounds_unnormalized_probabilities():
    """Removing BF16 probability rounding or including the sink in max changes this result."""
    q = torch.tensor([[[-0.005279541015625]]], device="cuda", dtype=torch.bfloat16)
    kv = torch.tensor([[1.5625], [0.11767578125], [0.5234375]], device="cuda", dtype=q.dtype)
    idx = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
    sink = torch.tensor([2.7068405151367188], device="cuda")
    assert sparse_attention(q, [(kv, idx)], sink, 1.0).item() == 0.12158203125


def test_sparse_attention_empty_and_strided():
    q = torch.randn(3, 2, 66, device="cuda", dtype=torch.bfloat16)[..., ::2]
    kv = torch.randn(17, 66, device="cuda", dtype=q.dtype)[..., ::2]
    idx = torch.arange(17, device="cuda", dtype=torch.int32).expand(3, 17)
    sink = torch.randn(2, device="cuda")
    expected = sparse_attention(q.cpu(), [(kv.cpu(), idx.cpu())], sink.cpu(), 0.2)
    actual = sparse_attention(q, [(kv, idx)], sink, 0.2)
    torch.testing.assert_close(actual.cpu().float(), expected.float(), rtol=2e-2, atol=2e-2)
    empty = sparse_attention(q[:0].contiguous(), [(kv.contiguous(), idx[:0].contiguous())], sink, 0.2)
    assert empty.shape == (0, 2, 33)


def test_sparse_attention_graph_reads_updated_inputs():
    q = torch.randn(7, 4, 33, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(20, 33, device="cuda", dtype=q.dtype)
    idx = torch.randint(0, 20, (7, 17), device="cuda", dtype=torch.int32)
    sink = torch.randn(4, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        sparse_attention(q, [(kv, idx)], sink, 0.2)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = sparse_attention(q, [(kv, idx)], sink, 0.2)
    torch.cuda.current_stream().wait_stream(stream)
    q.mul_(0.5)
    kv.neg_()
    idx[0] = -1
    idx[1, 1:] = -1
    sink.add_(2)
    graph.replay()
    expected = sparse_attention(q.cpu(), [(kv.cpu(), idx.cpu())], sink.cpu(), 0.2)
    torch.testing.assert_close(actual.cpu().float(), expected.float(), rtol=2e-2, atol=2e-2)
    assert actual[0].count_nonzero() == 0
