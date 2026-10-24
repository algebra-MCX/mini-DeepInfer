"""Grouped MoE experts against the per-expert loop they replace."""

from types import SimpleNamespace

import pytest
import torch

from minideepinfer.models.deepseek_v41.model import Gate, GroupedExperts

pytestmark = pytest.mark.gpu

N_EXPERTS, TOPK = 16, 4
CFG = SimpleNamespace(dim=128, moe_inter_dim=256, swiglu_limit=10.0)


def make_experts(local=range(N_EXPERTS), seed=0):
    torch.manual_seed(seed)
    with torch.device("cuda"):
        experts = GroupedExperts(CFG, local)
    for name, buf in experts.named_buffers():
        if name.endswith("_weight"):
            buf.copy_(torch.randint(-128, 128, buf.shape, dtype=torch.int8))
        else:  # E8M0 scales around 2^-4 keep the activations in a realistic range
            buf.copy_(torch.exp2(torch.randint(-6, -2, buf.shape, device="cuda").float()).to(buf.dtype))
    return experts


def loop_reference(experts, x, weights, indices):
    """The previous MoE forward: one dense call per local expert that received tokens."""
    y = torch.zeros_like(x, dtype=torch.float32)
    for i, expert in zip(experts.local, experts.experts):
        rows, slot = torch.where(indices == i)
        if rows.numel():
            y[rows] += expert(x[rows], weights[rows, slot, None])
    return y


def routing(T, n_experts=N_EXPERTS, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(T, CFG.dim, device="cuda", generator=g).bfloat16()
    indices = torch.rand(T, n_experts, device="cuda", generator=g).argsort(-1)[:, :TOPK]
    weights = torch.rand(T, TOPK, device="cuda", generator=g)
    return x, weights / weights.sum(-1, keepdim=True), indices


def assert_close(actual, expected):
    # bf16 GEMM outputs with fp32 accumulation; only the summation order of the k choices differs.
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2 * expected.abs().max().item())


@pytest.mark.parametrize("T", [1, 7, 64, 300])
def test_matches_per_expert_loop(T):
    experts = make_experts()
    x, weights, indices = routing(T)
    assert_close(experts(x, weights, indices), loop_reference(experts, x, weights, indices))


def test_skewed_routing_with_empty_experts():
    experts = make_experts()
    x, weights, _ = routing(33)
    indices = torch.arange(TOPK, device="cuda").expand(33, TOPK).contiguous()  # experts 4.. get nothing
    indices[0] = torch.tensor([15, 14, 13, 0])  # one token reaches the last experts
    assert_close(experts(x, weights, indices), loop_reference(experts, x, weights, indices))


def test_non_local_experts_contribute_zero():
    """Under expert parallelism a rank owns a slice of the experts; other ranks add the rest."""
    full = make_experts()
    halves = [make_experts(range(0, 8)), make_experts(range(8, 16))]
    for half in halves:  # same weights as the full set
        for name, buf in half.named_buffers():
            buf.copy_(getattr(full, name)[half.local.start : half.local.stop])
    x, weights, indices = routing(50)
    parts = [half(x, weights, indices) for half in halves]
    only_remote = (indices < 8).logical_not().all(-1)
    assert torch.all(parts[0][only_remote] == 0)
    assert_close(parts[0] + parts[1], loop_reference(full, x, weights, indices))


def test_rows_do_not_depend_on_batch_order():
    experts = make_experts()
    x, weights, indices = routing(64)
    perm = torch.randperm(64, device="cuda")
    y = experts(x, weights, indices)
    y_perm = experts(x[perm], weights[perm], indices[perm])
    assert_close(y_perm, y[perm])


def test_views_share_storage_with_stacks():
    """Loading an expert's checkpoint tensors must fill the stacked buffers the GEMM reads."""
    experts = make_experts()
    experts.experts[3].w2.weight.data.zero_()
    assert torch.all(experts.w2_weight[3] == 0) and torch.any(experts.w2_weight[2] != 0)


def test_chunked_dequant_matches_whole_stack(monkeypatch):
    from minideepinfer.models.deepseek_v41 import model
    from minideepinfer.models.deepseek_v41.weights import dequant_fp4

    experts = make_experts()
    whole = dequant_fp4(experts.w2_weight, experts.w2_scale).transpose(-2, -1)
    monkeypatch.setattr(model, "DEQUANT_CHUNK_VALUES", 3 * CFG.dim * CFG.moe_inter_dim)  # 3 experts per chunk
    assert torch.equal(experts._weight("w2"), whole)


@pytest.mark.parametrize("T,k,dim", [(1, 1, 7), (5, 6, 1030), (300, 4, 128)])
def test_combine_reads_only_local_rows_in_slot_order(T, k, dim):
    """Rows of non-local choices are never written by the GEMMs: poison them and expect no effect."""
    from minideepinfer.models.deepseek_v41.moe import combine_cuda

    g = torch.Generator(device="cuda").manual_seed(T)
    order = torch.randperm(T * k, device="cuda", generator=g)
    is_local = torch.rand(T * k, device="cuda", generator=g) < 0.5
    y = torch.randn(T * k, dim, device="cuda", generator=g).bfloat16()
    y[~is_local[order]] = float("nan")
    expected = torch.where(is_local[order, None], y.float(), 0.0)
    expected = torch.empty_like(expected).index_copy_(0, order, expected).view(T, k, dim).sum(1)
    actual = combine_cuda(y, order, is_local, k)
    assert actual.dtype == torch.float32 and actual.isfinite().all()
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_chunked_dequant_writes_the_stack_in_place():
    experts = make_experts()
    w = experts._weight("w1")
    assert w.dtype == torch.bfloat16 and w.shape == (N_EXPERTS, CFG.dim, CFG.moe_inter_dim)
    for i in (0, N_EXPERTS - 1):
        torch.testing.assert_close(w[i], experts.experts[i].w1.dequant().T, rtol=0, atol=0)


@pytest.mark.parametrize("score_func", ["sqrtsoftplus", "sigmoid", "softmax"])
def test_bf16_gate_matches_fp32_gate(score_func):
    # V4.1-Flash gate shape. The bf16 inputs are exact in fp32, so only the accumulation order differs.
    cfg = SimpleNamespace(dim=5120, n_routed_experts=384, n_activated_experts=6, score_func=score_func,
                          route_scale=1.5)
    torch.manual_seed(0)
    gate = Gate(cfg)
    gate.weight.data = (torch.randn_like(gate.weight) * cfg.dim**-0.5).bfloat16()  # bf16 weight, fp32 bias
    gate.bias.data = torch.randn_like(gate.bias) * 1e-2
    x = torch.randn(300, cfg.dim).bfloat16()
    expected_weights, expected_indices = gate(x)  # the fp32 path, on CPU
    weights, indices = gate.cuda()(x.cuda())
    torch.testing.assert_close(indices.cpu(), expected_indices, rtol=0, atol=0)
    torch.testing.assert_close(weights.cpu(), expected_weights, rtol=1e-4, atol=1e-5)
