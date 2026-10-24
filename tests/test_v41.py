"""DeepSeek-V4.1 against DeepSeek's official reference implementation (`inference/model.py`).

The reference ships inside the V4.1-Flash checkpoint repo; only its `inference/` folder is fetched.
These tests need a GPU on which the reference's TileLang kernels run.
"""

import pytest
import torch

from tests.conftest import MODEL

pytestmark = pytest.mark.gpu

LOGIT_ATOL = 0.05  # bf16 accumulation-order noise observed is ~0.02
PROMPT_LENS = [37, 125, 300, 520]  # 125 crosses a 128-token block boundary while decoding
NUM_DECODE = 12


@pytest.fixture(scope="module")
def ref():
    pytest.importorskip("tilelang")
    from tools.v41_reference import load_reference_module

    torch.set_default_device("cuda")
    yield load_reference_module()
    torch.set_default_device("cpu")


@pytest.fixture(scope="module")
def golden(ref):
    from minideepinfer.config import resolve_model_path
    from tools.v41_reference import build_reference, golden_logits, make_prompts

    model, _ = build_reference(resolve_model_path(MODEL), max_seq_len=1024)
    cases = []
    for prompt in make_prompts(PROMPT_LENS):
        tokens, logits = golden_logits(model, prompt, NUM_DECODE)
        cases.append({"prompt": prompt, "tokens": tokens, "logits": logits})
    del model
    torch.set_default_device("cpu")
    torch.cuda.empty_cache()
    return cases


# Building blocks against the reference TileLang kernels.


def test_fp8_fake_quant_matches_kernel(ref):
    from minideepinfer.models.deepseek_v41.quant import fp8_fake_quant

    x = (torch.randn(64, 512, device="cuda") * torch.logspace(-3, 2, 512, device="cuda")).bfloat16()
    expected = ref.act_quant(x.clone(), 32, "ue8m0", torch.float8_e8m0fnu, True)
    assert torch.equal(fp8_fake_quant(x), expected)


@pytest.mark.parametrize("block,e4m3", [(32, False), (16, True)])
def test_fp4_fake_quant_matches_kernel(ref, block, e4m3):
    from minideepinfer.models.deepseek_v41.quant import fp4_fake_quant

    x = (torch.randn(64, 512, device="cuda") * torch.logspace(-3, 2, 512, device="cuda")).bfloat16()
    scale_dtype = torch.float8_e4m3fn if e4m3 else torch.float8_e8m0fnu
    expected = ref.fp4_act_quant(x.clone(), block, True, scale_dtype=scale_dtype)
    assert torch.equal(fp4_fake_quant(x, block, e4m3_scale=e4m3), expected)


def test_sinkhorn_matches_kernel(ref):
    from minideepinfer.models.deepseek_v41.model import hc_split_sinkhorn

    mixes = torch.randn(1, 50, 24, device="cuda")
    scale, base = torch.rand(3, device="cuda"), torch.randn(24, device="cuda")
    expected = ref.hc_split_sinkhorn(mixes, scale, base, 4, 20, 1e-6)
    for a, b in zip(hc_split_sinkhorn(mixes[0], scale, base, 4, 20, 1e-6), expected):
        torch.testing.assert_close(a, b[0], atol=1e-5, rtol=1e-5)


def test_sparse_attention_matches_kernel(ref):
    from minideepinfer.models.deepseek_v41.model import sparse_attention

    T, H, D, N, K = 40, 4, 512, 200, 96
    q = torch.randn(1, T, H, D, device="cuda").bfloat16()
    kv = torch.randn(1, N, D, device="cuda").bfloat16()
    sink = torch.randn(H, device="cuda")
    idx = torch.randint(-1, N, (1, T, K), device="cuda", dtype=torch.int32)
    idx[0, 0] = -1  # a query with no visible key must produce zeros
    expected = ref.sparse_attn(q, kv, sink, idx, D**-0.5)[0]
    ours = sparse_attention(q[0], [(kv[0], idx[0])], sink, D**-0.5)
    torch.testing.assert_close(ours.float(), expected.float(), atol=2e-2, rtol=2e-2)
    assert ours[0].abs().max() == 0


def test_candidate_blocks_match_reference(ref):
    from minideepinfer.models.deepseek_v41.model import select_candidate_blocks

    T, n = 64, 300
    scores = torch.randn(T, n, device="cuda")
    visible = torch.randint(0, n + 1, (T,), device="cuda")
    scores = scores.masked_fill(torch.arange(n, device="cuda") >= visible[:, None], float("-inf"))
    for topk_blocks, block_size in [(4, 8), (2048, 8), (3, 16)]:
        expected = ref.select_candidate_blocks(scores, visible.unsqueeze(-1), topk_blocks, block_size)
        assert torch.equal(select_candidate_blocks(scores, visible, topk_blocks, block_size), expected)


# The engine end to end.


def check_against_golden(outputs, golden_cases, captured):
    """Tokens must match up to the first step where the reference's top-2 margin is within tolerance;
    while trajectories agree, logits must match within LOGIT_ATOL."""
    for case, out, logits in zip(golden_cases, outputs, captured):
        for step, (gold, ours) in enumerate(zip(case["logits"], logits)):
            top2 = gold.topk(2).values
            if step and out.token_ids[step - 1] != case["tokens"][step - 1]:
                break
            diff = (gold - ours).abs().max().item()
            assert diff < LOGIT_ATOL, (len(case["prompt"]), step, diff)
            if top2[0] - top2[1] < 2 * LOGIT_ATOL:
                break
            assert out.token_ids[step] == case["tokens"][step], (len(case["prompt"]), step)


def run_engine(golden_cases, *, force_reference_tokens=False, **kwargs):
    from minideepinfer import LLM, SamplingParams

    args = dict(max_num_seqs=8, max_model_len=1024, max_num_batched_tokens=1024, kv_cache_memory_gb=0.5) | kwargs
    llm = LLM(MODEL, **args)
    runner = llm.llm_engine.model_runner
    # Logits keyed by the position they predict; a step recomputed after preemption overwrites its slot.
    captured = {id(c): {} for c in golden_cases}
    by_prompt = {tuple(c["prompt"]): c for c in golden_cases}
    orig = runner.execute

    @torch.inference_mode()
    def execute(sched, prev):
        handle = orig(sched, prev)
        forced = []
        for row, (seq, _) in enumerate(sched.sample_seqs):
            case = by_prompt[tuple(seq.token_ids[: seq.prompt_len])]
            captured[id(case)][seq.num_cached_tokens] = handle.logits[row].float().cpu()
            if force_reference_tokens:
                step = seq.num_cached_tokens - seq.prompt_len
                forced.append(int(case["logits"][step].argmax()))
        if force_reference_tokens:
            # ZOS reads the GPU copy for the next step; CPU postprocessing reads the pinned copy.
            handle.sampled_gpu.copy_(torch.tensor(forced, dtype=torch.int64, device="cuda"))
            handle.sampled_cpu.copy_(handle.sampled_gpu, non_blocking=True)
            handle.event.record()
        return handle

    runner.execute = execute
    sp = SamplingParams(temperature=0, max_tokens=NUM_DECODE + 1, ignore_eos=True)
    outputs = llm.generate(prompt_token_ids=[c["prompt"] for c in golden_cases], sampling_params=sp, use_tqdm=False)
    preempted = llm.llm_engine.scheduler.num_preempted
    del llm
    torch.cuda.empty_cache()
    return outputs, [[v for _, v in sorted(captured[id(c)].items())] for c in golden_cases], preempted


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({}, id="batched"),
        pytest.param({"enable_cuda_graph": False}, id="eager"),
        pytest.param({"enable_zos": False}, id="no-zos"),
        pytest.param({"max_num_batched_tokens": 128}, id="chunked-128"),
        pytest.param({"max_num_batched_tokens": 200, "max_num_seqs": 2}, id="chunked-200-two-seqs"),
    ],
)
def test_engine_matches_reference(golden, kwargs):
    outputs, captured, _ = run_engine(golden, **kwargs)
    check_against_golden(outputs, golden, captured)


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({}, id="graph"),
        pytest.param({"enable_cuda_graph": False}, id="eager"),
        pytest.param({"max_num_batched_tokens": 200, "max_num_seqs": 2}, id="chunked-200-two-seqs"),
        pytest.param({"weight_cache_memory_gb": 0.02}, id="cached-weights"),
    ],
)
def test_engine_logits_on_reference_trajectory(golden, kwargs):
    """Compare every decode position, including those after an ambiguous reference top-2 margin."""
    outputs, captured, _ = run_engine(golden, force_reference_tokens=True, **kwargs)
    for case, out, logits in zip(golden, outputs, captured, strict=True):
        assert out.token_ids == [int(step.argmax()) for step in case["logits"]]
        assert len(logits) == len(case["logits"]) == NUM_DECODE + 1
        for step, (gold, ours) in enumerate(zip(case["logits"], logits, strict=True)):
            diff = (gold - ours).abs().max().item()
            assert diff < LOGIT_ATOL, (len(case["prompt"]), step, diff)


def test_engine_matches_reference_under_preemption(golden):
    # Prompts fill exactly 10 blocks; the 125-token one then needs an 11th while decoding.
    outputs, captured, preempted = run_engine(golden, num_gpu_blocks=11)  # block 0 is reserved
    assert preempted > 0
    check_against_golden(outputs, golden, captured)
