import importlib.util
import json
from pathlib import Path

import torch
import pytest


def test_frozen_source_and_deterministic_small_model(tmp_path):
    g = generator()
    assert hasattr(g, "build_model"), "frozen-source constructor not implemented"
    ref = g.load_reference(g.DEFAULT_SOURCE)
    model = g.build_model(ref, torch.float32)
    other = g.build_model(ref, torch.bfloat16)
    assert model.config.hidden_size == 8
    assert model.config.num_attention_heads * model.config.head_dim == 16
    assert model.config.layer_types == [
        "sliding_attention",
        "full_attention",
        "sliding_attention",
    ]
    assert all(p.device.type == "cpu" for p in model.parameters())
    for name, p in model.named_parameters():
        assert torch.equal(p.to(torch.bfloat16), other.state_dict()[name]), name
    assert model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr()
    assert [float(x.self_attn.value_res_lambda) for x in model.model.layers] == [
        -1.25,
        0.75,
        1.5,
    ]
    for name in g.SOURCE_HASHES:
        (tmp_path / name).write_bytes((g.DEFAULT_SOURCE / name).read_bytes())
    p = tmp_path / "modeling_ricursive.py"
    p.write_bytes(p.read_bytes() + b"\n# changed\n")
    with pytest.raises(ValueError, match="source hash mismatch"):
        g.load_reference(tmp_path)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_observed_normal_forward_and_separate_mask_control(dtype):
    g = generator()
    assert hasattr(g, "observed_forward"), "observed forward not implemented"
    ref = g.load_reference(g.DEFAULT_SOURCE)
    model = g.build_model(ref, dtype)
    ids = torch.tensor([[2, 7, 3, 11, 5, 13]])
    normal, cache = g.observed_forward(model, ids, mode="hf_normal", use_cache=True)
    control, _ = g.observed_forward(model, ids, mode="absolute_mask_control")
    torch.testing.assert_close(normal["logits"], control["logits"])
    torch.testing.assert_close(normal["hidden"], control["hidden"])
    assert cache.get_seq_length() == 6
    traces = normal["intermediates"]
    e = traces["model.layers.0.input"]
    assert torch.equal(e, model.model.embed_tokens(ids) * model.config.embed_scale)
    u = traces["model.layers.0.input_layernorm.output"]
    alpha = model.model.layers[0].self_attn.value_res_lambda[0].sigmoid().to(dtype)
    assert torch.equal(
        traces["model.layers.0.self_attn.v_proj.input"], torch.lerp(e, u, alpha)
    )
    for i in range(3):
        prefix = f"model.layers.{i}.mlp."
        gate = traces[prefix + "gate_proj.output"]
        up = traces[prefix + "up_proj.output"]
        expected = torch.nn.functional.silu(gate.clamp(max=20)) * up.clamp(-20, 20)
        assert torch.equal(traces[prefix + "down_proj.input"], expected)
    assert any(
        (v.abs() > 20).any()
        for k, v in traces.items()
        if k.endswith("mlp.up_proj.output")
    )
    assert any(
        (v < -20).any() and (v > 20).any()
        for k, v in traces.items()
        if k.endswith("mlp.gate_proj.output")
    )
    assert type(model).forward is ref.RicursiveForCausalLM.forward
    assert type(model.model).forward is ref.RicursiveModel.forward


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mask_audit_exposes_non_square_and_padding_behavior(dtype):
    g = generator()
    assert hasattr(g, "mask_audit"), "mask audit not implemented"
    ref = g.load_reference(g.DEFAULT_SOURCE)
    audit = {x["name"]: x for x in g.mask_audit(g.build_model(ref, dtype))}
    multi = audit["cached_multi_none"]
    full = multi["masks"][1]
    assert full["source_zero_probabilities"][0, 0, 0].nonzero().flatten().tolist() == [
        0
    ]
    assert full["absolute_mask"][0, 0, 0].nonzero().flatten().tolist() == [
        0,
        1,
        2,
        3,
        4,
    ]
    assert not torch.equal(multi["normal"]["logits"], multi["control"]["logits"])
    safe = audit["cached_single_none"]
    torch.testing.assert_close(safe["normal"]["logits"], safe["control"]["logits"])
    padded = audit["padded_cached_single"]
    assert not torch.equal(padded["normal"]["logits"], padded["control"]["logits"])
    assert torch.isfinite(padded["normal"]["logits"]).all()
    local = padded["masks"][0]
    assert local["source_zero_probabilities"][0, 0, 0, 0] > 0
    assert not local["absolute_mask"][0, 0, 0, 0]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fixture_rejects_wrong_architecture_mutations(dtype):
    g = generator()
    assert hasattr(g, "mutation_audit"), "mutation controls not implemented"
    ref = g.load_reference(g.DEFAULT_SOURCE)
    results = g.mutation_audit(ref, dtype)
    names = {row["name"] for row in results}
    assert {
        "reverse_value_blend",
        "omit_input_layernorm",
        "omit_post_attention_layernorm",
        "omit_pre_feedforward_layernorm",
        "omit_post_feedforward_layernorm",
        "omit_q_norm",
        "omit_k_norm",
        "global_rope",
        "window_minus_one",
        "window_plus_one",
        "omit_attention_gate",
        "omit_embedding_scale",
        "omit_ffn_clipping",
        "omit_layer_zero_value_residual",
    } <= names
    for row in results:
        assert not torch.equal(row["normal_logits"], row["mutant_logits"]), row["name"]
        assert torch.isfinite(row["mutant_logits"]).all(), row["name"]


def test_component_controls_expose_cast_and_clamp_differences():
    g = generator()
    assert hasattr(g, "component_cases"), "component controls not implemented"
    ref = g.load_reference(g.DEFAULT_SOURCE)
    cases = g.component_cases(ref, torch.bfloat16)
    assert {
        "rmsnorm",
        "embedding_scale",
        "value_blend",
        "rotary",
        "ffn_clamp_2",
        "ffn_clamp_20",
        "gate_sigmoid",
    } <= set(cases)
    for name, alternative in [
        ("rmsnorm", "weight_before_cast"),
        ("embedding_scale", "scalar_cast_first"),
        ("value_blend", "reverse"),
        ("rotary", "unrounded_tables"),
        ("ffn_clamp_2", "symmetric_gate"),
    ]:
        assert not torch.equal(
            cases[name]["expected"], cases[name]["alternatives"][alternative]
        ), name
    assert cases["rotary"]["input"].shape[-1] == 128
    assert cases["rotary"]["positions"].max() == 2097151
    fp32 = g.component_cases(ref, torch.float32)
    for row in fp32.values():
        assert torch.isfinite(row["expected"]).all()


def test_difference_measures_actual_elements_and_rejects_shape_mismatch():
    g = generator()
    assert hasattr(g, "difference"), "difference helper not implemented"
    got = g.difference(torch.tensor([1.0, -2.0, 3.0]), torch.tensor([2.0, -2.0, 1.0]))
    assert got["max_abs"] == 2
    assert got["mean_abs"] == 1
    assert got["different_elements"] == 2
    assert got["elements"] == 3
    with pytest.raises(ValueError, match="shape"):
        g.difference(torch.ones(1, 3), torch.ones(3))


def test_generated_fixture_repeats_and_covers_safe_trajectories():
    g = generator()
    assert hasattr(g, "generate"), "fixture assembly not implemented"
    first = g.generate(g.DEFAULT_SOURCE)
    second = g.generate(g.DEFAULT_SOURCE)
    assert first == second
    assert first["schema_version"] == 1
    assert len(first["forward_cases"]) == 12
    assert len(first["mask_audit"]["float32"]) == 9
    assert len(first["mutations"]["bfloat16"]) == 14
    assert set(first["weights"]) == {"float32", "bfloat16"}
    assert {c["name"] for c in first["forward_cases"]} == {
        "prefill",
        "cached_prefill",
        "decode_4",
        "decode_5",
        "high_rope_position_prefill",
        "explicit_prefill",
    }
    for case in first["forward_cases"]:
        assert len(case["intermediates"]) > 80
        assert case["hidden"]["shape"][-1] == 8
        assert case["logits"]["shape"][-1] == 19
        assert len(case["cache"]) in (0, 3)
    assert first["provenance"]["packages"]["transformers"] == "4.57.6"
    json.dumps(first, allow_nan=False)


def test_cli_writes_identical_fixture_twice(tmp_path):
    import subprocess
    import sys
    import hashlib

    output = tmp_path / "ripple.json"
    command = [
        sys.executable,
        "-B",
        str(ROOT / "generate_ripple.py"),
        "--output",
        str(output),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    assert output.exists(), "CLI did not write a fixture"
    first = hashlib.sha256(output.read_bytes()).hexdigest()
    subprocess.run(command, check=True, capture_output=True, text=True)
    assert hashlib.sha256(output.read_bytes()).hexdigest() == first
    assert output.with_suffix(".results.json").exists()
    assert json.loads(output.read_text())["provenance"]["invocation"] == command


ROOT = Path(__file__).resolve().parent


def generator():
    path = ROOT / "generate_ripple.py"
    assert path.exists(), "reference generator not implemented"
    spec = importlib.util.spec_from_file_location("generate_ripple", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tensor_record_nonfinite_values_are_explicit_strings():
    record = generator().tensor_record(
        torch.tensor([float("-inf"), float("inf"), float("nan")])
    )
    assert record["data"] == ["-inf", "+inf", "nan"]
    json.dumps(record, allow_nan=False)


def test_tensor_record_preserves_logical_strided_order():
    g = generator()
    x = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4).transpose(0, 1)
    assert not x.is_contiguous()
    record = g.tensor_record(x)
    assert record == {
        "shape": [4, 3],
        "dtype": "bfloat16",
        "data": [0.0, 4.0, 8.0, 1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0],
    }
    assert json.loads(json.dumps(record, allow_nan=False)) == record


def test_absolute_mask_uses_key_positions_window_and_padding():
    g = generator()
    assert hasattr(g, "absolute_mask"), "absolute mask helper not implemented"
    q = torch.tensor([103, 104])
    k = torch.tensor([100, 101, 102, 103, 104])
    padding = torch.tensor([[1, 0, 1, 1, 1], [1, 1, 1, 0, 1]])
    expected = torch.tensor(
        [
            [[[False, False, True, True, False], [False, False, True, True, True]]],
            [[[False, True, True, False, False], [False, False, True, False, True]]],
        ]
    )
    assert torch.equal(g.absolute_mask(q, k, window=3, padding=padding), expected)
    assert g.absolute_mask(q, k, window=1)[0, 0].tolist() == [
        [False, False, False, True, False],
        [False, False, False, False, True],
    ]
    assert g.absolute_mask(q, k)[0, 0].tolist() == [
        [True, True, True, True, False],
        [True, True, True, True, True],
    ]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_local_support_check_detects_shared_window_error(monkeypatch, dtype):
    g = generator()
    assert hasattr(g, "validate_local_support"), (
        "independent local-mask check not implemented"
    )
    ref = g.load_reference(g.DEFAULT_SOURCE)
    model = g.build_model(ref, dtype)
    g.validate_local_support(g.mask_audit(model))
    original = ref.RicursiveAttention._apply_sliding_window_mask

    def too_narrow(self, attention_mask, q_len, kv_len, dtype, device, cache_position):
        mask = original(
            self, attention_mask, q_len, kv_len, dtype, device, cache_position
        )
        if self.layer_type != "sliding_attention":
            return mask
        query = (
            cache_position
            if cache_position is not None
            else torch.arange(kv_len - q_len, kv_len)
        )
        keys = torch.arange(kv_len, device=device)
        keep = (keys[None, :] >= query[:, None] - (self.config.sliding_window - 2))[
            None, None
        ]
        return (
            mask & keep
            if mask.dtype == torch.bool
            else mask.masked_fill(~keep, torch.finfo(dtype).min)
        )

    monkeypatch.setattr(
        ref.RicursiveAttention, "_apply_sliding_window_mask", too_narrow
    )
    with pytest.raises(ValueError, match="local mask support"):
        g.generate(g.DEFAULT_SOURCE)


def test_fixture_labels_rope_positions_and_shared_prefix_scope():
    g = generator()
    fixture = g.generate(g.DEFAULT_SOURCE)
    high = [
        c for c in fixture["forward_cases"] if c["name"] == "high_rope_position_prefill"
    ]
    assert len(high) == 2
    for case in high:
        assert case["positions"]["data"] == list(range(6))
        assert case["position_ids"]["data"] == list(range(32768, 32774))
        assert case["cache"] == []
    assert "high-position cache/mask continuation" in fixture["incomplete_gates"]
    for cases in fixture["mask_audit"].values():
        for case in cases:
            cached = case["prefix_ids"]["shape"][1] > 0
            assert case["prefix_mode"] == ("hf_normal" if cached else None)
            assert case["comparison_scope"] == (
                "shared_prefix_continuation" if cached else "prefill"
            )
