import json
import pathlib
import struct
import tempfile
import unittest

import inspect_model


def write_json(path: pathlib.Path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def write_safetensors_header(path: pathlib.Path, header):
    raw = json.dumps(header).encode("utf-8")
    padding = (8 - len(raw) % 8) % 8
    raw += b" " * padding
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)


class InspectModelTest(unittest.TestCase):
    def test_writes_manifest_and_detects_risks(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            model = root / "TinyModel"
            out = root / "out"
            model.mkdir()
            write_json(
                model / "config.json",
                {
                    "architectures": ["TinyForCausalLM"],
                    "model_type": "tiny",
                    "hidden_size": 8,
                    "num_hidden_layers": 2,
                    "num_experts": 4,
                    "rope_theta": 10000,
                    "sliding_window": 16,
                    "tie_word_embeddings": True,
                    "attention_bias": True,
                    "layer_types": ["full_attention", "sliding_attention"],
                },
            )
            write_json(
                model / "tokenizer_config.json",
                {"chat_template": "{% if thinking %}<think>{{ thinking }}</think>{% endif %}"},
            )
            write_safetensors_header(
                model / "model.safetensors",
                {
                    "model.embed_tokens.weight": {
                        "dtype": "BF16",
                        "shape": [16, 8],
                        "data_offsets": [0, 256],
                    },
                    "model.layers.0.self_attn.q_proj.weight": {
                        "dtype": "F16",
                        "shape": [8, 8],
                        "data_offsets": [256, 384],
                    },
                },
            )

            rc = inspect_model.run(["--model", str(model), "--output", str(out)])
            self.assertEqual(rc, 0)
            manifest = json.loads((out / "porting_manifest.json").read_text())
            self.assertEqual(manifest["schema_version"], 1)
            self.assertEqual(manifest["models"][0]["architectures"], ["TinyForCausalLM"])
            risks = {r["name"] for r in manifest["models"][0]["risk_flags"]}
            self.assertIn("rope", risks)
            self.assertIn("sliding_window", risks)
            self.assertIn("moe", risks)
            self.assertIn("thinking_template", risks)
            self.assertEqual(
                manifest["models"][0]["safetensors"]["dtype_histogram"],
                {"BF16": 1, "F16": 1},
            )
            self.assertTrue((out / "porting_manifest.md").exists())

    def test_config_diffs_across_variants(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            models = []
            for name, hidden in [("small", 8), ("large", 16)]:
                model = root / name
                model.mkdir()
                write_json(
                    model / "config.json",
                    {
                        "architectures": ["TinyForCausalLM"],
                        "model_type": "tiny",
                        "hidden_size": hidden,
                    },
                )
                write_safetensors_header(model / "model.safetensors", {})
                models += ["--model", str(model)]
            out = root / "out"
            inspect_model.run([*models, "--output", str(out)])
            manifest = json.loads((out / "porting_manifest.json").read_text())
            self.assertIn("hidden_size", manifest["config_diffs"])


if __name__ == "__main__":
    unittest.main()
