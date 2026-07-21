import json
import pathlib
import tempfile
import unittest

import summarize_validation


def write_json(path: pathlib.Path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


class SummarizeValidationTest(unittest.TestCase):
    def test_writes_reviewer_report(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            manifest = root / "porting_manifest.json"
            activation = root / "activations.manifest.json"
            comparison = root / "activation-comparison.json"
            go_test = root / "go-test.json"
            ppl = root / "ppl.json"
            transcript = root / "generation.md"
            output = root / "report.md"

            write_json(
                manifest,
                {
                    "models": [
                        {
                            "label": "TinyModel",
                            "source": "models/TinyModel",
                            "architectures": ["TinyForCausalLM"],
                            "model_type": "tiny",
                            "safetensors": {"dtype_histogram": {"BF16": 2}},
                            "risk_flags": [{"name": "rope", "detail": "rope_theta"}],
                        }
                    ],
                    "config_diffs": {"hidden_size": {"small": 8, "large": 16}},
                },
            )
            write_json(
                activation,
                {
                    "output_path": "/tmp/ollama_ref/Tiny/activations.safetensors",
                    "model_path": "models/TinyModel",
                    "model_class": "TinyForCausalLM",
                    "dtype": "bfloat16",
                    "num_tokens": 12,
                    "prefill_num_tokens": 1024,
                    "decode_text": " the",
                    "decode_token_ids": [],
                    "prompt_sha256": "abc123",
                    "tensors": {"input_ids": {"shape": [1, 12]}},
                },
            )
            write_json(
                comparison,
                {
                    "got": "/tmp/ollama_ref/Tiny/eager.safetensors",
                    "want": "/tmp/ollama_ref/Tiny/sdpa.safetensors",
                    "filters": ["layers.*"],
                    "axis": 1,
                    "counts": {"fail": 1, "pass": 3},
                    "top_absolute": [
                        {
                            "name": "layers.2",
                            "max_diff": 1.25,
                            "first_tol": 9,
                        }
                    ],
                },
            )
            go_test.write_text(
                "\n".join(
                    [
                        json.dumps({"Action": "pass", "Package": "p", "Test": "TestForward"}),
                        json.dumps({"Action": "skip", "Package": "p", "Test": "TestLong"}),
                    ]
                ),
                encoding="utf-8",
            )
            write_json(
                ppl,
                {
                    "model": "tiny:bf16",
                    "mode": "harness",
                    "max_length": 128,
                    "total_tokens": 42,
                    "token_perplexity": 9.5,
                    "baseline_delta": {
                        "token_perplexity_abs": 0.1,
                        "token_perplexity_rel": 0.0106,
                    },
                },
            )
            transcript.write_text("Prompt: hello\nOutput: world\n", encoding="utf-8")

            rc = summarize_validation.run(
                [
                    "--manifest",
                    str(manifest),
                    "--activation-manifest",
                    str(activation),
                    "--activation-comparison-json",
                    str(comparison),
                    "--go-test-json",
                    str(go_test),
                    "--ppl-json",
                    str(ppl),
                    "--generation-transcript",
                    str(transcript),
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(rc, 0)
            report = output.read_text(encoding="utf-8")
            self.assertIn("TinyModel", report)
            self.assertIn("cached prefill tokens", report)
            self.assertIn("TestLong", report)
            self.assertIn("layers.2", report)
            self.assertIn("token PPL", report)
            self.assertIn("Prompt: hello", report)


if __name__ == "__main__":
    unittest.main()
