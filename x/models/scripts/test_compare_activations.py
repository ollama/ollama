import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

import compare_activations


class CompareActivationsTest(unittest.TestCase):
    def test_reports_first_position_and_ranked_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            want = root / "want.safetensors"
            got = root / "got.safetensors"
            report_json = root / "report.json"
            report_md = root / "report.md"

            save_file(
                {
                    "layers.0": torch.zeros((1, 4, 2), dtype=torch.float32),
                    "layers.1": torch.ones((1, 4, 2), dtype=torch.float32),
                    "skip.me": torch.zeros((1, 1), dtype=torch.float32),
                },
                want,
            )
            save_file(
                {
                    "layers.0": torch.tensor(
                        [[[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]],
                        dtype=torch.float32,
                    ),
                    "layers.1": torch.ones((1, 4, 2), dtype=torch.float32),
                    "skip.me": torch.ones((1, 1), dtype=torch.float32),
                },
                got,
            )

            rc = compare_activations.run(
                [
                    "--got",
                    str(got),
                    "--want",
                    str(want),
                    "--filter",
                    "layers.*",
                    "--atol",
                    "0.5",
                    "--rtol",
                    "0.0",
                    "--json-output",
                    str(report_json),
                    "--markdown-output",
                    str(report_md),
                ]
            )

            self.assertEqual(rc, 0)
            report = json.loads(report_json.read_text(encoding="utf-8"))
            self.assertEqual(report["counts"], {"fail": 1, "pass": 1})
            by_name = {result["name"]: result for result in report["results"]}
            self.assertEqual(by_name["layers.0"]["first_tol"], 2)
            self.assertEqual(by_name["layers.0"]["worst_pos"], 2)
            self.assertEqual(by_name["layers.1"]["status"], "pass")
            self.assertIn("layers.0", report_md.read_text(encoding="utf-8"))

    def test_require_pass_returns_nonzero_for_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            want = root / "want.safetensors"
            got = root / "got.safetensors"
            report_json = root / "report.json"
            save_file({"x": torch.zeros((1, 1), dtype=torch.float32)}, want)
            save_file({"x": torch.ones((1, 1), dtype=torch.float32)}, got)

            rc = compare_activations.run([
                "--got",
                str(got),
                "--want",
                str(want),
                "--json-output",
                str(report_json),
                "--require-pass",
            ])

            self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
