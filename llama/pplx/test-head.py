#!/usr/bin/env python3
"""Check the native linear head against NumPy, including calibration and limits."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--llama-cpp', type=Path, required=True)
parser.add_argument('--runner', type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.llama_cpp / 'gguf-py'))
import gguf

rng = np.random.default_rng(42)
weights = rng.standard_normal((255, 32)).astype(np.float32)
hidden = rng.standard_normal(32).astype(np.float32)
temperature = np.float32(2.207568021892729)
with tempfile.TemporaryDirectory() as tmp:
    model = Path(tmp) / 'readout.gguf'
    writer = gguf.GGUFWriter(model, 'qwen35')
    writer.add_float32('qwen35.decision.temperature', temperature)
    writer.add_tensor('pplx.readout.weight', weights)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    for count in (1, 2, 26, 255):
        result = subprocess.run([args.runner, model], input=json.dumps({'hidden': hidden.tolist(), 'options': count}), text=True, capture_output=True, check=True)
        expected = weights[:count].astype(np.float64) @ hidden.astype(np.float64) / temperature
        np.testing.assert_allclose(json.loads(result.stdout), expected, rtol=1e-6, atol=1e-6)
    for vector, count in ((hidden, 0), (hidden, 256), (hidden[:-1], 2)):
        result = subprocess.run([args.runner, model], input=json.dumps({'hidden': vector.tolist(), 'options': count}), text=True, capture_output=True)
        assert result.returncode != 0, 'invalid request accepted'
print('PPLX native readout matches NumPy; invalid dimensions and option counts rejected')
