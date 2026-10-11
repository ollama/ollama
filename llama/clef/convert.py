#!/usr/bin/env python3
"""Convert Cloudflare Clef with its trained joint head, using pinned llama.cpp.

Usage: python convert.py --llama-cpp build/llama-src MODEL --outtype q8_0 --outfile MODEL.gguf
For the vision projector, add --mmproj --outtype f16 and use a separate output file.
All remaining arguments are passed to convert_hf_to_gguf.py.
"""
import argparse
import json
from pathlib import Path
import sys

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--llama-cpp', type=Path, required=True)
args, remaining = parser.parse_known_args()
sys.path[:0] = [str(args.llama_cpp.resolve()), str((args.llama_cpp / 'gguf-py').resolve())]

import numpy as np
from safetensors import safe_open
from conversion.qwen import Qwen3_5TextModel
import convert_hf_to_gguf

original_tensors = Qwen3_5TextModel.prepare_tensors
original_metadata = Qwen3_5TextModel.set_gguf_parameters


def prepare_tensors(self):
    original_tensors(self)
    with safe_open(self.dir_model / 'joint_head.safetensors', framework='pt', device='cpu') as head:
        for name in head.keys():
            # Keep the small head in float32, including its scalar gates.
            data = head.get_tensor(name).float().numpy()
            self.gguf_writer.add_tensor('clef.' + name, np.atleast_1d(data))


def set_gguf_parameters(self):
    original_metadata(self)
    config = json.loads((self.dir_model / 'joint_head_config.json').read_text())
    self.gguf_writer.add_string(f'{self.gguf_writer.arch}.decision.type', 'clef')
    for name, value in config.items():
        self.gguf_writer.add_uint32(f'{self.gguf_writer.arch}.decision.{name}', value)


Qwen3_5TextModel.prepare_tensors = prepare_tensors
Qwen3_5TextModel.set_gguf_parameters = set_gguf_parameters
sys.argv = [sys.argv[0], '--no-mtp', *remaining]
convert_hf_to_gguf.main()
