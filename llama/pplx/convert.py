#!/usr/bin/env python3
"""Convert pplx-decider-v1-27b and its readout using the pinned llama.cpp.

python llama/pplx/convert.py --llama-cpp build/llama-src MODEL --outtype q8_0 --outfile MODEL.gguf
Use --mmproj --outtype f16 to export the vision projector separately.
"""
import argparse
import json
import math
from pathlib import Path
import sys

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--llama-cpp', type=Path, required=True)
args, remaining = parser.parse_known_args()
sys.path[:0] = [str(args.llama_cpp.resolve()), str((args.llama_cpp / 'gguf-py').resolve())]

from safetensors import safe_open
from transformers import AutoTokenizer
from conversion import ModelBase, TEXT_MODEL_MAP, MMPROJ_MODEL_MAP
from conversion.qwen import Qwen3_5TextModel
from conversion.qwen3vl import Qwen3VLVisionModel
import convert_hf_to_gguf

# The release saves the backbone without the language-model wrapper.
TEXT_MODEL_MAP['Qwen3_5Model'] = 'qwen'
MMPROJ_MODEL_MAP['Qwen3_5Model'] = 'qwen3vl'
ModelBase.register('Qwen3_5Model')(Qwen3_5TextModel)
ModelBase.register('Qwen3_5Model')(Qwen3VLVisionModel)
original_filter = ModelBase.filter_tensors.__func__
original_prepare = Qwen3_5TextModel.prepare_tensors
original_metadata = Qwen3_5TextModel.set_gguf_parameters


def filter_tensors(cls, item):
    name, tensor = item
    if name.startswith('language_model.'):
        name = 'model.' + name
    return original_filter(cls, (name, tensor))


def prepare_tensors(self):
    original_prepare(self)
    with safe_open(self.dir_model / 'readout.safetensors', framework='pt', device='cpu') as head:
        data = head.get_tensor('weight').float().numpy()
        if data.shape != (255, self.hparams['hidden_size']):
            raise ValueError('Invalid PPLX readout shape')
        self.gguf_writer.add_tensor('pplx.readout.weight', data)


def set_gguf_parameters(self):
    original_metadata(self)
    config = json.loads((self.dir_model / 'decision_config.json').read_text())
    scale = config['temperature']
    if config['format_version'] != 1 or not math.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid PPLX decision configuration')
    # Validate the checkpoint's saved vocabulary against its tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(self.dir_model)
    import itertools
    import string
    candidates = list(string.ascii_uppercase) + [''.join(p) for p in itertools.product(string.ascii_uppercase, repeat=2)]
    codes = [c for c in candidates if len(tokenizer.encode(c, add_special_tokens=False)) == 1][:255]
    ids = [tokenizer.encode(c, add_special_tokens=False)[0] for c in codes]
    if len(codes) != 255 or config['codes'] != codes or config['token_ids'] != ids:
        raise ValueError('PPLX answer vocabulary does not match the tokenizer')
    self.gguf_writer.add_string(f'{self.gguf_writer.arch}.decision.type', 'pplx')
    self.gguf_writer.add_float32(f'{self.gguf_writer.arch}.decision.temperature', scale)


ModelBase.filter_tensors = classmethod(filter_tensors)
Qwen3_5TextModel.prepare_tensors = prepare_tensors
Qwen3_5TextModel.set_gguf_parameters = set_gguf_parameters
sys.argv = [sys.argv[0], *([] if '--mmproj' in remaining else ['--no-mtp']), *remaining]
convert_hf_to_gguf.main()
