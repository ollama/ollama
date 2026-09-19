"""Regenerate the small offline fixture with the export script's reference environment."""
import json
from pathlib import Path

import torch
from gliner.config import GLiNERConfig
from gliner.modeling.base import SpanModel
from safetensors.torch import save_file

torch.manual_seed(42)
torch.set_num_threads(1)
out = Path(__file__).parent
config = GLiNERConfig(
    hidden_size=8, max_width=3, max_len=32, class_token_index=4,
    encoder_config=dict(model_type="deberta-v2", hidden_size=16, num_hidden_layers=2,
                        num_attention_heads=4, intermediate_size=32, max_position_embeddings=32,
                        vocab_size=32, relative_attention=True, position_buckets=8,
                        max_relative_positions=-1, position_biased_input=False, type_vocab_size=0,
                        share_att_key=True, norm_rel_ebd="layer_norm", pos_att_type=["p2c", "c2p"],
                        layer_norm_eps=1e-7, hidden_act="gelu")
)
model = SpanModel(config, False).eval()
ids = torch.tensor([[1, 4, 7, 4, 9, 5] + list(range(10, 28)) + [2]])
word_indices = list(range(6, 24))
label_indices = [1, 3]
spans = [[s, s + k] for s in range(18) for k in range(3) if s + k < 18]
with torch.inference_mode():
    encoded = model.token_rep_layer(ids, torch.ones_like(ids))
    words = model.rnn(encoded[:, word_indices], torch.ones(1, 18))
    start = model.span_rep_layer.span_rep_layer.project_start(words)
    end = model.span_rep_layer.span_rep_layer.project_end(words)
    joined = torch.cat([start[:, [s for s, e in spans]], end[:, [e for s, e in spans]]], -1).relu()
    projected = model.span_rep_layer.span_rep_layer.out_project(joined)
    labels = model.prompt_rep_layer(encoded[:, label_indices])
    logits = projected @ labels.transpose(-1, -2)
save_file(model.state_dict(), str(out / "model.safetensors"))
(out / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")
(out / "reference.json").write_text(json.dumps(dict(ids=ids[0].tolist(), word_indices=word_indices,
    label_indices=label_indices, spans=spans, encoded=encoded.flatten().tolist(),
    logits=logits.flatten().tolist()), indent=2) + "\n")
