#!/usr/bin/env python3
"""Export a GLiNER markerV0 checkpoint for Ollama's native MLX runner.

Reference environment: gliner==0.2.13 transformers==4.51.3 sentencepiece.
Python is only needed for export and reference generation, not serving.
"""

import argparse
import json
from pathlib import Path

import torch
from gliner import GLiNER
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Hugging Face model ID or local GLiNER directory")
    parser.add_argument("output", type=Path)
    parser.add_argument("--revision", help="Hugging Face commit or revision")
    parser.add_argument("--reference", action="store_true", help="write parity cases for Go integration tests")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    kwargs = {"revision": args.revision} if args.revision else {}
    model = GLiNER.from_pretrained(args.model, map_location="cpu", **kwargs).float().eval()
    config = model.config.to_dict()
    encoder = model.model.token_rep_layer.bert_layer.model.config.to_dict()
    if (config["span_mode"] != "markerV0" or config.get("labels_encoder")
            or config.get("fuse_layers") or config.get("post_fusion_schema")
            or encoder["model_type"] != "deberta-v2"
            or encoder.get("conv_kernel_size", 0) != 0):
        raise ValueError("This exporter supports uni-encoder DeBERTa GLiNER markerV0 models only")
    tokenizer = model.data_processor.transformer_tokenizer
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer with tokenizer.json is required")
    config.update(architectures=["GLiNER"], model_type="gliner", encoder_config=encoder, torch_dtype="float32")
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    tokenizer.save_pretrained(args.output)
    save_file({k: v.detach().float().contiguous() for k, v in model.model.state_dict().items()},
              str(args.output / "model.safetensors"))
    if args.reference:
        cases = []
        examples = [
            ("John works at Google in Paris.", ["person", "organization", "location"]),
            ("Élodie met José in São Paulo. 東京 is in Japan. 👋", ["person", "city", "country"]),
            ("Alice joined Acme Corp in New York on January 2, 2024.", ["person", "organization", "location", "date"]),
            ("Dr. Jean-Luc Picard paid $42.50 for café crème.", ["person", "amount", "food"]),
            ("A\u0308nne works at Ｇｏｏｇｌｅ. مرحبا بالعالم 😀", ["person", "organization"]),
        ]
        for text, labels in examples:
            words = [w[0] for w in model.data_processor.words_splitter(text)]
            batch = model.data_processor.tokenize_inputs([words], {label: i + 1 for i, label in enumerate(labels)})
            with torch.inference_mode():
                encoded = model.model.token_rep_layer(batch["input_ids"], batch["attention_mask"])[0]
                entities = model.predict_entities(text, labels)
            cases.append({"input": text, "labels": labels, "ids": batch["input_ids"][0].tolist(),
                          "encoded": encoded.flatten().tolist(), "entities": entities})
        (args.output / "reference.json").write_text(json.dumps(cases) + "\n")
        words = ["John", "Jean-Luc", "Élodie", "A\u0308nne", "Ｇｏｏｇｌｅ", "東京", "😀🤖", "مرحبا",
                 "São Paulo", "person", "  a   b  ", "▁test", "\u200btest", "\u00a0space", "ﬁancée"]
        token_cases = [{"word": w, "ids": tokenizer([w], is_split_into_words=True, add_special_tokens=False)["input_ids"]}
                       for w in words]
        (args.output / "tokenizer_reference.json").write_text(json.dumps(token_cases) + "\n")
    print(f"Exported to {args.output}. Import with: ollama create --experimental gliner -f <Modelfile>")


if __name__ == "__main__":
    main()
