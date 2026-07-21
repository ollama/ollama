#!/usr/bin/env python3
"""Dump intermediate activations from a PyTorch reference model using forward
hooks.

Loads any model that the `transformers` library can construct, runs a single
forward pass on a prompt, and saves every submodule's output tensor to a
safetensors file. The Go-side `forward_test.go` files load this reference
via `testutil.LoadReference` and compare it against the MLX implementation's
outputs to find the first layer/position where the two diverge.

The script does not assume anything about *where* the weights came from —
they just need to live in a directory that `transformers` can load.

Usage:
    # Auto-detect model class (works for any model registered with
    # AutoModelForCausalLM / AutoModelForConditionalGeneration):
    python x/models/scripts/dump_activations.py --model models/<org>/<name>

    # Filter to a subset of modules:
    python x/models/scripts/dump_activations.py --model models/<org>/<name> \\
        --filter "model.layers.*"

    # Force a specific transformers class (when auto-detection picks the
    # wrong one, or when the class isn't registered with AutoModel):
    python x/models/scripts/dump_activations.py --model models/<org>/<name> \\
        --model-class Qwen3MoeForCausalLM

Output: /tmp/ollama_ref/<model_basename>/activations.safetensors
"""

import argparse
import fnmatch
import hashlib
import inspect
import json
import os
import struct
import sys
import warnings

os.environ["TRANSFORMERS_NO_TQDM"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")


DEFAULT_PROMPT = (
    "The quick brown fox jumps over the lazy dog. A bright sunny day in the meadow. "
    "Robert Boulter is an English film, television and theatre actor. He had a guest "
    "starring role on the television series The Bill in 2000."
)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safetensors_header_sha256(path):
    with open(path, "rb") as f:
        raw_size = f.read(8)
        if len(raw_size) != 8:
            raise ValueError(f"{path}: missing safetensors header size")
        (header_size,) = struct.unpack("<Q", raw_size)
        header = f.read(header_size)
        if len(header) != header_size:
            raise ValueError(f"{path}: truncated safetensors header")
    parsed = json.loads(header.decode("utf-8"))
    parsed.pop("__metadata__", None)
    stable = json.dumps(parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(stable.encode("utf-8")).hexdigest(), len(parsed)


def model_file_digests(model_path):
    digests = {}
    for name in [
        "config.json",
        "tokenizer.json",
        "generation_config.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]:
        path = os.path.join(model_path, name)
        if os.path.exists(path):
            digests[name] = file_sha256(path)

    safetensors = []
    for name in sorted(os.listdir(model_path)):
        if not name.endswith(".safetensors"):
            continue
        path = os.path.join(model_path, name)
        header_digest, tensor_count = safetensors_header_sha256(path)
        safetensors.append({
            "file": name,
            "size_bytes": os.path.getsize(path),
            "header_sha256": header_digest,
            "tensor_count": tensor_count,
        })
    return digests, safetensors


def tensor_info(tensor):
    return {
        "dtype": str(tensor.dtype),
        "shape": [int(d) for d in tensor.shape],
        "numel": int(tensor.nelement()),
    }


def config_value(model, name):
    config = getattr(model, "config", None)
    if config is None:
        return None
    text_config_getter = getattr(config, "get_text_config", None)
    if callable(text_config_getter):
        text_config = text_config_getter()
        value = getattr(text_config, name, None)
        if value is not None:
            return value
    return getattr(config, name, None)


def accepts_forward_kwarg(model, name):
    try:
        parameters = inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return True
    return name in parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


def write_manifest(path, args, output_path, device, input_ids, save_tensors, actual_attn_implementation, prefill_input_ids=None):
    import torch
    import transformers

    config_digests, safetensors_digests = model_file_digests(args.model)
    manifest = {
        "schema_version": 1,
        "generated_by": "x/models/scripts/dump_activations.py",
        "model_path": args.model,
        "model_class": args.model_class or "auto",
        "dtype": args.dtype,
        "device": device,
        "prompt": args.prompt,
        "prompt_sha256": hashlib.sha256(args.prompt.encode("utf-8")).hexdigest(),
        "num_tokens": int(input_ids.shape[1]),
        "input_ids": input_ids.cpu().int().tolist(),
        "prefill_num_tokens": int(prefill_input_ids.shape[1]) if prefill_input_ids is not None else 0,
        "prefill_input_ids": prefill_input_ids.cpu().int().tolist() if prefill_input_ids is not None else [],
        "decode_text": args.decode_text or "",
        "decode_token_ids": args.decode_token_id or [],
        "filters": args.filters or [],
        "skip_logits": bool(args.skip_logits),
        "output_path": output_path,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "transformers_path": args.transformers_path or os.environ.get("TRANSFORMERS_PATH") or "",
        "requested_attn_implementation": args.attn_implementation or "",
        "attn_implementation": actual_attn_implementation or "",
        "use_cache": bool(args.use_cache),
        "config_file_sha256": config_digests,
        "safetensors_headers": safetensors_digests,
        "tensors": {
            name: tensor_info(tensor)
            for name, tensor in sorted(save_tensors.items())
        },
    }
    manifest_dir = os.path.dirname(path)
    if manifest_dir:
        os.makedirs(manifest_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")


def load_model_auto(model_path, dtype, attn_implementation=None, trust_remote_code=False):
    """Load model via the auto classes that ship with transformers."""
    from transformers import AutoModelForCausalLM, AutoModelForConditionalGeneration

    kwargs = {"torch_dtype": dtype, "trust_remote_code": trust_remote_code}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    for cls in [AutoModelForCausalLM, AutoModelForConditionalGeneration]:
        try:
            return cls.from_pretrained(model_path, **kwargs)
        except (ValueError, OSError):
            continue
    raise RuntimeError(
        f"Could not load model from {model_path} with auto classes; "
        f"pass --model-class CLASSNAME to specify the transformers class "
        f"explicitly."
    )


def load_model_class(model_path, dtype, class_name, attn_implementation=None, trust_remote_code=False):
    """Load model using a specific transformers class by name.

    The class must already be importable from `transformers` (i.e. either
    upstream-supported or installed via a custom transformers source on
    sys.path -- see --transformers-path / TRANSFORMERS_PATH).
    """
    import transformers

    cls = getattr(transformers, class_name, None)
    if cls is None:
        if trust_remote_code:
            from transformers.dynamic_module_utils import get_class_from_dynamic_module

            config_path = os.path.join(model_path, "config.json")
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            auto_map = config.get("auto_map", {})
            class_reference = auto_map.get("AutoModelForCausalLM") or auto_map.get("AutoModelForConditionalGeneration")
            if class_reference and class_reference.endswith("." + class_name):
                cls = get_class_from_dynamic_module(
                    class_reference,
                    model_path,
                    local_files_only=True,
                )
        if cls is None:
            raise RuntimeError(
                f"transformers has no class named {class_name!r}. "
                f"If this is a model that requires custom code, pass "
                f"--trust-remote-code or set --transformers-path to a "
                f"transformers source tree that exports the class."
            )
    kwargs = {"torch_dtype": dtype, "trust_remote_code": trust_remote_code}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    return cls.from_pretrained(model_path, **kwargs)


def attach_hooks(model, filters=None):
    """Register forward hooks on all named modules.

    Returns (captures dict, list of hook handles for cleanup).
    Captures dict maps module name -> output tensor (float32, CPU).
    """
    import torch

    captures = {}
    handles = []

    for name, module in model.named_modules():
        if name == "":
            continue  # skip root module

        # Apply filters if specified
        if filters:
            if not any(fnmatch.fnmatch(name, f) for f in filters):
                continue

        def make_hook(hook_name):
            def hook(mod, inp, output):
                if isinstance(output, torch.Tensor):
                    captures[hook_name] = output.detach().float().cpu()
                elif isinstance(output, tuple) and len(output) > 0:
                    # Decoder layers return (hidden_states, ...) tuples
                    if isinstance(output[0], torch.Tensor):
                        captures[hook_name] = output[0].detach().float().cpu()
            return hook

        handles.append(module.register_forward_hook(make_hook(name)))

    return captures, handles


def find_text_model(model):
    """Navigate to the text-model submodule for multimodal architectures."""
    # Multimodal wrappers commonly nest the text decoder under
    # model.model.language_model. Plain CausalLMs expose model.model.
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model
    if hasattr(model, "model"):
        return model.model
    return model


def main():
    parser = argparse.ArgumentParser(description="Dump model activations as safetensors")
    parser.add_argument("--model", required=True, help="Path to a model directory loadable by transformers (config.json + weights)")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Input prompt")
    parser.add_argument("--decode-text", default=None,
                        help="Run --prompt as a cached prefill, then capture activations for this text "
                             "as the decode pass. Tokenized without special tokens.")
    parser.add_argument("--decode-token-id", action="append", type=int, default=[],
                        help="Token id to decode after cached prefill; can repeat. Mutually exclusive with --decode-text.")
    parser.add_argument("--output", default=None, help="Output safetensors path (default: /tmp/ollama_ref/<model>/activations.safetensors)")
    parser.add_argument("--manifest-output", default=None,
                        help="Output JSON sidecar manifest path "
                             "(default: <output>.manifest.json)")
    parser.add_argument("--model-class", default=None,
                        help="transformers class name to use for loading "
                             "(e.g. Qwen3MoeForCausalLM). Default: auto-detect "
                             "via AutoModelForCausalLM / AutoModelForConditionalGeneration.")
    parser.add_argument("--filter", action="append", dest="filters", help="fnmatch patterns for module names to capture (can repeat)")
    parser.add_argument("--transformers-path", default=None, help="Custom transformers source path")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Allow local/Hub custom model code referenced by auto_map")
    parser.add_argument("--skip-logits", action="store_true", help="Skip saving full logits tensor")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"], help="Model dtype")
    parser.add_argument("--attn-implementation", default=None,
                        help="Optional transformers attention backend to pass to from_pretrained "
                             "(for example: eager, sdpa, flash_attention_2). Pin this for "
                             "deterministic references when backends are numerically different.")
    parser.add_argument("--use-cache", action="store_true",
                        help="Run the reference forward pass with use_cache=True. "
                             "By default references use use_cache=False so layer dumps "
                             "compare the plain prefill graph.")
    parser.add_argument("--list-modules", action="store_true", help="List all module names and exit")
    args = parser.parse_args()
    if args.decode_text is not None and args.decode_token_id:
        parser.error("--decode-text and --decode-token-id are mutually exclusive")

    # Custom transformers path
    transformers_path = args.transformers_path or os.environ.get("TRANSFORMERS_PATH")
    if transformers_path:
        sys.path.insert(0, transformers_path)
        print(f"Using custom transformers: {transformers_path}", flush=True)

    import torch
    import logging
    logging.disable(logging.WARNING)

    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    dtype = dtype_map[args.dtype]

    # Output path
    model_basename = os.path.basename(os.path.normpath(args.model))
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join("/tmp", "ollama_ref", model_basename, "activations.safetensors")
    manifest_output_path = args.manifest_output or output_path + ".manifest.json"

    print(f"Model: {args.model}", flush=True)
    print(f"Loader: {args.model_class or 'auto'}", flush=True)
    if args.attn_implementation:
        print(f"Attention: {args.attn_implementation}", flush=True)
    print(f"Output: {output_path}", flush=True)
    print(f"Manifest: {manifest_output_path}", flush=True)

    # Load model
    print("Loading model...", flush=True)
    if args.model_class:
        model = load_model_class(args.model, dtype, args.model_class, args.attn_implementation, args.trust_remote_code)
    else:
        model = load_model_auto(args.model, dtype, args.attn_implementation, args.trust_remote_code)
    actual_attn_implementation = config_value(model, "_attn_implementation")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    print(f"Device: {device}", flush=True)
    if actual_attn_implementation:
        print(f"Resolved attention: {actual_attn_implementation}", flush=True)

    if args.list_modules:
        print("\nAll modules:")
        for name, mod in model.named_modules():
            if name:
                print(f"  {name} ({type(mod).__name__})")
        return

    # Tokenize
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    input_ids = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=True)["input_ids"]

    # Prepend BOS if not already present
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if bos_id is not None and input_ids[0, 0].item() != bos_id:
        input_ids = torch.cat([torch.tensor([[bos_id]]), input_ids], dim=1)

    input_ids = input_ids.to(device)
    prefill_input_ids = None
    if args.decode_text is not None or args.decode_token_id:
        prefill_input_ids = input_ids
        if args.decode_token_id:
            input_ids = torch.tensor([args.decode_token_id], dtype=torch.long, device=device)
        else:
            input_ids = tokenizer(args.decode_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
        if input_ids.shape[1] == 0:
            raise RuntimeError("decode input tokenized to an empty sequence")
        print(f"Prefill prompt: {prefill_input_ids.shape[1]} tokens", flush=True)
        print(f"Decode input: {input_ids.shape[1]} tokens", flush=True)
    else:
        print(f"Prompt: {input_ids.shape[1]} tokens", flush=True)

    captures = {}
    handles = []
    past_key_values = None
    if prefill_input_ids is not None:
        print("Running cached prefill pass...", flush=True)
        with torch.no_grad():
            prefill_outputs = model(input_ids=prefill_input_ids, use_cache=True)
        past_key_values = getattr(prefill_outputs, "past_key_values", None)
        if past_key_values is None:
            raise RuntimeError(f"{type(model).__name__} did not return past_key_values for cached prefill")

    # Attach hooks (on the full model to capture all paths)
    captures, handles = attach_hooks(model, args.filters)

    # Run forward pass
    print("Running forward pass...", flush=True)
    forward_kwargs = {"input_ids": input_ids}
    if past_key_values is not None:
        forward_kwargs["past_key_values"] = past_key_values
        forward_kwargs["use_cache"] = True
    elif accepts_forward_kwarg(model, "use_cache"):
        forward_kwargs["use_cache"] = args.use_cache
    elif args.use_cache:
        raise RuntimeError(f"{type(model).__name__}.forward does not accept use_cache")
    with torch.no_grad():
        outputs = model(**forward_kwargs)

    # Remove hooks
    for h in handles:
        h.remove()

    # Build output tensors
    save_tensors = {
        "input_ids": input_ids.cpu().int(),
    }
    if prefill_input_ids is not None:
        save_tensors["prefill_input_ids"] = prefill_input_ids.cpu().int()

    # Add logits
    if not args.skip_logits:
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        save_tensors["logits"] = logits.detach().float().cpu()

    # Add all captured activations
    for name, tensor in sorted(captures.items()):
        save_tensors[name] = tensor

    # Save
    from safetensors.torch import save_file

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # safetensors requires contiguous tensors
    for k in save_tensors:
        save_tensors[k] = save_tensors[k].contiguous()

    metadata = {
        "prompt": args.prompt[:200],
        "model_path": args.model,
        "num_tokens": str(input_ids.shape[1]),
        "dtype": args.dtype,
        "model_class": args.model_class or "auto",
    }

    save_file(save_tensors, output_path, metadata=metadata)
    write_manifest(
        manifest_output_path,
        args,
        output_path,
        device,
        input_ids,
        save_tensors,
        actual_attn_implementation,
        prefill_input_ids,
    )

    # Report
    print(f"\nSaved {len(save_tensors)} tensors to {output_path}", flush=True)
    print(f"Saved manifest to {manifest_output_path}", flush=True)
    total_bytes = 0
    for name in sorted(save_tensors.keys()):
        t = save_tensors[name]
        size = t.nelement() * t.element_size()
        total_bytes += size
        shape_str = "x".join(str(d) for d in t.shape)
        print(f"  {name:50s} {str(t.dtype):12s} [{shape_str}]", flush=True)
    print(f"\nTotal: {total_bytes / 1024 / 1024:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
