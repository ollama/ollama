# Vision Modelfile Guidelines

## Objectives
Provide consistent patterns for constructing Modelfiles that reliably expose vision capability while optimizing memory and performance.

## Core Sections
| Directive | Purpose | Notes |
| --------- | ------- | ----- |
| `FROM` | Base model (vision-capable) | Ensure latest compatible variant |
| `MODEL` | Additional projector or adapter | Only if separate projector GGUF needed |
| `ADAPTER` | LoRA / fine-tune layers | Optional; confirm compatibility |
| `PARAMETER` | Runtime configuration (ctx, gpu, etc.) | Keep minimal; profile empirically |
| `SYSTEM` | System prompt | Optional; keep concise |
| `TEMPLATE` | Explicit chat template | Include image placeholders if custom |

## Example: Referencing the Exact Model Under Test
```modelfile
FROM hf.co/unsloth/Mistral-Small-3.1-24B-Instruct-2503-GGUF:Q3_K_XL
PARAMETER num_ctx 8192
PARAMETER num_gpu 1
PARAMETER temperature 0.7
SYSTEM You are a concise multimodal assistant.
```

## Example 2: Model + Separate Projector
```modelfile
FROM mllama-base
MODEL mllama-projector.gguf
PARAMETER num_ctx 8192
PARAMETER num_gpu 1
PARAMETER temperature 0.6
SYSTEM You analyze images and text with factual accuracy.
```

## Custom Template (If Needed) with The Same Model
```modelfile
FROM hf.co/unsloth/Mistral-Small-3.1-24B-Instruct-2503-GGUF:Q3_K_XL
TEMPLATE {{ range .Messages }}{{ if eq .Role "system" }}<|system|>{{ .Content }}{{ end }}{{ if eq .Role "user" }}<|user|>{{ .Content }}{{ end }}{{ if eq .Role "assistant" }}<|assistant|>{{ .Content }}{{ end }}{{ end }}
PARAMETER num_ctx 6144
PARAMETER temperature 0.5
```
If the runtime expects images expansion (e.g. `{{ .Images }}`), ensure your template or internal prompt handling path includes it. When unsure, prefer not to override the default template so built-in multimodal handling remains intact.

## Parameter Guidance
| Parameter | Impact | Recommendation |
| --------- | ------ | -------------- |
| `num_ctx` | Context window length | Use baseline (4k–8k) unless model supports larger windows |
| `num_gpu` | GPU layers offloaded | Start with `1`; increase if GPU memory allows |
| `temperature` | Sampling diversity | 0.6–0.8 typical | 
| `repeat_penalty` | Reduces repetition | 1.05–1.15 if needed |
| `top_p` / `top_k` | Sampling nucleus / cutoff | Tune after correctness verified |

## Validation Checklist
| Item | Goal |
| ---- | ---- |
| `ollama show <model> --json` lists `vision` | Capability present |
| Vision inference returns tokens | Functional multimodal pipeline |
| Baseline latency acceptable | Performance validated |
| No unexpected memory OOM | Resource fit |

## Common Pitfalls
| Issue | Cause | Mitigation |
| ----- | ----- | ---------- |
| Vision missing | Overridden template removed image handling | Avoid unnecessary template override |
| High memory | Excessive context / gpu layers | Reduce `num_ctx`, adjust `num_gpu` |
| Slow warm-up | Large projector or tiles | Preload once, reuse container |

## Change Control
Track Modelfile revisions in version control; annotate with reason (e.g. "Reduced ctx from 8192→6144 to avoid VRAM pressure").

## Next Steps
After stabilizing Modelfile, integrate into bisect tests to ensure capability detection unaffected by prompt layer changes.
