# Quick Vision Capability Checklist

Use this rapid list before deeper debugging.

## 1. Capability Presence
[] `ollama show hf.co/unsloth/Mistral-Small-3.1-24B-Instruct-2503-GGUF:Q3_K_XL --json` includes `"vision"` in capabilities

## 2. Metadata Keys
[] `<arch>.vision.block_count` present in GGUF
[] Other `vision.*` keys (image_size, patch_size, num_channels)

## 3. Projector (If Applicable)
[] `ProjectorPaths` non-empty
[] Projector GGUF accessible (no open errors)

## 4. Request Structure
[] JSON includes image part (`type":"image"` or correct field)
[] Image file path / base64 valid

## 5. Logs
[] No `couldn't open model file` errors
[] Presence of vision encoder log line

## 6. Environment
[] Same model digest in working & failing containers
[] No conflicting overrides (custom template removing images)

## 7. Performance (Optional)
[] First inference completes under expected latency
[] GPU memory usage within limits

## 8. If Any Box Fails
→ Refer to detailed documents (`02_metadata_inspection.md`, `03_bisect_plan.md`, `05_logging_and_instrumentation.md`).
