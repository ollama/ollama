# Vision Capability Regression - Reproduction Guide

## Goal
Provide a deterministic procedure to reproduce (a) working vision inference and (b) failing vision inference for comparison and later bisecting.

## Preconditions
1. Docker installed and functioning.
2. Access to both a "known good" image (snapshot) and a "suspect" / newer image.
3. Host has enough disk space for model downloads.
4. Network access to pull models (unless cached locally).

## Environment Variables (optional)
| Variable | Purpose | Recommended |
| -------- | ------- | ----------- |
| `OLLAMA_DEBUG` | Increases log verbosity | `2` during reproduction |
| `OLLAMA_NUM_PARALLEL` | Fix parallelism to reduce noise | `1` |

## Step 1: Launch Good (Baseline) Container
```powershell
docker run -d --name ollama-vision-good -p 11434:11434 `
  -e OLLAMA_DEBUG=2 <GOOD_IMAGE_DIGEST_OR_TAG>
```
Confirm it is healthy:
```powershell
docker logs --tail 50 ollama-vision-good
```

## Step 2: Pull / Verify Working Model
Model under test (fixed): `hf.co/unsloth/Mistral-Small-3.1-24B-Instruct-2503-GGUF:Q3_K_XL`.
```powershell
curl -s http://localhost:11434/api/pull -d '{"name":"hf.co/unsloth/Mistral-Small-3.1-24B-Instruct-2503-GGUF:Q3_K_XL"}' | Write-Output
```
Show capabilities:
```powershell
curl -s http://localhost:11434/api/show -d '{"name":"hf.co/unsloth/Mistral-Small-3.1-24B-Instruct-2503-GGUF:Q3_K_XL"}' | Out-File working_show.json
```
Inspect `working_show.json` for a capabilities array containing `vision`.

## Step 3: Vision Inference (Baseline)
Prepare a simple test image `test.jpg` (small, ~100KB). Send multimodal prompt (example JSON structure—adjust to actual API contract if different):
```powershell
$body = '{
  "model":"hf.co/unsloth/Mistral-Small-3.1-24B-Instruct-2503-GGUF:Q3_K_XL",
  "messages":[{"role":"user","content":[{"type":"text","text":"Describe the image succinctly"},{"type":"image","image":"file:test.jpg"}]}]
}'
curl -s http://localhost:11434/api/chat -d $body | Write-Output
```
Expected: Model returns description tokens (no vision error).

## Step 4: Launch Suspect Container
Use a different host port to run in parallel.
```powershell
docker run -d --name ollama-vision-bad -p 21434:11434 `
  -e OLLAMA_DEBUG=2 <SUSPECT_IMAGE_DIGEST_OR_TAG>
```

Tail logs in separate window:
```powershell
docker logs -f ollama-vision-bad
```

## Step 5: Pull Same Model in Suspect Container
```powershell
curl -s http://localhost:21434/api/pull -d '{"name":"hf.co/unsloth/Mistral-Small-3.1-24B-Instruct-2503-GGUF:Q3_K_XL"}' | Write-Output
curl -s http://localhost:21434/api/show -d '{"name":"hf.co/unsloth/Mistral-Small-3.1-24B-Instruct-2503-GGUF:Q3_K_XL"}' | Out-File failing_show.json
```
Compare capability arrays:
```powershell
Compare-Object (Get-Content working_show.json) (Get-Content failing_show.json) | Select-Object -First 40
```

## Step 6: Vision Inference (Suspect)
Send the same request against port 21434, saving response:
```powershell
curl -s http://localhost:21434/api/chat -d $body | Out-File failing_chat.json
```
Expected (failure scenario): Error referencing lack of vision support (or paraphrased message).

## Step 7: Collect Logs & Artifacts
```powershell
docker logs ollama-vision-good   > logs_good.txt
docker logs ollama-vision-bad    > logs_bad.txt
```
Store artifacts:
```
working_show.json
failing_show.json
failing_chat.json
logs_good.txt
logs_bad.txt
```

## Step 8: Extract Model Blob Paths
Look inside each container for model blobs (adjust path as needed):
```powershell
docker exec ollama-vision-bad  powershell -Command "Get-ChildItem -Recurse C:/root/.ollama/models/blobs | Select-Object -First 10"
```
(Linux base image may use `/root/.ollama/models/blobs`.)

## Step 9: Inspect GGUF Keys (Suspect)
If `strings` available inside container:
```powershell
docker exec ollama-vision-bad sh -c "strings /root/.ollama/models/blobs/sha256-<HASH> | grep -i vision.block_count" || echo "No key"
```
For Windows host extraction, copy blob out and run a local `strings` utility.

## Step 10: Summarize Outcome
Record in a table (see separate pass/fail template) the digest, model, presence of `vision` capability, and inference result.

## Clean Up
```powershell
docker rm -f ollama-vision-good
docker rm -f ollama-vision-bad
```

## Notes
- Maintain identical prompt and image file between containers to reduce variance.
- Set fixed random seed if the API supports it to keep responses stable (not currently shown).

## Next Step After Reproduction
If capability missing only in suspect container → proceed to metadata inspection and commit range diffs.
