============================
## Quick Reference (Build, Run & Image Inventory)

This section centralizes the 3 things you asked for:
1. How to BUILD the Docker image (clean, reproducible).
2. How to RUN the container (baseline & debug vision).
3. How to IDENTIFY the last working image (digest, creation date, commit / code revision, tag) and keep an inventory.

### 1. Build (local GPU image)
Clean build pulling latest base layers and fixing architecture / CUDA arches (adjust `CUDA_ARCHES` if your GPU differs; 86 covers Ampere e.g. RTX 30xx):

```powershell
Set-Location C:\IA\test\ollama\ollama
$Env:DOCKER_BUILDKIT = "1"  # optional but faster & reproducible caching metadata
docker build --pull `
  --platform linux/amd64 `
  --build-arg FLAVOR=amd64 `
  --build-arg CUDA=1 `
  --build-arg CUDA_ARCHES=86 `
  --build-arg CMAKE_BUILD_PARALLEL_LEVEL=4 `
  -t ollama:local-gpu .
```

Record the git commit you built from (inside repo):
```powershell
git rev-parse HEAD > build_commit.txt
Get-Content build_commit.txt
```

If you want a tag embedding the short commit for traceability:
```powershell
$commit=(git rev-parse --short HEAD)
docker tag ollama:local-gpu ollama:local-gpu-$commit
```

### 2. Run (baseline vs debug vision)

Baseline minimal (no extra debug noise):
```powershell
$OLLAMA_HOME="$env:USERPROFILE\.ollama"
mkdir $OLLAMA_HOME -Force | Out-Null
docker run -d --gpus all `
  -p 11434:11434 `
  -e OLLAMA_HOST=0.0.0.0:11434 `
  -v "$OLLAMA_HOME:/root/.ollama" `
  --name ollama-baseline `
  ollama:local-gpu
```

Debug (vision focus) with verbose logging and external models dir:
```powershell
docker run -d --gpus all `
  -p 11434:11434 `
  -e OLLAMA_MODELS=/models `
  -e OLLAMA_DEBUG=2 `
  -e GIN_MODE=release `
  -v "C:\Users\iosuc\.ollama\models:/models" `
  --name ollama-vision-debug `
  ollama:local-gpu
```

Follow logs (look for vision projector / image count):
```powershell
docker logs -f --tail=200 ollama-vision-debug
```

Test a vision request (replace IMAGE_PATH with a local file). This uses your real model ONLY:
```powershell
$img=[Convert]::ToBase64String([IO.File]::ReadAllBytes("C:\path\to\image.jpg"))
$body = @{ model = 'hf.co/unsloth/Mistral-Small-3.1-24B-Instruct-2503-GGUF:Q3_K_XL'; messages = @(@{role='user';content=@(@{type='image';image="$img"}, @{type='text';text='Describe the image briefly.'})}) } | ConvertTo-Json -Depth 6
Invoke-RestMethod -Method Post -Uri http://localhost:11434/api/chat -Body $body -ContentType 'application/json'
```

### 3. Identify LAST working image & keep inventory

Images you already referenced (candidate digests):
- sha256:4beecd86698d97d7ccd7ef3e343d2b568dc5821904f00785044080524deff715 (labeled funciona)
- sha256:90f09096f883f0c48e3a1e5c30a2d0b8d7b2616426e5c4d086d53246cc3c5815 (labeled nofunciona)
- sha256:903c3672849e32832c5d14ba896b8226cd87acc3bd50492d271057db0e48411e (duda / test)
- sha256:2a28ab5e9c7a569535813cedf7a20aebe62d3dfb2a1e173fd3bfe2b82aa38ae5 (duda / test)

You want for each: CREATED time, TAG (if any), COMMIT (if embedded), and Vision result.

#### 3.1 Collect metadata automatically
PowerShell helper to output a CSV (adjust the digests list as needed):
```powershell
$digests = @(
  'sha256:4beecd86698d97d7ccd7ef3e343d2b568dc5821904f00785044080524deff715',
  'sha256:90f09096f883f0c48e3a1e5c30a2d0b8d7b2616426e5c4d086d53246cc3c5815',
  'sha256:903c3672849e32832c5d14ba896b8226cd87acc3bd50492d271057db0e48411e',
  'sha256:2a28ab5e9c7a569535813cedf7a20aebe62d3dfb2a1e173fd3bfe2b82aa38ae5'
)

$rows = foreach($d in $digests){
  $inspect = docker inspect $d 2>$null | ConvertFrom-Json
  if(-not $inspect){ continue }
  $cfg = $inspect[0]
  $labels = $cfg.Config.Labels
  [PSCustomObject]@{
    Digest        = $d
    CreatedUTC    = $cfg.Created
    RepoTags      = ($cfg.RepoTags -join ';')
    OCI_Revision  = $labels.'org.opencontainers.image.revision'
    OCI_Source    = $labels.'org.opencontainers.image.source'
    OCI_Version   = $labels.'org.opencontainers.image.version'
    VisionStatus  = ''  # fill manually: OK / FAIL / ?
    Notes         = ''
  }
}

$rows | Export-Csv -NoTypeInformation image_inventory.csv
$rows | Format-Table -AutoSize
```

If the images are only local by digest without repo tag, `RepoTags` may be empty. You can still reference by digest.

#### 3.2 Determine associated commit
If the build process added `org.opencontainers.image.revision`, the CSV will show it. Otherwise, for locally built images right after building record:
```powershell
git rev-parse HEAD
```
and manually copy into the table for the image you just tagged/pushed.

If none of the labels exist, you can extract a pseudo commit by entering the container (if source baked in):
```powershell
docker run --rm -it <digest_or_tag> sh -c 'grep -r "module ollama" -n . | head'
```
Adjust to locate a VERSION or embed a `COMMIT` file in future builds (add a build arg + `LABEL org.opencontainers.image.revision=$GIT_COMMIT`). Example Docker build invocation capturing commit:
```powershell
$commit=(git rev-parse HEAD)
docker build --pull --platform linux/amd64 `
  --build-arg FLAVOR=amd64 `
  --build-arg CUDA=1 `
  --build-arg CUDA_ARCHES=86 `
  -t ollama:local-gpu --label org.opencontainers.image.revision=$commit .
```

#### 3.3 Table (fill VisionStatus after testing)

| Digest (short) | Full Digest | Created (UTC) | Tag(s) | Revision | VisionStatus | Notes |
|----------------|-------------|---------------|--------|----------|-------------|-------|
| 4beecd8 | sha256:4beecd86698d97d7ccd7ef3e343d2b568dc5821904f00785044080524deff715 | (auto) | (auto) | (auto) | OK? | Labeled funciona |
| 90f0909 | sha256:90f09096f883f0c48e3a1e5c30a2d0b8d7b2616426e5c4d086d53246cc3c5815 | (auto) | (auto) | (auto) | FAIL? | Labeled nofunciona |
| 903c367 | sha256:903c3672849e32832c5d14ba896b8226cd87acc3bd50492d271057db0e48411e | (auto) | (auto) | (auto) | ? | Duda / retest |
| 2a28ab5 | sha256:2a28ab5e9c7a569535813cedf7a20aebe62d3dfb2a1e173fd3bfe2b82aa38ae5 | (auto) | (auto) | (auto) | ? | Duda / retest |

After you fill the table, the "last working image" is simply the one with VisionStatus=OK and the newest CreatedUTC (or highest sequence in your test order). Keep both earliest failing and last passing for regression boundary.

#### 3.4 Mark regression boundary
Once statuses filled:
```text
Last Known Good (LKG): <digest>
First Known Bad (FKB): <digest>
Boundary Range: LKG..FKB (exclusive of good, inclusive of bad)
```
Then map both revisions to commits (labels) and run:
```powershell
git diff <LKG_commit>..<FKB_commit> -- server/images.go server/routes.go api/types.go model/models llm/server.go integration/llm_image_test.go
```

### 4. Quickly test each image (vision)
Reusable PowerShell function (attach image + ask a question):
```powershell
function Test-Vision($containerName, $imagePath, $question){
  $b64=[Convert]::ToBase64String([IO.File]::ReadAllBytes($imagePath))
  $body = @{ model = 'hf.co/unsloth/Mistral-Small-3.1-24B-Instruct-2503-GGUF:Q3_K_XL'; messages = @(@{role='user';content=@(@{type='image';image=$b64}, @{type='text';text=$question})}) } | ConvertTo-Json -Depth 6
  try {
    Invoke-RestMethod -Method Post -Uri http://localhost:11434/api/chat -Body $body -ContentType 'application/json'
  } catch {
    Write-Host "Request failed: $($_.Exception.Message)" -ForegroundColor Red
  }
}
```
For each container (already started on a distinct port if parallel) call:
```powershell
Test-Vision -containerName ollama-vision-funciona -imagePath C:\path\img.jpg -question 'What is in the picture?'
```

Add the result summary (response / error) into the table.

### 5. Decide the LAST working image
When table is filled, sort by CreatedUTC descending where VisionStatus=OK to identify the last working. Provide that digest + revision in future docs (LKG anchor).

---
If you share the filled CSV or table, I can generate the precise regression boundary commands next.

============================
docker build -t ollama/ollama .

docker stop ollama; docker rm ollama


<!-- Nota: El contenedor usa ENV OLLAMA_MODELS=/models, así que monta tu carpeta de modelos en /models -->
docker run -d --gpus=all -p 11434:11434 --name ollama -v "C:\Users\iosuc\.ollama\models:/models" ollama/ollama

docker build -t ollama/ollama:cuda `
  --build-arg FLAVOR=amd64 `
  --build-arg CUDA_ARCHES="86" `
  --progress=plain .
  
  docker build -t ollama/ollama:cuda --build-arg FLAVOR=amd64 --build-arg CUDA_ARCHES="86" --progress=plain .

docker run -d --gpus=all -p 11434:11434 --name ollama -v "C:\Users\iosuc\.ollama\models:/models" ollama/ollama:cuda



docker build -t ollama/ollama:cuda --build-arg FLAVOR=amd64 --build-arg CUDA_ARCHES="86" --progress=plain .


===================
# Opcional: activar BuildKit para builds más rápidos
$Env:DOCKER_BUILDKIT = "1"

# Desde C:\IA\test\memorydoc\ollama
docker build --pull `
  --platform linux/amd64 `
  --build-arg CUDA=1 `
  --build-arg CUDA_ARCHES=86 `
  --build-arg CMAKE_BUILD_PARALLEL_LEVEL=4 `
  -t ollama:local-gpu .

  docker run -d --gpus all `
  -p 11434:11434 `
  -v "C:\Users\iosuc\.ollama\models:/models" `
  --name ollama-gpu `
  ollama:local-gpu



  =======================================
  docker build --pull --platform linux/amd64 `
  --build-arg FLAVOR=amd64 `
  --build-arg CUDA_ARCHES=86 `
  --progress=plain `
  -t ollama:local-gpu .

  docker rm -f ollama-gpu 2>$null

docker run -d --gpus all `
  -p 11434:11434 `
  -e OLLAMA_MODELS=/models `
  -e GIN_MODE=release `
  -e OLLAMA_DEBUG=2 `
  -v "C:\Users\iosuc\.ollama\models:/models" `
  --name ollama-gpu `
  ollama:local-gpu


  ===============================
  docker build --pull `
  --platform linux/amd64 `
  --build-arg FLAVOR=amd64 `
  --build-arg CUDA_ARCHES=86 `
  --progress=plain `
  -t ollama:local-gpu .

  $OLLAMA_HOME="$env:USERPROFILE\.ollama"
mkdir $OLLAMA_HOME -Force | Out-Null

docker run --rm -it `
  --gpus all `
  -p 11434:11434 `
  -e OLLAMA_HOST=0.0.0.0:11434 `
  -v "$OLLAMA_HOME:/root/.ollama" `
  ollama:local-gpu



  =================
  docker build --pull `
  --platform linux/amd64 `
  --build-arg FLAVOR=amd64 `
  --build-arg CUDA_ARCHES=86 `
  --progress=plain `
  -t ollama:local-gpu .
  $OLLAMA_HOME="$env:USERPROFILE\.ollama"
mkdir $OLLAMA_HOME -Force | Out-Null

docker run --rm -it `
  --gpus all `
  -p 11434:11434 `
  -e OLLAMA_HOST=0.0.0.0:11434 `
  -v "$OLLAMA_HOME:/root/.ollama" `
  ollama:local-gpu



  ==========
  docker build --pull `
>>   --platform linux/amd64 `
>>   --build-arg FLAVOR=amd64 `
>>   --build-arg CUDA_ARCHES=86 `
>>   --progress=plain `
>>   -t ollama:local-gpu .

docker build --pull --platform linux/amd64 --build-arg FLAVOR=amd64 --build-arg CUDA_ARCHES=86 --progress=plain -t ollama:local-gpu .

docker run -d --gpus all `  -p 11434:11434 `  -e OLLAMA_MODELS=/models `  -e GIN_MODE=release `  -e OLLAMA_DEBUG=2 `  -v "C:\Users\iosuc\.ollama\models:/models" `  --name ollama-gpu `  ollama:local-gpu


test vision
docker run -d --gpus all `
  -p 11434:11434 `
  -e OLLAMA_MODELS=/models `
  -e GIN_MODE=release `
  -e OLLAMA_DEBUG=2 `
  -v "C:\Users\iosuc\.ollama\models:/models" `
  --name ollama-vision-funciona `
  sha256:4beecd86698d97d7ccd7ef3e343d2b568dc5821904f00785044080524deff715




docker run -d --gpus all `
  -p 11434:11434 `
  -e OLLAMA_MODELS=/models `
  -e GIN_MODE=release `
  -e OLLAMA_DEBUG=2 `
  -v "C:\Users\iosuc\.ollama\models:/models" `
  --name ollama-vision-Nofunciona `
  sha256:90f09096f883f0c48e3a1e5c30a2d0b8d7b2616426e5c4d086d53246cc3c5815



docker run -d --gpus all `
  -p 11434:11434 `
  -e OLLAMA_MODELS=/models `
  -e GIN_MODE=release `
  -e OLLAMA_DEBUG=2 `
  -v "C:\Users\iosuc\.ollama\models:/models" `
  --name ollama-vision-Dudafunciona `
  sha256:903c3672849e32832c5d14ba896b8226cd87acc3bd50492d271057db0e48411e

docker run -d --gpus all `
  -p 11434:11434 `
  -e OLLAMA_MODELS=/models `
  -e GIN_MODE=release `
  -e OLLAMA_DEBUG=2 `
  -v "C:\Users\iosuc\.ollama\models:/models" `
  --name ollama-vision-Duda2funciona `
  sha256:2a28ab5e9c7a569535813cedf7a20aebe62d3dfb2a1e173fd3bfe2b82aa38ae5

docker run -d --gpus all `
  -p 11434:11434 `
  -e OLLAMA_MODELS=/models `
  -e GIN_MODE=release `
  -e OLLAMA_DEBUG=2 `
  -v "C:\Users\iosuc\.ollama\models:/models" `
  --name ollama-vision-Duda3funciona `
  sha256:2a28ab5e9c7a569535813cedf7a20aebe62d3dfb2a1e173fd3bfe2b82aa38ae5

docker run -d --gpus all `
  -p 11434:11434 `
  -e OLLAMA_MODELS=/models `
  -e GIN_MODE=release `
  -e OLLAMA_DEBUG=2 `
  -v "C:\Users\iosuc\.ollama\models:/models" `
  --name ollama-vision-Duda4funciona `
  sha256:903c3672849e32832c5d14ba896b8226cd87acc3bd50492d271057db0e48411e

============================
## Snapshot vs Upstream Diff (vision)

Compared branches: `IosuPruebas` (local working snapshot) vs `upstream/main` (HEAD after fetch).

Key files with vision / image pipeline changes:
- `server/routes.go`: adds Harmony integration, `DebugRenderOnly` flag, returns `ImageCount` in debug responses, adjusts `KeepAlive` logic, and capability selection for `gpt-oss` families.
- `server/images.go`: broadens thinking capability detection for `gpt-oss` and adds `ImageCount` in debug responses.
- `api/types.go`: new fields (e.g. `ImageCount` inside `DebugInfo`).
- `llm/server.go` & `server/sched.go`: renames sizing fields (`estimatedTotal` -> `totalSize` etc.) which may affect memory heuristics for large vision models.
- `model/models/*`: new multimodal models (gemma3, qwen25vl, mllama) and GGUF struct tag change (`gguf:"v,vision"` -> `gguf:"v"`) indicating a metadata annotation shift for vision capability.
- `integration/llm_image_test.go`: refactors to use pointers to `input.Input`; changes in special tokens and placeholder sequence (patch / tile handling evolution).
- `llm/memory.go` & `ml/backend/ggml/*`: backend updates (not fully analyzed) that could impact loading of vision components.

Relevant patterns observed:
1. Migration from value `input.Input` to pointer `*input.Input` for image / patch inputs (reduces copies, enables shared mutation). Legacy code expecting value semantics could mis-handle image counting.
2. Removal of the `,vision` suffix in GGUF struct tags for `*VisionModel`, shifting how vision capability is inferred (detection logic may have moved or broadened).
3. New debug render path (`DebugRenderOnly`) returning the rendered template + image count, useful to validate insertion of `[img-0]` or `<|image|>` markers.
4. Capability matching now treats `gptoss` and `gpt-oss` equivalently (thinking / harmony), eliminating potential mismatch if the family string varied.
5. Context length reporting no longer divides by `numParallel`, potentially altering scheduling heuristics impacting vision-enabled models.

Proposed next steps:
- Verify in the snapshot how vision is detected (search for `gguf:"v,vision"`) and confirm upstream still recognizes capability without the suffix.
- Use the new debug endpoint to compare rendered template with and without images and verify `ImageCount`.
- Identify the commit that changed `gguf:"v,vision"` to `gguf:"v"` to bound the regression window.
- Examine `integration/llm_image_test.go` diff to extract the precise token sequence for images and adjust Modelfile if required.

Preliminary regression hypothesis (vision):
"Regression may stem from the removal of the extra `vision` GGUF tag and/or the shift to pointer-based multimodal input slices causing older logic (assuming value inputs) to skip or mis-handle image tokens, resulting in missing vision processing in intermediate builds." Validate by reviewing GGUF parsing logic and the `inputs` construction function (diffs around `func (s *Server) inputs`).

Outstanding actions (see TODO): enumerate vision-related commits, diff version tags (e.g. `v0.11.3..v0.11.8`), and isolate the exact responsible commit.
