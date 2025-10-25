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


=00000000000000
docker build --pull --platform linux/amd64 -t ollama:v0.11.10 .
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
