# Perplexity Decider

Perplexity's [pplx-decider-v1-27b](https://huggingface.co/perplexity-ai/pplx-decider-v1-27b) is a 27B decision model for classification, yes/no probabilities, and ordinal scoring. It accepts text and images through Ollama's System One API.

The model uses a Qwen backbone and a trained 255-option linear readout. Each question is evaluated separately. Its saved temperature calibrates the output probabilities. The runtime skips vocabulary logits and generates no answer tokens; text-generation requests are rejected.

## Usage

After creating the local model as described below:

```sh
curl http://localhost:11434/v1/systemone -d '{
  "model": "pplx-decider",
  "state": "My Stripe integration keeps failing. Please help ASAP.",
  "questions": {
    "urgency": {
      "type": "noul",
      "instructions": "Does this message express urgency?"
    },
    "routing": {
      "type": "choice",
      "instructions": "Which team should handle this request?",
      "criteria": {
        "billing": "Charges and refunds",
        "technical_support": "Integration errors",
        "sales": "Questions about buying a product"
      }
    }
  }
}'
```

Use `choice` for 1–255 named options, `noul` for a yes/no probability, and `score` for 2–255 ordered descriptions. A score is the probability-weighted level index, starting at zero. Choice order follows the request. Confidence follows Perplexity's reference implementation: the winner's margin above a uniform distribution for choices, and expected distance from the winner for scores.

State, instructions, and option descriptions can contain structured JSON. JSON member order is preserved in the prompt. Omitted or empty instructions default to “Choose the best matching option.” A null choice description uses the option key alone; omitted or empty yes/no descriptions use “No / false” and “Yes / true”.

For images, add an `images` array of base64-encoded image data alongside `state` and `questions`. Images are shared across the questions. The model's vision projector must be included when creating the model. Video is not supported by this API.

There can be up to 64 questions per request. The default context below is 8192 tokens per question, including images. Oversized inputs are rejected without truncation. Usage counts the complete input for every question, with zero generated output tokens.

## Build

This implementation uses the llama-server backend. Build the branch's native runtime before serving:

```sh
cmake -B build .
cmake --build build --parallel 8
./ollama serve
```

## Download and convert

The conversion script uses the repository's pinned llama.cpp converter with PyTorch, Transformers, safetensors, NumPy, sentencepiece, and huggingface-hub installed.

```sh
hf download perplexity-ai/pplx-decider-v1-27b \
  --revision 9ce1abcf1f00209405376b5bcc81225c8f8cf514 \
  --local-dir ~/git/models/pplx-decider-v1-27b

python llama/pplx/convert.py --llama-cpp build/_deps/llama_cpp-src \
  ~/git/models/pplx-decider-v1-27b --outtype q8_0 \
  --outfile ~/git/models/pplx-decider-v1-27b/pplx-decider-Q8_0.gguf

python llama/pplx/convert.py --llama-cpp build/_deps/llama_cpp-src \
  ~/git/models/pplx-decider-v1-27b --mmproj --outtype f16 \
  --outfile ~/git/models/pplx-decider-v1-27b/mmproj-pplx-decider-F16.gguf
```

If a source override was used for the native build, pass that directory to `--llama-cpp`. The converter keeps the small readout in float32 and writes `qwen35.decision.type = pplx` and the saved calibration temperature. Keep the readout in float32 when producing other quantizations.

Create `~/git/models/pplx-decider-v1-27b/Modelfile`:

```dockerfile
FROM ./pplx-decider-Q8_0.gguf
FROM ./mmproj-pplx-decider-F16.gguf
CAPABILITY decision
CAPABILITY vision
PARAMETER num_ctx 8192
```

Then create the local tags with the updated server running:

```sh
./ollama create pplx-decider:27b-q8_0 -f ~/git/models/pplx-decider-v1-27b/Modelfile
./ollama cp pplx-decider:27b-q8_0 pplx-decider:27b
./ollama cp pplx-decider:27b-q8_0 pplx-decider
```

The release is Apache-2.0 licensed. Include its `LICENSE` and `NOTICE` when redistributing model artifacts.

## Verification

```sh
go test ./decision ./llm ./server
cmake -S llama/server -B build/llama-server \
  -DFETCHCONTENT_SOURCE_DIR_LLAMA_CPP="$PWD/build/_deps/llama_cpp-src" \
  -DOLLAMA_PPLX_TESTS=ON
cmake --build build/llama-server --target ollama-pplx-head-test --parallel 8
python llama/pplx/test-head.py --llama-cpp build/_deps/llama_cpp-src \
  --runner build/llama-server/ollama-pplx-head-test
```
