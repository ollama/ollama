# API

## Endpoints

- [Generate a completion](#generate-a-completion)
- [Generate a chat completion](#generate-a-chat-completion)
- [Create a Model](#create-a-model)
- [List Local Models](#list-local-models)
- [Show Model Information](#show-model-information)
- [Copy a Model](#copy-a-model)
- [Delete a Model](#delete-a-model)
- [Pull a Model](#pull-a-model)
- [Push a Model](#push-a-model)
- [Generate Embeddings](#generate-embeddings)

## Conventions

### Model names

Model names follow a `model:tag` format. Some examples are `orca-mini:3b-q4_1` and `llama2:70b`. The tag is optional and, if not provided, will default to `latest`. The tag is used to identify a specific version. Models uploaded by users will have the format `namespace/model:tag`.

### Durations

All durations are returned in nanoseconds.

### Streaming responses

Certain endpoints stream responses as JSON objects and can optional return non-streamed responses.

<hr>

## Generate a completion

```shell
POST /api/generate
```

Generate a response for a given prompt with a provided model. This is a streaming endpoint, so there will be a series of responses. The final response object will include statistics and additional data from the request.

### Parameters

- `model`: (required) the [model name](#model-names)
- `prompt`: the prompt to generate a response for

Advanced parameters (optional):

- `format`: the format to return a response in. Currently the only accepted value is `json`
- `options`: additional model parameters listed in the documentation for the [Modelfile](./modelfile.md#valid-parameters-and-values) such as `temperature`
- `system`: system prompt to (overrides what is defined in the `Modelfile`)
- `template`: the full prompt or prompt template (overrides what is defined in the `Modelfile`)
- `context`: the context parameter returned from a previous request to `/generate`, this can be used to keep a short conversational memory
- `stream`: if `false` the response will be returned as a single response object, rather than a stream of objects
- `raw`: if `true` no formatting will be applied to the prompt. You may choose to use the `raw` parameter if you are specifying a full templated prompt in your request to the API.

#### JSON mode

Enable JSON mode by setting the `format` parameter to `json`. This will structure the response as valid JSON. See the JSON mode [example](#request-json-mode) below.

> Note: it's important to instruct the model to use JSON in the `prompt`. Otherwise, the model may generate large amounts whitespace.

### Examples

#### Generate request (Streaming)

##### Request

```shell
curl http://localhost:11434/api/generate -d '{
  "model": "llama2",
  "prompt": "Why is the sky blue?"
}'
```

##### Response

A stream of JSON objects is returned:

```json
{
  "model": "llama2",
  "created_at": "2023-08-04T08:52:19.385406455-07:00",
  "response": "The",
  "done": false
}
```

The final response in the stream also includes additional data about the generation:

- `total_duration`: time spent generating the response
- `load_duration`: time spent in nanoseconds loading the model
- `sample_count`: number of samples generated
- `sample_duration`: time spent generating samples
- `prompt_eval_count`: number of tokens in the prompt
- `prompt_eval_duration`: time spent in nanoseconds evaluating the prompt
- `eval_count`: number of tokens the response
- `eval_duration`: time in nanoseconds spent generating the response
- `context`: an encoding of the conversation used in this response, this can be sent in the next request to keep a conversational memory
- `response`: empty if the response was streamed, if not streamed, this will contain the full response

To calculate how fast the response is generated in tokens per second (token/s), divide `eval_count` / `eval_duration`.

```json
{
  "model": "llama2",
  "created_at": "2023-08-04T19:22:45.499127Z",
  "response": "",
  "done": true,
  "context": [1, 2, 3],
  "total_duration": 5589157167,
  "prompt_eval_count": 46,
  "prompt_eval_duration": 1160282000,
  "eval_count": 113,
  "eval_duration": 1325948000
}
```

#### Generate request (No streaming)

##### Request

A response can be received in one reply when streaming is off.

```shell
curl http://localhost:11434/api/generate -d '{
  "model": "llama2",
  "prompt": "Why is the sky blue?",
  "stream": false
}'
```

##### Response

If `stream` is set to `false`, the response will be a single JSON object:

```json
{
  "model": "llama2",
  "created_at": "2023-08-04T19:22:45.499127Z",
  "response": "The sky is blue because it is the color of the sky.",
  "done": true,
  "context": [1, 2, 3],
  "total_duration": 5589157167,
  "prompt_eval_count": 46,
  "prompt_eval_duration": 1160282000,
  "eval_count": 13,
  "eval_duration": 1325948000
}
```

#### Generate request (Raw Mode)

In some cases, you may wish to bypass the templating system and provide a full prompt. In this case, you can use the `raw` parameter to disable templating. Also note that raw mode will not return a context.

##### Request

```shell
curl http://localhost:11434/api/generate -d '{
  "model": "mistral",
  "prompt": "[INST] why is the sky blue? [/INST]",
  "raw": true,
  "stream": false
}'
```

##### Response

```json
{
  "model": "mistral",
  "created_at": "2023-11-03T15:36:02.583064Z",
  "response": " The sky appears blue because of a phenomenon called Rayleigh scattering.",
  "done": true,
  "total_duration": 14648695333,
  "prompt_eval_count": 14,
  "prompt_eval_duration": 286243000,
  "eval_count": 129,
  "eval_duration": 10931424000
}
```

#### Generate request (JSON mode)

When `format` is set to `json`, the output will always be a well-formed JSON object. You must also include text like 'respond using JSON' or 'output in JSON' in the prompt. If you include a schema in the prompt listing the desired property names and types, most models tend to use it. In cases where results aren't perfect, few-shot prompting has a positive effect.

##### Request 

```shell
curl http://localhost:11434/api/generate -d '{
  "model": "llama2",
  "prompt": "What color is the sky at different times of the day? Respond using JSON",
  "format": "json",
  "stream": false
}'
```

##### Response

```json
{
  "model": "llama2",
  "created_at": "2023-11-09T21:07:55.186497Z",
  "response": "{\n\"morning\": {\n\"color\": \"blue\"\n},\n\"noon\": {\n\"color\": \"blue-gray\"\n},\n\"afternoon\": {\n\"color\": \"warm gray\"\n},\n\"evening\": {\n\"color\": \"orange\"\n}\n}\n",
  "done": true,
  "context": [1, 2, 3], 
  "total_duration": 4661289125,
  "prompt_eval_count": 36,
  "prompt_eval_duration": 264132000,
  "eval_count": 75,
  "eval_duration": 2112149000
}
```

The value of `response` will be a string containing JSON similar to:

```json
{
  "morning": {
    "color": "blue"
  },
  "noon": {
    "color": "blue-gray"
  },
  "afternoon": {
    "color": "warm gray"
  },
  "evening": {
    "color": "orange"
  }
}
```


#### Generate request (With options)

If you want to set custom options for the model at runtime rather than in the Modelfile, you can do so with the `options` parameter. This example sets every available option, but you can set any of them individually and omit the ones you do not want to override.

##### Request

```shell
curl http://localhost:11434/api/generate -d '{
  "model": "llama2",
  "prompt": "Why is the sky blue?",
  "stream": false,
  "options": {
    "num_keep": 5,
    "seed": 42,
    "num_predict": 100,
    "top_k": 20,
    "top_p": 0.9,
    "tfs_z": 0.5,
    "typical_p": 0.7,
    "repeat_last_n": 33,
    "temperature": 0.8,
    "repeat_penalty": 1.2,
    "presence_penalty": 1.5,
    "frequency_penalty": 1.0,
    "mirostat": 1,
    "mirostat_tau": 0.8,
    "mirostat_eta": 0.6,
    "penalize_newline": true,
    "stop": ["\n", "user:"],
    "numa": false,
    "num_ctx": 1024,
    "num_batch": 2,
    "num_gqa": 1,
    "num_gpu": 1,
    "main_gpu": 0,
    "low_vram": false,
    "f16_kv": true,
    "logits_all": false,
    "vocab_only": false,
    "use_mmap": true,
    "use_mlock": false,
    "embedding_only": false,
    "rope_frequency_base": 1.1,
    "rope_frequency_scale": 0.8,
    "num_thread": 8
  }
}'
```

##### Response

```json
{
  "model": "llama2",
  "created_at": "2023-08-04T19:22:45.499127Z",
  "response": "The sky is blue because it is the color of the sky.",
  "done": true,
  "context": [1, 2, 3], 
  "total_duration": 5589157167,
  "prompt_eval_count": 46,
  "prompt_eval_duration": 1160282000,
  "eval_count": 13,
  "eval_duration": 1325948000
}
```

#### Load a model

If you post a body with just the model, it will load the model into memory, but not do a generation.

##### Request

```shell
curl http://localhost:11434/api/generate -d '{
  "model": "llama2"
}
```

##### Response

A single JSON object is returned:

```json
{
  "model": "llama2",
  "created_at": "2023-12-12T02:34:16.373531Z",
  "response": "",
  "model_configuration": {
    "model_format": "gguf",
    "model_family": "llama",
    "model_families": null,
    "model_type": "7B",
    "file_type": "Q4_0"
  },
  "done": true
}
```

## Generate a chat completion

```shell
POST /api/chat
```

Generate the next message in a chat with a provided model. This is a streaming endpoint, so there will be a series of responses. Streaming can be disabled using `stream: false`. The final response object will include statistics and additional data from the request.

### Parameters

- `model`: (required) the [model name](#model-names)
- `messages`: the messages of the chat, this can be used to keep a chat memory

Advanced parameters (optional):

- `format`: the format to return a response in. Currently the only accepted value is `json`
- `options`: additional model parameters listed in the documentation for the [Modelfile](./modelfile.md#valid-parameters-and-values) such as `temperature`
- `template`: the full prompt or prompt template (overrides what is defined in the `Modelfile`)
- `stream`: if `false` the response will be returned as a single response object, rather than a stream of objects

### Examples

#### Chat Request (Streaming)

##### Request

Send a chat message with a streaming response.

```shell
curl http://localhost:11434/api/chat -d '{
  "model": "llama2",
  "messages": [
    {
      "role": "user",
      "content": "why is the sky blue?"
    }
  ]
}'
```

##### Response

A stream of JSON objects is returned:

```json
{
  "model": "llama2",
  "created_at": "2023-08-04T08:52:19.385406455-07:00",
  "message": {
    "role": "assisant",
    "content": "The"
  },
  "done": false
}
```

Final response:

```json
{
  "model": "llama2",
  "created_at": "2023-08-04T19:22:45.499127Z",
  "done": true,
  "total_duration": 5589157167,
  "prompt_eval_count": 46,
  "prompt_eval_duration": 1160282000,
  "eval_count": 113,
  "eval_duration": 1325948000
}
```

#### Chat request (No streaming)

##### Request

```shell
curl http://localhost:11434/api/chat -d '{
  "model": "llama2",
  "messages": [
    {
      "role": "user",
      "content": "why is the sky blue?"
    }
  ], 
  "stream": false
}'
```

##### Response

```json
{
  "model": "registry.ollama.ai/library/llama2:latest",
  "created_at": "2023-12-12T14:13:43.416799Z",
  "message": {
    "role": "assistant",
    "content": "Hello! How are you today?"
  },
  "done": true,
  "total_duration": 390291583,
  "prompt_eval_count": 21,
  "prompt_eval_duration": 285840000,
  "eval_count": 7,
  "eval_duration": 96220000
}
```

#### Chat request (With History)

Send a chat message with a conversation history. You can use this same approach to start the conversation using multi-shot prompting.

##### Request

```shell
curl http://localhost:11434/api/chat -d '{
  "model": "llama2",
  "messages": [
    {
      "role": "user",
      "content": "why is the sky blue?"
    },
    {
      "role": "assistant",
      "content": "due to rayleigh scattering."
    },
    {
      "role": "user",
      "content": "how is that different than mie scattering?"
    }
  ]
}'
```

##### Response

A stream of JSON objects is returned:

```json
{
  "model": "llama2",
  "created_at": "2023-08-04T08:52:19.385406455-07:00",
  "message": {
    "role": "assisant",
    "content": "The"
  },
  "done": false
}
```

Final response:

```json
{
  "model": "llama2",
  "created_at": "2023-08-04T19:22:45.499127Z",
  "done": true,
  "total_duration": 5589157167,
  "prompt_eval_count": 46,
  "prompt_eval_duration": 1160282000,
  "eval_count": 113,
  "eval_duration": 1325948000
}
```

## Create a Model

```shell
POST /api/create
```

Create a model from a [`Modelfile`](./modelfile.md). It is recommended to set `modelfile` to the content of the Modelfile rather than just set `path`. This is a requirement for remote create. Remote model creation must also create any file blobs, fields such as `FROM` and `ADAPTER`, explicitly with the server using [Create a Blob](#create-a-blob) and the value to the path indicated in the response. Note that this just creates the model on the Ollama server and doesn't push it to Ollama.ai.

### Parameters

- `name`: name of the model to create
- `modelfile` (optional): contents of the Modelfile
- `stream`: (optional) if `false` the response will be returned as a single response object, rather than a stream of objects
- `path` (optional): path to the Modelfile

### Examples

#### Create a new model

Create a new model from a modelfile.

##### Request

```shell
curl http://localhost:11434/api/create -d '{
  "name": "mario",
  "modelfile": "FROM llama2\nSYSTEM You are mario from Super Mario Bros."
}'
```

##### Response

A stream of JSON objects. Notice that the final JSON object shows a `"status": "success"`.

```json
{"status":"reading model metadata"}
{"status":"creating system layer"}
{"status":"using already created layer sha256:22f7f8ef5f4c791c1b03d7eb414399294764d7cc82c7e94aa81a1feb80a983a2"}
{"status":"using already created layer sha256:8c17c2ebb0ea011be9981cc3922db8ca8fa61e828c5d3f44cb6ae342bf80460b"}
{"status":"using already created layer sha256:7c23fb36d80141c4ab8cdbb61ee4790102ebd2bf7aeff414453177d4f2110e5d"}
{"status":"using already created layer sha256:2e0493f67d0c8c9c68a8aeacdf6a38a2151cb3c4c1d42accf296e19810527988"}
{"status":"using already created layer sha256:2759286baa875dc22de5394b4a925701b1896a7e3f8e53275c36f75a877a82c9"}
{"status":"writing layer sha256:df30045fe90f0d750db82a058109cecd6d4de9c90a3d75b19c09e5f64580bb42"}
{"status":"writing layer sha256:f18a68eb09bf925bb1b669490407c1b1251c5db98dc4d3d81f3088498ea55690"}
{"status":"writing manifest"}
{"status":"success"}
```

### Check if a Blob Exists

```shell
HEAD /api/blobs/:digest
```

Ensures that the file blob used for a FROM or ADAPTER field exists on the server. This is just checking your Ollama server and not Ollama.ai.


#### Query Parameters

- `digest`: the SHA256 digest of the blob

#### Examples

##### Request

```shell
curl -I http://localhost:11434/api/blobs/sha256:29fdb92e57cf0827ded04ae6461b5931d01fa595843f55d36f5b275a52087dd2
```

##### Response

Return 200 OK if the blob exists, 404 Not Found if it does not.

### Create a Blob

```shell
POST /api/blobs/:digest
```

Create a blob from a file on the server. Returns the server file path.

#### Query Parameters

- `digest`: the expected SHA256 digest of the file

#### Examples

##### Request

```shell
curl -T model.bin -X POST http://localhost:11434/api/blobs/sha256:29fdb92e57cf0827ded04ae6461b5931d01fa595843f55d36f5b275a52087dd2
```

##### Response

Return 201 Created if the blob was successfully created, 400 Bad Request if the digest used is not expected.

## List Local Models

```shell
GET /api/tags
```

List models that are available locally.

### Examples

#### Request

```shell
curl http://localhost:11434/api/tags
```

#### Response

A single JSON object will be returned.

```json
{
  "models": [
    {
      "name": "codellama:13b",
      "modified_at": "2023-11-04T14:56:49.277302595-07:00",
      "size": 7365960935,
      "digest": "9f438cb9cd581fc025612d27f7c1a6669ff83a8bb0ed86c94fcf4c5440555697",
      "details": {
        "format": "gguf",
        "family": "llama",
        "families": null,
        "parameter_size": "13B",
        "quantization_level": "Q4_0"
      }
    },
    {
      "name": "llama2:latest",
      "modified_at": "2023-12-07T09:32:18.757212583-08:00",
      "size": 3825819519,
      "digest": "fe938a131f40e6f6d40083c9f0f430a515233eb2edaa6d72eb85c50d64f2300e",
      "details": {
        "format": "gguf",
        "family": "llama",
        "families": null,
        "parameter_size": "7B",
        "quantization_level": "Q4_0"
      }
    }
  ]
}
```

## Show Model Information

```shell
POST /api/show
```

Show details about a model including modelfile, template, parameters, license, and system prompt.

### Parameters

- `name`: name of the model to show

### Examples

#### Request

```shell
curl http://localhost:11434/api/show -d '{
  "name": "llama2"
}'
```

#### Response

```json
{
  "license": "<contents of license block>",
  "modelfile": "# Modelfile generated by \"ollama show\"\n# To build a new Modelfile based on this one, replace the FROM line with:\n# FROM llama2:latest\n\nFROM /Users/username/.ollama/models/blobs/sha256:8daa9615cce30c259a9555b1cc250d461d1bc69980a274b44d7eda0be78076d8\nTEMPLATE \"\"\"[INST] {{ if and .First .System }}<<SYS>>{{ .System }}<</SYS>>\n\n{{ end }}{{ .Prompt }} [/INST] \"\"\"\nSYSTEM \"\"\"\"\"\"\nPARAMETER stop [INST]\nPARAMETER stop [/INST]\nPARAMETER stop <<SYS>>\nPARAMETER stop <</SYS>>\n",
  "parameters": "stop                           [INST]\nstop                           [/INST]\nstop                           <<SYS>>\nstop                           <</SYS>>",
  "template": "[INST] {{ if and .First .System }}<<SYS>>{{ .System }}<</SYS>>\n\n{{ end }}{{ .Prompt }} [/INST] "
  "details": {
    "format": "gguf",
    "family": "llama",
    "families": null,
    "parameter_size": "7B",
    "quantization_level": "Q4_0"
  }
}
```

## Copy a Model

```shell
POST /api/copy
```

Copy a model. Creates a model with another name from an existing model.

### Examples

#### Request

```shell
curl http://localhost:11434/api/copy -d '{
  "source": "llama2",
  "destination": "llama2-backup"
}'
```

#### Response

Returns a 200 OK if successful, or a 404 Not Found if the source model doesn't exist.

## Delete a Model

```shell
DELETE /api/delete
```

Delete a model and its data.

### Parameters

- `name`: model name to delete

### Examples

#### Request

```shell
curl -X DELETE http://localhost:11434/api/delete -d '{
  "name": "llama2:13b"
}'
```

#### Response

Returns a 200 OK if successful, 404 Not Found if the model to be deleted doesn't exist.

## Pull a Model

```shell
POST /api/pull
```

Download a model from the ollama library. Cancelled pulls are resumed from where they left off, and multiple calls will share the same download progress.

### Parameters

- `name`: name of the model to pull
- `insecure`: (optional) allow insecure connections to the library. Only use this if you are pulling from your own library during development.
- `stream`: (optional) if `false` the response will be returned as a single response object, rather than a stream of objects

### Examples

#### Request

```shell
curl http://localhost:11434/api/pull -d '{
  "name": "llama2"
}'
```

#### Response

If `stream` is not specified, or set to `true`, a stream of JSON objects is returned:

The first object is the manifest:

```json
{
  "status": "pulling manifest"
}
```

Then there is a series of downloading responses. Until any of the download is completed, the `completed` key may not be included. The number of files to be downloaded depends on the number of layers specified in the manifest.

```json
{
  "status": "downloading digestname",
  "digest": "digestname",
  "total": 2142590208,
  "completed": 241970
}
```

After all the files are downloaded, the final responses are:

```json
{
    "status": "verifying sha256 digest"
}
{
    "status": "writing manifest"
}
{
    "status": "removing any unused layers"
}
{
    "status": "success"
}
```

if `stream` is set to false, then the response is a single JSON object:

```json
{
  "status": "success"
}
```

## Push a Model

```shell
POST /api/push
```

Upload a model to a model library. Requires registering for ollama.ai and adding a public key first.

### Parameters

- `name`: name of the model to push in the form of `<namespace>/<model>:<tag>`
- `insecure`: (optional) allow insecure connections to the library. Only use this if you are pushing to your library during development.
- `stream`: (optional) if `false` the response will be returned as a single response object, rather than a stream of objects

### Examples

#### Request

```shell
curl http://localhost:11434/api/push -d '{
  "name": "mattw/pygmalion:latest"
}'
```

#### Response

If `stream` is not specified, or set to `true`, a stream of JSON objects is returned:

```json
{ "status": "retrieving manifest" }
```

and then:

```json
{
  "status": "starting upload",
  "digest": "sha256:bc07c81de745696fdf5afca05e065818a8149fb0c77266fb584d9b2cba3711ab",
  "total": 1928429856
}
```

Then there is a series of uploading responses:

```json
{
  "status": "starting upload",
  "digest": "sha256:bc07c81de745696fdf5afca05e065818a8149fb0c77266fb584d9b2cba3711ab",
  "total": 1928429856
}
```

Finally, when the upload is complete:

```json
{"status":"pushing manifest"}
{"status":"success"}
```

If `stream` is set to `false`, then the response is a single JSON object:

```json
{ "status": "success" }
```

## Generate Embeddings

```shell
POST /api/embeddings
```

Generate embeddings from a model

### Parameters

- `model`: name of model to generate embeddings from
- `prompt`: text to generate embeddings for

Advanced parameters:

- `options`: additional model parameters listed in the documentation for the [Modelfile](./modelfile.md#valid-parameters-and-values) such as `temperature`

### Examples

#### Request

```shell
curl http://localhost:11434/api/embeddings -d '{
  "model": "llama2",
  "prompt": "Here is an article about llamas..."
}'
```

#### Response

```json
{
  "embedding": [
    0.5670403838157654, 0.009260174818336964, 0.23178744316101074, -0.2916173040866852, -0.8924556970596313,
    0.8785552978515625, -0.34576427936553955, 0.5742510557174683, -0.04222835972905159, -0.137906014919281
  ]
}
```
