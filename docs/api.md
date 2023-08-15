# API

## Endpoints

- [Generate a completion](#generate-a-completion)
- [Create a model](#create-a-model)
- [List local models](#list-local-models)
- [Copy a model](#copy-a-model)
- [Delete a model](#delete-a-model)
- [Pull a model](#pull-a-model)
- [Generate embeddings](#generate-embeddings)

## Conventions

### Model names

Model names follow a `model:tag` format. Some examples are `orca:3b-q4_1` and `llama2:70b`. The tag is optional and if not provided will default to `latest`. The tag is used to identify a specific version.

### Durations

All durations are returned in nanoseconds.

## Generate a completion

```
POST /api/generate
```

Generate a response for a given prompt with a provided model. This is a streaming endpoint, so will be a series of responses. The final response object will include statistics and additional data from the request.

### Parameters

- `model`: (required) the [model name](#model-names)
- `prompt`: the prompt to generate a response for

Advanced parameters:

- `options`: additional model parameters listed in the documentation for the [Modelfile](./modelfile.md#valid-parameters-and-values) such as `temperature`
- `system`: system prompt to (overrides what is defined in the `Modelfile`)
- `template`: the full prompt or prompt template (overrides what is defined in the `Modelfile`)
- `context`: the context parameter returned from a previous request to `/generate`, this can be used to keep a short conversational memory

### Request

```
curl -X POST http://localhost:11434/api/generate -d '{
  "model": "llama2:7b",
  "prompt": "Why is the sky blue?"
}'
```

### Response

A stream of JSON objects:

```json
{
  "model": "llama2:7b",
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

To calculate how fast the response is generated in tokens per second (token/s), divide `eval_count` / `eval_duration`.

```json
{
  "model": "llama2:7b",
  "created_at": "2023-08-04T19:22:45.499127Z",
  "context": [1, 2, 3],
  "done": true,
  "total_duration": 5589157167,
  "load_duration": 3013701500,
  "sample_count": 114,
  "sample_duration": 81442000,
  "prompt_eval_count": 46,
  "prompt_eval_duration": 1160282000,
  "eval_count": 113,
  "eval_duration": 1325948000
}
```

## Create a Model

```
POST /api/create
```

Create a model from a [`Modelfile`](./modelfile.md)

### Parameters

- `name`: name of the model to create
- `path`: path to the Modelfile

### Request

```
curl -X POST http://localhost:11434/api/create -d '{
  "name": "mario",
  "path": "~/Modelfile"
}'
```

### Response

A stream of JSON objects. When finished, `status` is `success`

```json
{
  "status": "parsing modelfile"
}
```

## List Local Models

```
GET /api/tags
```

List models that are available locally.

### Request

```
curl http://localhost:11434/api/tags
```

### Response

```json
{
  "models": [
    {
      "name": "llama2:7b",
      "modified_at": "2023-08-02T17:02:23.713454393-07:00",
      "size": 3791730596
    },
    {
      "name": "llama2:13b",
      "modified_at": "2023-08-08T12:08:38.093596297-07:00",
      "size": 7323310500
    }
  ]
}
```

## Copy a Model

```
POST /api/copy
```

Copy a model. Creates a model with another name from an existing model.

### Request

```
curl http://localhost:11434/api/copy -d '{
  "source": "llama2:7b",
  "destination": "llama2-backup"
}'
```

## Delete a Model

```
DELETE /api/delete
```

Delete a model and its data.

### Parameters

- `model`: model name to delete

### Request

```
curl -X DELETE http://localhost:11434/api/delete -d '{
  "name": "llama2:13b"
}'
```

## Pull a Model

```
POST /api/pull
```

Download a model from a the model registry. Cancelled pulls are resumed from where they left off, and multiple calls to will share the same download progress.

### Parameters

- `name`: name of the model to pull

### Request

```
curl -X POST http://localhost:11434/api/pull -d '{
  "name": "llama2:7b"
}'
```

### Response

```json
{
  "status": "downloading digestname",
  "digest": "digestname",
  "total": 2142590208
}
```

## Generate Embeddings

```
POST /api/embeddings
```

Generate embeddings from a model

### Parameters

- `model`: name of model to generate embeddings from
- `prompt`: text to generate embeddings for

### Request

```
curl -X POST http://localhost:11434/api/embeddings -d '{
  "model": "llama2:7b",
  "prompt": "Here is an article about llamas..."
}'
```

### Response

```json
{
  "embeddings": [
    0.5670403838157654, 0.009260174818336964, 0.23178744316101074, -0.2916173040866852, -0.8924556970596313,
    0.8785552978515625, -0.34576427936553955, 0.5742510557174683, -0.04222835972905159, -0.137906014919281
  ]
}
```
