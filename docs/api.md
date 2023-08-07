[Documentation Home](./README.md)

# API

- [Generate a Prompt](#generate-a-prompt)
- [Create a Model](#create-a-model)
- [List Local Models](#list-local-models)
- [Copy a Model](#copy-a-model)
- [Delete a Model](#delete-a-model)
- [Pull a Model](#pull-a-model)

## Things to keep in mind when using the API

### Model name format

The model name format today is `model:tag`. Some examples are `orca:3b-q4_1` and `llama2:70b`. The tag is optional and if not provided will default to `latest`. The tag is used to identify a specific version.

### Durations

All durations are in nanoseconds.

## Generate a Prompt

**POST /api/generate**

### Description

**Generate** is the main endpoint that you will use when working with Ollama. This is used to generate a response to a prompt sent to a model. This is a streaming endpoint, so will be a series of responses. The final response will include the context and what is usually seen in the output from verbose mode.

### Request

The **Generate** endpoint takes a JSON object with the following fields:

```JSON
{
  "model": "site/namespace/model:tag",
  "prompt": "You are a software engineer working on building docs for Ollama.",
  "options": {
    "temperature": 0.7,
  }
}
```

**Options** can include any of the parameters listed in the [Modelfile](./modelfile.mdvalid-parameters-and-values) documentation. The only required parameter is **model**. If no **prompt** is provided, the model will generate a response to an empty prompt. If no **options** are provided, the model will use the default options from the Modelfile of the parent model.

### Response

The response is a stream of JSON objects with the following fields:

```JSON
{
  "model": "modelname",
  "created_at": "2023-08-04T08:52:19.385406455-07:00"
  "response": "the current token",
  "done": false
}
```

The final response in the stream also includes the context and what is usually seen in the output from verbose mode. For example:

```JSON
{
  "model":"orca",
  "created_at":"2023-08-04T19:22:45.499127Z",
  "done":true,
  "total_duration":5589157167,
  "load_duration":3013701500,
  "sample_count":114,
  "sample_duration":81442000,
  "prompt_eval_count":46,
  "prompt_eval_duration":1160282000,
  "eval_count":113,
  "eval_duration":1325948000
}
```

| field                | description                                             |
| -------------------- | ------------------------------------------------------- |
| model                | the name of the model                                   |
| created_at           | the time the response was generated                     |
| response             | the current token                                       |
| done                 | whether the response is complete                        |
| total_duration       | total time in nanoseconds spent generating the response |
| load_duration        | time spent in nanoseconds loading the model             |
| sample_count         | number of samples generated                             |
| sample_duration      | time spent generating samples                           |
| prompt_eval_count    | number of times the prompt was evaluated                |
| prompt_eval_duration | time spent in nanoseconds evaluating the prompt         |
| eval_count           | number of times the response was evaluated              |
| eval_duration        | time in nanoseconds spent evaluating the response       |

### Example

#### Request

```shell
curl -X POST 'http://localhost:11434/api/generate' -d \
'{
    "model": "orca",
    "prompt": "why is the sky blue"
}'
```

#### Response

```json
{"model":"orca","created_at":"2023-08-04T19:22:44.085127Z","response":" The","done":false}
{"model":"orca","created_at":"2023-08-04T19:22:44.176425Z","response":" sky","done":false}
{"model":"orca","created_at":"2023-08-04T19:22:44.18883Z","response":" appears","done":false}
{"model":"orca","created_at":"2023-08-04T19:22:44.200852Z","response":" blue","done":false}
{"model":"orca","created_at":"2023-08-04T19:22:44.213644Z","response":" because","done":false}
{"model":"orca","created_at":"2023-08-04T19:22:44.225706Z","response":" of","done":false}
{"model":"orca","created_at":"2023-08-04T19:22:44.237686Z","response":" a","done":false}
...
{"model":"orca","created_at":"2023-08-04T19:22:45.487113Z","response":".","done":false}
{"model":"orca","created_at":"2023-08-04T19:22:45.499127Z","done":true,"total_duration":5589157167,"load_duration":3013701500,"sample_count":114,"sample_duration":81442000,"prompt_eval_count":46,"prompt_eval_duration":1160282000,"eval_count":113,"eval_duration":1325948000}
```

## Create a Model

**POST /api/create**

### Description

**Create** takes a path to a Modelfile and creates a model. The Modelfile is documented [here](./modelfile.md).

### Request

The **Create** endpoint takes a JSON object with the following fields:

```JSON
{
  "name": "modelname",
  "path": "absolute path to Modelfile"
}
```

### Response

The response is a stream of JSON objects that have a single key/value pair for status. For example:

```JSON
{
  "status": "parsing modelfile"
}
```

### Example

#### Request

```shell
curl --location --request POST 'http://localhost:11434/api/create' \
--header 'Content-Type: text/plain' \
--data-raw '{
    "name": "myCoolModel",
    "path": "/Users/matt/ollamamodelfiles/sentiments"
}'
```

#### Response

```JSON
{"status":"parsing modelfile"}
{"status":"looking for model"}
{"status":"creating model template layer"}
{"status":"creating config layer"}
{"status":"using already created layer sha256:e84705205f71dd55be7b24a778f248f0eda9999a125d313358c087e092d83148"}
{"status":"using already created layer sha256:93ca9b3d83dc541f11062c0b994ae66a7b327146f59a9564aafef4a4c15d1ef5"}
{"status":"writing layer sha256:d3fe6fb39620a477da7720c5fa00abe269a018a9675a726320e18122b7142ee7"}
{"status":"writing layer sha256:16cc83359b0395026878b41662f7caef433f5260b5d49a3257312b6417b7d8a8"}
{"status":"writing manifest"}
{"status":"success"}
```

## List Local Models

**GET /api/tags**

### Description

**List** will list out the models that are available locally.

### Request

The **List** endpoint takes no parameters and is a simple GET request.

### Response

The response is a JSON object with a single key/value pair for models. For example:

```JSON
{
  "models": [
    {
      "name": "modelname:tags",
      "modified_at": "2023-08-04T08:52:19.385406455-07:00",
      "size": 7323310500
    }
  ]
}
```

### Example

#### Request

```shell
curl 'http://localhost:11434/api/tags'
```

#### Response

```JSON
{
    "models": [
        {
            "name": "llama2:70b",
            "modified_at": "2023-08-04T08:52:19.385406455-07:00",
            "size": 38871966966
        },
        {
            "name": "llama2:70b-chat-q4_0",
            "modified_at": "2023-08-04T09:21:27.703371485-07:00",
            "size": 38871974480
        },
        {
            "name": "midjourney-prompter:latest",
            "modified_at": "2023-08-04T08:45:46.399609053-07:00",
            "size": 7323311708
        },
        {
            "name": "raycast_orca:3b",
            "modified_at": "2023-08-04T06:23:20.10832636-07:00",
            "size": 1928446602
        },
        {
            "name": "stablebeluga:13b-q4_K_M",
            "modified_at": "2023-08-04T09:48:26.416547463-07:00",
            "size": 7865679045
        }
    ]
}
```

## Copy a Model

**POST /api/copy**

### Description

**Copy** will copy a model from one name to another. This is useful for creating a new model from an existing model. It is often used as the first step to renaming a model.

### Request

The **Copy** endpoint takes a JSON object with the following fields:

```JSON
{
  "source": "modelname",
  "destination": "newmodelname"
}
```

### Response

There is no response other than a 200 status code.

### Example

#### Request

```shell
curl -X POST 'http://localhost:11434/api/copy' -d \
'{
    "source": "MyCoolModel",
    "destination": "ADifferentModel"
}'
```

#### Response

No response is returned other than a 200 status code.

## Delete a Model

**DELETE /api/delete**

### Description

**Delete** will delete a model from the local machine. This is useful for cleaning up models that are no longer needed.

### Request

The **Delete** endpoint takes a JSON object with a single key/value pair for modelname. For example:

```JSON
{
  "model": "modelname"
}
```

### Response

No response is returned other than a 200 status code.

### Example

#### Request

```shell
curl -X DELETE 'http://localhost:11434/api/delete' -d \
'{
    "name": "adifferentModel"
}'
```

#### Response

No response is returned other than a 200 status code.

## Pull a Model

**POST /api/pull**

### Description

**Pull** will pull a model from a remote registry. This is useful for getting a model from the Ollama registry and in the future from alternate registries.

### Request

The **Pull** endpoint takes a JSON object with the following fields:

```JSON
{
  "name": "modelname"
}
```

### Response

The response is a stream of JSON objects with the following format:

```JSON
{
  "status":"downloading digestname",
  "digest":"digestname",
  "total":2142590208
}
```

### Example

#### Request

```shell
curl -X POST 'http://localhost:11434/api/pull' -d \
'{
    "name": "orca:3b-q4_1"
}'
```

#### Response

```JSON
{"status":"pulling manifest"}
{"status":"downloading sha256:63151c63f792939bb4a40b35f37ea06e047c02486399d1742113aaefd0d33e29","digest":"sha256:63151c63f792939bb4a40b35f37ea06e047c02486399d1742113aaefd0d33e29","total":2142590208}
{"status":"downloading sha256:63151c63f792939bb4a40b35f37ea06e047c02486399d1742113aaefd0d33e29","digest":"sha256:63151c63f792939bb4a40b35f37ea06e047c02486399d1742113aaefd0d33e29","total":2142590208,"completed":1048576}
...
{"status":"downloading sha256:20714f2ebe4be44313358bfa58556d783652398ed47f12178914c706c4ad12c4","digest":"sha256:20714f2ebe4be44313358bfa58556d783652398ed47f12178914c706c4ad12c4","total":299}
{"status":"downloading sha256:20714f2ebe4be44313358bfa58556d783652398ed47f12178914c706c4ad12c4","digest":"sha256:20714f2ebe4be44313358bfa58556d783652398ed47f12178914c706c4ad12c4","total":299,"completed":299}
{"status":"verifying sha256 digest"}
{"status":"writing manifest"}
{"status":"success"}

```
