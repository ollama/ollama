[Documentation Home](./README.md)

# API

## Generate a Prompt

**POST /api/generate**

**Generate** is the main endpoint that you will use when working with Ollama. This is used to generate a response to a prompt sent to a model.

The **Generate** endpoint takes a JSON object with the following fields:

```
{
  Model: "modelname",
  Prompt: "prompt",
}
```

The response is a stream of JSON objects with the following fields:

```
{
  "model": "modelname",
  "created_at": "2023-08-04T08:52:19.385406455-07:00"
  "response": "the current token",
  "done": false
}
```

The final response in the stream also includes the context and what is usually seen in the output from verbose mode. For example:

```
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

| field      | description                         |
| ---------- | ----------------------------------- |
| model      | the name of the model               |
| created_at | the time the response was generated |
| response   | the current token                   |
| done       | whether the response is complete    |
| total_duration | total time spent generating the response |
| load_duration | time spent loading the model |
| sample_count | number of samples generated |
| sample_duration | time spent generating samples |
| prompt_eval_count | number of times the prompt was evaluated |
| prompt_eval_duration | time spent evaluating the prompt |
| eval_count | number of times the response was evaluated |
| eval_duration | time spent evaluating the response |



### Example Request

```curl
curl --location --request POST 'http://localhost:11434/api/generate' \
--header 'Content-Type: text/plain' \
--data-raw '{
    "model": "orca",
    "prompt": "why is the sky blue"
}'
```

### Example Response

```json
{"model":"orca","created_at":"2023-08-04T19:22:44.085127Z","response":" The","done":false}
{"model":"orca","created_at":"2023-08-04T19:22:44.176425Z","response":" sky","done":false}
{"model":"orca","created_at":"2023-08-04T19:22:44.18883Z","response":" appears","done":false}
{"model":"orca","created_at":"2023-08-04T19:22:44.200852Z","response":" blue","done":false}
{"model":"orca","created_at":"2023-08-04T19:22:44.213644Z","response":" because","done":false}
{"model":"orca","created_at":"2023-08-04T19:22:44.225706Z","response":" of","done":false}
{"model":"orca","created_at":"2023-08-04T19:22:44.237686Z","response":" a","done":false}
.
.
.
{"model":"orca","created_at":"2023-08-04T19:22:45.487113Z","response":".","done":false}
{"model":"orca","created_at":"2023-08-04T19:22:45.499127Z","done":true,"total_duration":5589157167,"load_duration":3013701500,"sample_count":114,"sample_duration":81442000,"prompt_eval_count":46,"prompt_eval_duration":1160282000,"eval_count":113,"eval_duration":1325948000}
```

## Create a Model

**POST /api/create**

## List Local Models

**GET /api/tags**

### Return Object

```
{
  "models": [
    {
      "name": "modelname:tags",
      "modified_at": "2023-08-04T08:52:19.385406455-07:00",
      "size": size
    }
  ]

}
```

## Copy a Model

**/api/copy**

## Delete a Model

**/api/delete**

## Pull a Model

**/api/pull**

## Push a Model

**/api/push**

## Heartbeat

**/**
