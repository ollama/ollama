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
  Context: "context",
}
```

Context is optional, but is used to provide additional context, such as memory of earlier prompts.

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
  "context":[1,31822,1,13,8458,31922 ... 382,871,550,389,266,7661,31844,382,820,541,4842,1954,661,645,590,3465,31843,2],
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
| context    | vectorize context that can be supplied in the next request to continue the conversation |
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
{"model":"orca","created_at":"2023-08-04T19:22:45.499127Z","done":true,"context":[1,31822,1,13,8458,31922,3244,31871,13,3838,397,363,7421,8825,342,5243,10389,5164,828,31843,9530,362,988,362,365,473,31843,13,13,8458,31922,9779,31871,13,23712,322,266,7661,4842,13,13,8458,31922,13166,31871,13,347,7661,4725,4842,906,287,260,12329,1676,6697,27554,27289,31843,4025,2990,322,985,550,287,260,9949,287,8286,31844,10990,427,2729,289,399,20036,31843,1408,21062,16858,266,4556,31876,31829,7965,31844,357,19322,16450,287,1900,859,362,22329,291,11944,31843,1872,16450,397,988,5497,661,266,23893,287,266,1954,31844,560,526,640,3304,266,1954,288,484,11468,31843,1813,31844,4842,1954,470,260,13830,23893,661,590,8286,31844,560,357,322,18752,541,4083,31843,672,1901,342,662,382,871,550,389,266,7661,31844,382,820,541,4842,1954,661,645,590,3465,31843,2],"total_duration":5589157167,"load_duration":3013701500,"sample_count":114,"sample_duration":81442000,"prompt_eval_count":46,"prompt_eval_duration":1160282000,"eval_count":113,"eval_duration":1325948000}
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
