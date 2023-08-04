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

| field | description |
| --- | --- |
| model | the name of the model |
| created_at | the time the response was generated |
| response | the current token |
| done | whether the response is complete |
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

