# API Reference

This page provides an overview of the Ollama API. For detailed information, examples, and parameters, please refer to the [full API documentation](https://github.com/ollama/ollama/blob/main/docs/api.md).

## Conventions

### Model Names

Model names follow a `model:tag` format, where `model` can have an optional namespace such as `example/model`. Some examples are `orca-mini:3b-q4_1` and `llama3:70b`. The tag is optional and, if not provided, will default to `latest`. The tag is used to identify a specific version.

### Durations

All durations are returned in nanoseconds.

### Streaming Responses

Certain endpoints stream responses as JSON objects. Streaming can be disabled by providing `{"stream": false}` for these endpoints.

## Endpoints

### Generate a Completion

```
POST /api/generate
```

Generate a response for a given prompt with a provided model. This is a streaming endpoint, so there will be a series of responses. The final response object will include statistics and additional data from the request.

### Generate a Chat Completion

```
POST /api/chat
```

Generate the next message in a chat with a provided model. This is a streaming endpoint, so there will be a series of responses. The final response object will include statistics and additional data from the request.

### Create a Model

```
POST /api/create
```

Create a model from a Modelfile.

### List Local Models

```
GET /api/tags
```

List all models that are available locally.

### Show Model Information

```
POST /api/show
```

Show information about a model.

### Copy a Model

```
POST /api/copy
```

Create a copy of a model.

### Delete a Model

```
DELETE /api/delete
```

Delete a model.

### Pull a Model

```
POST /api/pull
```

Download a model from the Ollama library. This is a streaming endpoint that returns the status of the pull.

### Push a Model

```
POST /api/push
```

Upload a model to a model library. This is a streaming endpoint that returns the status of the push.

### Generate Embeddings

```
POST /api/embeddings
```

Generate embeddings for a given prompt with a provided model.

### List Running Models

```
GET /api/running
```

List all models that are currently running.

### Version

```
GET /api/version
```

Get the version of Ollama.

## Examples

For detailed examples of how to use each endpoint, please refer to the [full API documentation](https://github.com/ollama/ollama/blob/main/docs/api.md).

## Using the API with Libraries

Ollama provides official libraries for Python and JavaScript:

- [ollama-python](https://github.com/ollama/ollama-python)
- [ollama-js](https://github.com/ollama/ollama-js)

These libraries provide a convenient way to interact with the Ollama API from your applications.