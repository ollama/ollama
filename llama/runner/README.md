# `runner`

> Note: this is a work in progress

A minimial runner for loading a model and running inference via a http web server.

```
./runner -model <model binary>
```

### Completion

```
curl -X POST -H "Content-Type: application/json" -d '{"prompt": "hi"}' http://localhost:8080/completion
```

### Embeddings

```
curl -X POST -H "Content-Type: application/json" -d '{"prompt": "turn me into an embedding"}' http://localhost:8080/embeddings
```
