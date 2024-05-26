# `runner`

A subprocess runner for loading a model and running inference via a small http web server.

```
./runner -model <model binary>
```

```
curl -X POST -H "Content-Type: application/json" -d '{"prompt": "hi"}' http://localhost:8080/
```
