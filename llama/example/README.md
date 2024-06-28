# `example`

Demo app for the `llama` package

Pull a model:

```
ollama pull mistral:7b-instruct-v0.3-q4_0
```

Then run it:

```
go run -x . \
    -model ~/.ollama/models/blobs/sha256-ff82381e2bea77d91c1b824c7afb83f6fb73e9f7de9dda631bcdbca564aa5435 \
    -prompt "[ISNT] Why is the sky blue? [/INST]"
```

## Vision

```
ollama pull llava:7b-v1.6-mistral-q4_0
```

```
go run -x . \
    -model ~/.ollama/models/blobs/sha256-170370233dd5c5415250a2ecd5c71586352850729062ccef1496385647293868 \
    -projector ~/.ollama/models/blobs/sha256-72d6f08a42f656d36b356dbe0920675899a99ce21192fd66266fb7d82ed07539 \
    -image ./alonso.jpg \
    -prompt "[ISNT] What is in this image? <image> [/INST]"
```
