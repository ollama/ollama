# `llava`

Demo app for running Llava and other clip-based vision models.

```
ollama pull llava
```

```
go run -x . \
    -model ~/.ollama/models/blobs/sha256-170370233dd5c5415250a2ecd5c71586352850729062ccef1496385647293868 \
    -projector ~/.ollama/models/blobs/sha256-72d6f08a42f656d36b356dbe0920675899a99ce21192fd66266fb7d82ed07539 \
    -image ./alonso.png
```
