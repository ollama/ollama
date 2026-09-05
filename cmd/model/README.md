# Model diff

`model-diff` is a tool for finding differences between model artifacts. It performs offline comparisons of local Ollama models, safetensors files or
directories, and GGUF files.

```sh
go run ./cmd/model baseline candidate
go run ./cmd/model --stats baseline candidate
go run ./cmd/model --tensor 'language_model.*gate_proj' baseline candidate
go run ./cmd/model ./baseline.gguf ./candidate.gguf
```

The report covers metadata, tensor descriptors, and tensor data.

`--stats` adds supported MXFP/NVFP saturation and dequantized NMSE diagnostics.
`--summary`, `--all`, `--tensor`, and `--metadata-only` adjust the comparison or
report. Run `-h` for the current flags.

The binary exits `0` for equal in the requested scope, `1` for different, and
`2` for an error or incomplete inspection. `go run` collapses nonzero program
exit codes to `1`.
