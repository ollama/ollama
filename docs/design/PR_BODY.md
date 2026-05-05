## Summary

Adds Gemma 4 **visual token budget** options `image_min_tokens` and `image_max_tokens` on `api.Options`, with ladder snap `{70,140,280,560,1120}`, defaults **70** / **560**, per-completion wiring in **ollamarunner**, and **non-MLX** scheduler reload when those options change (alongside existing `Runner` comparison).

## Design

See [docs/design/gemma4-vision-token-budgets.md](gemma4-vision-token-budgets.md) (full plan preserved in-repo).

## Testing

- `go test ./internal/gemma4vision/... ./model/models/gemma4/...`
- `go test ./runner/ollamarunner/... ./server/...` (or full `go test ./...` in CI)

## Notes

- MLX engine: options decode; `slog.Debug` only when non-zero (vision not on MLX).
- `OLLAMA_DEBUG=1`: Gemma4 vision path emits structured `slog.Debug` for budgets and token count.
