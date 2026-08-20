# AGENTS.md

## Building

For a full build from the repository root:

```sh
cmake -B build .
cmake --build build --parallel 8
./ollama serve
```

For quick Go-only iteration against an existing native payload:

```sh
go build .
go run . serve
```

See `docs/development.md` for prerequisites, platform notes, GPU backends, and
the full development workflow.

## Testing

For full-repo Go validation, `go test ./...` expects the frontend bundle at
`app/dist` because `app/ui/app.go` embeds it. If that bundle has not been built
yet, either generate it first or exclude `./app/cmd/app` and `./app/ui` from
Go-only test runs.
