# Makefile — convenience targets for local development builds
#
# This is intentionally minimal. The real work lives in scripts/build-local.sh
# (which handles cmake + Go + native llama-server payload for both the
# API layer and the runner).
#
# Primary targets for quick iteration (especially gRPC/feature branches):
#
#   make local     — full build of Go binary + llama-server (Metal on darwin-arm64)
#   make dev       — alias for local
#   make go        — fast Go-only rebuild (ollama-go target). Use this for most
#                    gRPC handler / client / converter / scheduler changes once
#                    you have built the native payload at least once.
#   make clean     — remove the build/ tree
#   make serve     — run the just-built binary (no env vars)
#
# gRPC-specific quick test example (after make local or make go):
#   OLLAMA_GRPC_HOST=127.0.0.1:11435 ./ollama serve
#   # in another shell:
#   OLLAMA_GRPC_HOST=127.0.0.1:11435 go test -tags=integration -run TestGRPCStreaming ./integration -count=1
#
# See scripts/build-local.sh --help and docs/development.md for details.
# All build artifacts (build/, /ollama, dist/, integration/ollama, etc.) are
# already in .gitignore.

.PHONY: local dev go clean serve help

local:
	./scripts/build-local.sh

dev: local

go:
	./scripts/build-local.sh --go-only

clean:
	./scripts/build-local.sh clean

serve:
	./ollama serve

help:
	./scripts/build-local.sh --help
	@echo
	@echo "Makefile targets:"
	@echo "  make local   (or make dev)  — full Go + native payload"
	@echo "  make go                   — Go only (fast iteration)"
	@echo "  make clean"
	@echo "  make serve"
