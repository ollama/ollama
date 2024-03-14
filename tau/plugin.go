package main

import (
	"context"
	"log/slog"

	"github.com/taubyte/vm-orbit/satellite"
)

func main() {
	slog.SetLogLoggerLevel(slog.LevelDebug)
	server := new(context.TODO(), "/tmp/ollama-wasm")
	server.init()
	satellite.Export("ollama", server)
}
