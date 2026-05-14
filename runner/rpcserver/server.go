// Package rpcserver implements the ollama runner --rpc-server subcommand.
// Run this on any machine (e.g. Raspberry Pi) to expose its RAM/GPU as a
// remote compute device that the main Ollama host can offload model layers to.
package rpcserver

import (
	"flag"
	"fmt"
	"log/slog"
	"os"

	rpc "github.com/ollama/ollama/ml/backend/ggml/ggml/src/ggml-rpc"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/llama"
	"github.com/ollama/ollama/logutil"
)

func Execute(args []string) error {
	fs := flag.NewFlagSet("rpc-server", flag.ExitOnError)
	host     := fs.String("host", "0.0.0.0", "Address to listen on")
	port     := fs.Int("port", 50052, "Port to listen on")
	threads  := fs.Int("threads", 0, "Number of CPU threads (0 = auto)")
	cacheDir := fs.String("cache-dir", "", "Directory for tensor caching (optional)")

	fs.Usage = func() {
		fmt.Fprintf(fs.Output(), "Usage: ollama runner --rpc-server [flags]\n\n")
		fmt.Fprintf(fs.Output(), "Starts an RPC server that exposes local compute resources to a remote Ollama host.\n")
		fmt.Fprintf(fs.Output(), "Run this on the Raspberry Pi; point the Windows PC at it via OLLAMA_RPC_SERVERS.\n\n")
		fs.PrintDefaults()
	}

	if err := fs.Parse(args); err != nil {
		return err
	}

	slog.SetDefault(logutil.NewLogger(os.Stderr, envconfig.LogLevel()))

	llama.BackendInit()

	endpoint := fmt.Sprintf("%s:%d", *host, *port)
	slog.Info("starting RPC server", "endpoint", endpoint, "threads", *threads)

	// Blocks until the server is shut down.
	rpc.StartServer(endpoint, *cacheDir, *threads)
	return nil
}
