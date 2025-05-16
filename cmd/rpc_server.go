package cmd

/*
#cgo CPPFLAGS: -I${SRCDIR}/../ml/backend/ggml/ggml/include
#include "rpc-server.h"
*/
import "C"

import (
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"

	"github.com/spf13/cobra"
)

var (
	rpcHost string
	rpcPort int
)

func rpcServerRun(cmd *cobra.Command, args []string) error {
	endpoint := fmt.Sprintf("%s:%d", rpcHost, rpcPort)
	log.Printf("Starting RPC server on %s", endpoint)

	// Run ggml_backend_rpc_start_server in a goroutine
	// so we can handle signals and potentially stop it later if needed.
	// Note: ggml_backend_rpc_start_server is likely a blocking call.
	go func() {
		C.run_rpc_server()
	}()

	log.Printf("RPC server started with Metal backend on %s. Press Ctrl+C to exit.", endpoint)

	// Wait for an interrupt signal
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	<-sigChan

	log.Println("Shutting down RPC server...")
	// Add any cleanup logic here if ggml_backend_rpc_start_server has a corresponding stop function.
	// For now, we assume exiting the program stops the server.
	return nil
}
