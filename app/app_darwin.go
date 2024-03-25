package main

// #cgo CFLAGS: -x objective-c -Wno-deprecated-declarations
// #cgo LDFLAGS: -framework Cocoa -framework LocalAuthentication
// #include "app_darwin.h"
import "C"
import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"syscall"
)

func init() {
	home, err := os.UserHomeDir()
	if err != nil {
		panic(err)
	}

	ServerLogFile = filepath.Join(home, ".ollama", "logs", "server.log")
}

func run() {
	initLogging()
	slog.Info("ollama macOS app started")

	C.killOtherInstances()

	exe, err := os.Executable()
	if err != nil {
		panic(err)
	}

	resources := filepath.Join(filepath.Dir(exe), "..", "Resources")

	ctx, cancel := context.WithCancel(context.Background())
	var done chan int

	done, err = SpawnServer(ctx, filepath.Join(resources, "ollama"))
	if err != nil {
		slog.Error(fmt.Sprintf("Failed to spawn ollama server %s", err))
		done = make(chan int, 1)
		done <- 1
	}

	// Run the native app
	C.run()

	cancel()
	slog.Info("Waiting for ollama server to shutdown...")
	if done != nil {
		<-done
	}
	slog.Info("Ollama app exiting")
}

//export Quit
func Quit() {
	syscall.Kill(os.Getpid(), syscall.SIGTERM)
}
