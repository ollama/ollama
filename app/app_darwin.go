package main

// #cgo CFLAGS: -x objective-c
// #cgo LDFLAGS: -framework Cocoa -framework LocalAuthentication -framework ServiceManagement
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

	// Ask to move to applications directory
	moving := C.askToMoveToApplications()
	if moving {
		return
	}

	C.killOtherInstances()

	code := C.installSymlink()
	if code != 0 {
		slog.Error("Failed to install symlink")
	}

	exe, err := os.Executable()
	if err != nil {
		panic(err)
	}

	var options ServerOptions

	ctx, cancel := context.WithCancel(context.Background())
	var done chan int

	done, err = SpawnServer(ctx, filepath.Join(filepath.Dir(exe), "..", "Resources", "ollama"), options)
	if err != nil {
		slog.Error(fmt.Sprintf("Failed to spawn ollama server %s", err))
		done = make(chan int, 1)
		done <- 1
	}

	// Run the native macOS app
	// Note: this will block until the app is closed
	C.run()

	slog.Info("ollama macOS app closed")

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
