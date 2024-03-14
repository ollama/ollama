package main

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"sync"
	"syscall"

	"github.com/jmorganca/ollama/gpu"
	"github.com/jmorganca/ollama/llm"
	"github.com/jmorganca/ollama/server"
)

type ollama struct {
	ctx  context.Context
	ctxC context.CancelFunc

	pullLock sync.RWMutex
	pulls    map[uint64]*pull

	generateLock sync.RWMutex
	generateJobs map[uint64]*genJob

	workdir string
}

func new(ctx context.Context, workdir string) *ollama {
	s := &ollama{
		pulls:        make(map[uint64]*pull),
		generateJobs: make(map[uint64]*genJob),
		workdir:      workdir,
	}
	s.ctx, s.ctxC = context.WithCancel(ctx)

	os.Setenv("OLLAMA_MODELS", workdir)

	return s
}

func (s *ollama) init() error {
	level := slog.LevelInfo
	if debug := os.Getenv("OLLAMA_DEBUG"); debug != "" {
		level = slog.LevelDebug
	}

	handler := slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{
		Level:     level,
		AddSource: true,
		ReplaceAttr: func(_ []string, attr slog.Attr) slog.Attr {
			if attr.Key == slog.SourceKey {
				source := attr.Value.Any().(*slog.Source)
				source.File = filepath.Base(source.File)
			}

			return attr
		},
	})

	slog.SetDefault(slog.New(handler))

	if noprune := os.Getenv("OLLAMA_NOPRUNE"); noprune == "" {
		// clean up unused layers and manifests
		if err := server.PruneLayers(); err != nil {
			return err
		}

		manifestsPath, err := server.GetManifestPath()
		if err != nil {
			return err
		}

		if err := server.PruneDirectory(manifestsPath); err != nil {
			return err
		}
	}

	// listen for a ctrl+c and stop any loaded llm
	signals := make(chan os.Signal, 1)
	signal.Notify(signals, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-signals
		// if loaded.runner != nil {
		// 	loaded.runner.Close()
		// }
		// os.RemoveAll(s.workdir)
		os.Exit(0)
	}()

	if err := llm.Init(); err != nil {
		return fmt.Errorf("unable to initialize llm library %w", err)
	}
	if runtime.GOOS == "linux" { // TODO - windows too
		// check compatibility to log warnings
		if _, err := gpu.CheckVRAM(); err != nil {
			slog.Info(err.Error())
		}
	}
	return nil
}
