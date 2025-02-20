package backend

import (
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"testing"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/server"

	_ "github.com/ollama/ollama/model/models/llama"
)

var modelName = flag.String("m", "", "Name of the model to benchmark")

func suppressOutput() (cleanup func()) {
	oldStdout, oldStderr := os.Stdout, os.Stderr
	os.Stdout, os.Stderr = nil, nil
	log.SetOutput(io.Discard)

	return func() {
		os.Stdout, os.Stderr = oldStdout, oldStderr
		log.SetOutput(os.Stderr)
	}
}

func setupModel(b *testing.B) model.Model {
	if *modelName == "" {
		b.Fatal("Error: -m flag is required for benchmark tests")
	}

	sm, err := server.GetModel(*modelName)
	if err != nil {
		b.Fatal(err)
	}

	m, err := model.New(sm.ModelPath)
	if err != nil {
		b.Fatal(err)
	}

	m.Config().Cache.Init(m.Backend(), ml.DTypeF32, 2048)
	return m
}

func BenchmarkGGMLOperations(b *testing.B) {
	// loading the GGML back-end logs to standard out and makes the bench output messy
	cleanup := suppressOutput()
	defer cleanup()

	b.Setenv("OLLAMA_BENCHMARK", "1")
	b.Setenv("OLLAMA_BACKEND", "ggml")

	m := setupModel(b)

	// Sample input data
	inputIDs := []int32{1, 2, 3, 4, 5}
	options := model.Options{
		Inputs:    inputIDs,
		Positions: []int32{1, 2, 3, 4, 5},
		Sequences: []int{1, 1, 1, 1, 1},
		Outputs:   []int32{int32(len(inputIDs) - 1)},
	}

	b.ResetTimer()

	for range b.N {
		ctx := m.Backend().NewContext()
		defer ctx.Close()

		modelOutput, err := model.Forward(ctx, m, options)
		if err != nil {
			b.Fatal(fmt.Errorf("forward pass failed: %v", err))
		}

		ctx.Compute(modelOutput)

		for _, op := range ctx.Timing() {
			b.ReportMetric(op.Duration, fmt.Sprintf("%s_ms", op.Type))
		}
	}
}
