// Package model_test provides external tests for the model package.
// This test file specifically tests the forward pass functionality on models.
// It is in a separate package (model_test) to avoid import cycles while still
// being able to test the public API of the model package.
package model_test

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/sample"

	_ "github.com/ollama/ollama/model/models"
)

type modelTest struct {
	Prompt            string   `json:"prompt"`
	OutputContainsOne []string `json:"output_contains_one"`
}

func TestForwardSimple(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in short mode")
	}

	// Read all JSON files from testdata/models
	files, err := os.ReadDir("testdata/models")
	if err != nil {
		t.Fatal(err)
	}

	for _, file := range files {
		if !strings.HasSuffix(file.Name(), ".json") {
			continue
		}

		jsonPath := filepath.Join("testdata/models", file.Name())
		ggufPath := filepath.Join("testdata/models", strings.TrimSuffix(file.Name(), ".json")+".gguf")

		// Skip if no corresponding .gguf file exists
		if _, err := os.Stat(ggufPath); err != nil {
			t.Logf("skipping %s: no corresponding GGUF file found", file.Name())
			continue
		}

		data, err := os.ReadFile(jsonPath)
		if err != nil {
			t.Fatal(err)
		}

		var test modelTest
		if err := json.Unmarshal(data, &test); err != nil {
			t.Fatal(err)
		}

		t.Run(strings.TrimSuffix(file.Name(), ".json"), func(t *testing.T) {
			m, err := model.New(ggufPath)
			if err != nil {
				t.Fatal(err)
			}

			m.Config().Cache.Init(m.Backend(), ml.DTypeF32, 2048)

			inputs, err := m.(model.TextProcessor).Encode(test.Prompt)
			if err != nil {
				t.Fatal(err)
			}

			var result []string
			for len(result) < 100 { // Limit to 100 tokens max
				options := model.Options{
					Inputs:    inputs,
					Positions: make([]int32, len(inputs)),
					Sequences: make([]int, len(inputs)),
					Outputs:   []int32{int32(len(inputs) - 1)},
				}
				for i := range options.Positions {
					options.Positions[i] = int32(i)
					options.Sequences[i] = 0
				}

				ctx := m.Backend().NewContext()

				modelOutput, err := model.Forward(ctx, m, options)
				if err != nil {
					ctx.Close()
					t.Fatal(fmt.Errorf("forward pass failed: %v", err))
				}

				f32s := modelOutput.Floats()
				logits := make([]float64, len(f32s))
				for i, f32 := range f32s {
					logits[i] = float64(f32)
				}

				token, err := sample.Sample(logits, sample.Greedy())
				if err != nil {
					ctx.Close()
					t.Fatal(fmt.Errorf("sampling failed: %v", err))
				}

				ctx.Close()

				// Greedy sampling: take the token with the highest logit
				nextToken := int32(token[0])
				if m.(model.TextProcessor).Is(nextToken, model.SpecialEOS) {
					break
				}

				piece, err := m.(model.TextProcessor).Decode([]int32{nextToken})
				if err != nil {
					t.Fatal(err)
				}

				result = append(result, piece)
				output := strings.Join(result, "")

				for _, expectedOutput := range test.OutputContainsOne {
					if strings.Contains(output, expectedOutput) {
						t.Logf("Test passed with output: %q (matched expected: %q)", output, expectedOutput)
						return
					}
				}

				// Maintain full context by appending new token
				inputs = append(inputs, nextToken)
			}

			t.Fatalf("Expected output containing one of %q but got: %q", test.OutputContainsOne, strings.Join(result, ""))
		})
	}
}
