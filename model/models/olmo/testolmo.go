package olmo

import (
	"context"
	"fmt"
	"log"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
	"github.com/ollama/ollama/sample"
)

func main() {
	modelPath := "/Users/nicole/models/Olmo-3-7B-Think/olmo-3-7b-think-q8_0.gguf"

	fmt.Println("Loading OLMo model...")
	m, err := model.New(modelPath, ml.BackendParams{AllocMemory: true})
	if err != nil {
		log.Fatal(err)
	}

	if err := m.Backend().Load(context.Background(), func(f float32) {}); err != nil {
		log.Fatal(err)
	}

	fmt.Println("✅ Model loaded successfully!")

	// Initialize the cache
	cache := m.Config().Cache
	if cache != nil {
		// Initialize with reasonable defaults:
		// - dtype: F16
		// - maxSequences: 1 (single sequence)
		// - capacity: 2048 (context length)
		// - maxBatch: 512
		cache.Init(m.Backend(), ml.DTypeF16, 1, 2048, 512)
		fmt.Printf("✅ Cache initialized (type: %T)\n", cache)
	}

	// Test generation
	prompt := "Question: What is machine learning? Answer:"
	fmt.Printf("\nPrompt: %s\n", prompt)

	tp := m.(model.TextProcessor)
	tokens, err := tp.Encode(prompt, true)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Tokens: %v (count: %d)\n", tokens, len(tokens))

	// Generate 20 tokens
	maxTokens := 20
	generated := make([]int32, 0, maxTokens)

	// Create sampler (temperature=0 for greedy sampling)
	sampler := sample.NewSampler(0, 0, 0, 0, -1, nil)

	for i := 0; i < maxTokens; i++ {
		// Create a new context for each generation step to avoid memory buildup
		ctx := m.Backend().NewContext()

		var inputTokens []int32
		var positions []int32

		if i == 0 {
			// First iteration: process all prompt tokens
			inputTokens = tokens
			positions = make([]int32, len(tokens))
			for j := range positions {
				positions[j] = int32(j)
			}
		} else {
			// Subsequent iterations: only process the newly generated token
			// The last token is at position len(tokens)-1 (its index in the sequence)
			inputTokens = []int32{tokens[len(tokens)-1]}
			positions = []int32{int32(len(tokens) - 1)}
		}

		sequences := make([]int, len(inputTokens))
		// All tokens belong to sequence 0

		inputsTensor := ctx.Input().FromInts(inputTokens, len(inputTokens))
		outputs := ctx.Input().FromInts([]int32{int32(len(inputTokens) - 1)}, 1)

		batch := input.Batch{
			Inputs:    inputsTensor,
			Positions: positions,
			Sequences: sequences,
			Outputs:   outputs,
		}

		// Forward pass (model.Forward handles cache.StartForward internally)
		logits, err := model.Forward(ctx, m, batch)
		if err != nil {
			ctx.Close()
			log.Fatal(err)
		}

		logits = logits.Contiguous(ctx)
		ctx.Forward(logits).Compute(logits)

		logitValues := logits.Floats()

		// Sample next token
		nextToken, err := sampler.Sample(logitValues)
		if err != nil {
			ctx.Close()
			log.Fatal(err)
		}

		// Close context before moving to next iteration
		ctx.Close()

		generated = append(generated, nextToken)
		tokens = append(tokens, nextToken)

		// Decode and print
		decoded, _ := tp.Decode([]int32{nextToken})
		fmt.Print(decoded)

		// Stop on EOS
		if nextToken == 2 || nextToken == 1 { // Common EOS tokens
			break
		}
	}

	fmt.Println("\n\n✅ Generation completed!")
	fullText, _ := tp.Decode(generated)
	fmt.Printf("Generated: %s\n", fullText)
}
