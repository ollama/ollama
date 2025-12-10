package olmo

import (
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"log/slog"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/model/input"
	typemodel "github.com/ollama/ollama/types/model"
)

var args struct {
	model,
	prompt string
	layers int
}

func TestMain(m *testing.M) {
	flag.StringVar(&args.model, "model", "", "path to model (e.g., olmo3:latest)")
	flag.StringVar(&args.prompt, "prompt", "Hello, how are", "model prompt")
	flag.IntVar(&args.layers, "layers", math.MaxInt, "num of gpu layers")
	flag.Parse()

	slog.SetDefault(logutil.NewLogger(os.Stderr, envconfig.LogLevel()))

	os.Exit(m.Run())
}

func blob(tb testing.TB, modelName string) string {
	tb.Helper()

	models := envconfig.Models()
	manifest, err := os.Open(filepath.Join(models, "manifests", typemodel.ParseName(modelName).Filepath()))
	if err != nil {
		tb.Fatal(err)
	}
	defer manifest.Close()

	var m struct {
		Layers []struct {
			MediaType string `json:"mediaType"`
			Digest    string `json:"digest"`
		} `json:"layers"`
	}

	if err := json.NewDecoder(manifest).Decode(&m); err != nil {
		tb.Fatal(err)
	}

	for _, layer := range m.Layers {
		if layer.MediaType == "application/vnd.ollama.image.model" {
			tb.Log("using model blob", layer.Digest)
			return filepath.Join(models, "blobs", strings.ReplaceAll(layer.Digest, ":", "-"))
		}
	}

	tb.Fatal("model blob not found")
	return ""
}

func loadFloatsFromBinary(filename string) ([]float32, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	fi, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if fi.Size()%4 != 0 {
		return nil, fmt.Errorf("file size %d not multiple of 4", fi.Size())
	}

	n := int(fi.Size() / 4)
	floats := make([]float32, n)
	if err := binary.Read(f, binary.LittleEndian, floats); err != nil {
		return nil, err
	}
	return floats, nil
}

func TestTokenization(t *testing.T) {
	if args.model == "" {
		t.Skip("no model specified, use -model flag")
	}

	m, err := model.New(blob(t, args.model), ml.BackendParams{AllocMemory: true})
	if err != nil {
		t.Fatal(err)
	}
	if err := m.Backend().Load(t.Context(), func(float32) {}); err != nil {
		t.Fatal(err)
	}

	prompt := args.prompt
	if prompt == "" {
		prompt = "hello"
	}

	tp := m.(model.TextProcessor)
	tokens, err := tp.Encode(prompt, false)
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("prompt: %q", prompt)
	t.Logf("tokens: %v", tokens)
	t.Logf("num tokens: %d", len(tokens))

	decoded, err := tp.Decode(tokens)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("decoded: %q", decoded)
}

func TestAttentionForward(t *testing.T) {
	if args.model == "" {
		t.Skip("no model specified, use -model flag")
	}

	m, err := model.New(blob(t, args.model), ml.BackendParams{AllocMemory: true})
	if err != nil {
		t.Fatal(err)
	}

	if err := m.Backend().Load(t.Context(), func(float32) {}); err != nil {
		t.Fatal(err)
	}

	olmoModel := m.(*Model)
	t.Logf("Model options: hiddenSize=%d, numHeads=%d, numKVHeads=%d",
		olmoModel.hiddenSize, olmoModel.numHeads, olmoModel.numKVHeads)
	t.Logf("Layer 0 attention: %+v", olmoModel.Layers[0].SelfAttention)

	ctx := m.Backend().NewContext()

	// Create test hidden states: (hiddenSize, batchSize)
	batchSize := 4
	hiddenSize := olmoModel.hiddenSize
	hsFloats := make([]float32, hiddenSize*batchSize)
	for i := range hsFloats {
		hsFloats[i] = float32(i%100) / 100.0 // Simple test values
	}

	hiddenStates := ctx.Input().FromFloats(hsFloats, hiddenSize, batchSize)
	t.Logf("hiddenStates shape: %v", hiddenStates.Shape())

	positions := ctx.Input().FromInts([]int32{0, 1, 2, 3}, batchSize)

	// Test attention forward (without cache for simplicity)
	attentionBlock := olmoModel.Layers[0].SelfAttention
	isSWA := olmoModel.isSWALayer(0)
	t.Logf("Layer 0 isSWA: %v", isSWA)

	result := attentionBlock.Forward(ctx, hiddenStates, positions, nil, olmoModel, isSWA)
	result = result.Contiguous(ctx)
	ctx.Forward(result).Compute(result)

	t.Logf("Attention result shape: %v dtype: %v", result.Shape(), result.DType())

	// Optionally dump to file
	// if err := os.WriteFile("/tmp/olmo_attention_output.bin", result.Bytes(), 0644); err != nil {
	// 	t.Fatal(err)
	// }
}

func TestMLPForward(t *testing.T) {
	if args.model == "" {
		t.Skip("no model specified, use -model flag")
	}

	m, err := model.New(blob(t, args.model), ml.BackendParams{AllocMemory: true})
	if err != nil {
		t.Fatal(err)
	}

	if err := m.Backend().Load(t.Context(), func(float32) {}); err != nil {
		t.Fatal(err)
	}

	olmoModel := m.(*Model)
	ctx := m.Backend().NewContext()

	// Create test hidden states
	batchSize := 4
	hiddenSize := olmoModel.hiddenSize
	hsFloats := make([]float32, hiddenSize*batchSize)
	for i := range hsFloats {
		hsFloats[i] = float32(i%100) / 100.0
	}

	hiddenStates := ctx.Input().FromFloats(hsFloats, hiddenSize, batchSize)
	t.Logf("hiddenStates shape: %v", hiddenStates.Shape())

	mlpBlock := olmoModel.Layers[0].MLP
	result := mlpBlock.Forward(ctx, hiddenStates, olmoModel)
	result = result.Contiguous(ctx)
	ctx.Forward(result).Compute(result)

	t.Logf("MLP result shape: %v dtype: %v", result.Shape(), result.DType())

	// Parse result bytes to float32
	resultBytes := result.Bytes()
	resultFloats := make([]float32, len(resultBytes)/4)
	for i := range resultFloats {
		bits := binary.LittleEndian.Uint32(resultBytes[i*4 : (i+1)*4])
		resultFloats[i] = math.Float32frombits(bits)
	}

	// Compute statistics
	var minVal, maxVal, sum float32
	minVal = resultFloats[0]
	maxVal = resultFloats[0]
	for _, v := range resultFloats {
		if v < minVal {
			minVal = v
		}
		if v > maxVal {
			maxVal = v
		}
		sum += v
	}
	mean := sum / float32(len(resultFloats))

	// Build readable output
	var sb strings.Builder
	sb.WriteString("# MLP Forward Output\n\n")
	sb.WriteString(fmt.Sprintf("# Input Shape: [%d, %d] (hiddenSize, batchSize)\n", hiddenSize, batchSize))
	sb.WriteString(fmt.Sprintf("# Output Shape: %v\n", result.Shape()))
	sb.WriteString(fmt.Sprintf("# DType: %v\n", result.DType()))
	sb.WriteString(fmt.Sprintf("# Layer: 0\n\n"))

	sb.WriteString("## Statistics\n\n")
	sb.WriteString(fmt.Sprintf("  Total elements: %d\n", len(resultFloats)))
	sb.WriteString(fmt.Sprintf("  Min: %v\n", minVal))
	sb.WriteString(fmt.Sprintf("  Max: %v\n", maxVal))
	sb.WriteString(fmt.Sprintf("  Mean: %v\n\n", mean))

	sb.WriteString("## Input Hidden States (first 20 values)\n\n")
	sb.WriteString("  [")
	for i := 0; i < min(20, len(hsFloats)); i++ {
		if i > 0 {
			sb.WriteString(", ")
		}
		sb.WriteString(fmt.Sprintf("%v", hsFloats[i]))
	}
	sb.WriteString("]\n\n")

	sb.WriteString("## Output Values\n\n")

	// Per-position output (each position in batch)
	for pos := 0; pos < batchSize; pos++ {
		sb.WriteString(fmt.Sprintf("Position %d (hiddenSize=%d values):\n", pos, hiddenSize))

		// Extract values for this position
		posStart := pos * hiddenSize
		posEnd := posStart + hiddenSize
		if posEnd > len(resultFloats) {
			posEnd = len(resultFloats)
		}
		posValues := resultFloats[posStart:posEnd]

		// Full tensor values
		sb.WriteString("  [")
		for i, v := range posValues {
			if i > 0 {
				sb.WriteString(", ")
			}
			sb.WriteString(fmt.Sprintf("%v", v))
		}
		sb.WriteString("]\n\n")
	}

	// Save to file
	if err := os.WriteFile("/tmp/olmo_mlp_forward.txt", []byte(sb.String()), 0644); err != nil {
		t.Fatal(err)
	}
	t.Log("Saved /tmp/olmo_mlp_forward.txt")

	// Also save binary
	if err := os.WriteFile("/tmp/olmo_mlp_forward.bin", resultBytes, 0644); err != nil {
		t.Fatal(err)
	}
	t.Log("Saved /tmp/olmo_mlp_forward.bin")

	// Print summary to console
	fmt.Println(sb.String())
}

func TestFullForward(t *testing.T) {
	if args.model == "" {
		t.Skip("no model specified, use -model flag")
	}

	m, err := model.New(blob(t, args.model), ml.BackendParams{AllocMemory: true})
	if err != nil {
		t.Fatal(err)
	}
	if err := m.Backend().Load(t.Context(), func(float32) {}); err != nil {
		t.Fatal(err)
	}

	ctx := m.Backend().NewContext()

	prompt := args.prompt
	if prompt == "" {
		prompt = "Hello, how are you?"
	}

	tp := m.(model.TextProcessor)
	tokens, err := tp.Encode(prompt, false)
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("prompt: %q", prompt)
	t.Logf("tokens: %v", tokens)

	decoded, err := tp.Decode(tokens)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("decoded: %q", decoded)

	seqLen := len(tokens)
	inputsTensor := ctx.Input().FromInts(tokens, seqLen)
	positions := make([]int32, seqLen)
	sequences := make([]int, seqLen)
	for i := range tokens {
		positions[i] = int32(i)
		sequences[i] = 0
	}

	// Output ALL positions
	outputIndices := make([]int32, seqLen)
	for i := range outputIndices {
		outputIndices[i] = int32(i)
	}
	outputs := ctx.Input().FromInts(outputIndices, seqLen)

	batch := input.Batch{
		Inputs:    inputsTensor,
		Positions: positions,
		Sequences: sequences,
		Outputs:   outputs,
	}

	// Initialize cache
	if cache := m.Config().Cache; cache != nil {
		cache.Init(m.Backend(), ml.DTypeF16, 1, 4096, seqLen)
	}

	result, err := model.Forward(ctx, m, batch)
	if err != nil {
		t.Fatal(err)
	}

	result = result.Contiguous(ctx)
	ctx.Forward(result).Compute(result)

	t.Logf("Forward pass completed, result shape: %v", result.Shape())

	// Dump logits to binary file
	if err := os.WriteFile("/tmp/olmo_logits.bin", result.Bytes(), 0644); err != nil {
		t.Fatal(err)
	}
	t.Log("Saved /tmp/olmo_logits.bin")

	// Parse logits from bytes for detailed analysis
	logitsBytes := result.Bytes()
	vocabSize := result.Shape()[0]

	// Read float32 values - shape is (vocab_size, seq_len)
	allLogits := make([]float32, len(logitsBytes)/4)
	for i := range allLogits {
		bits := binary.LittleEndian.Uint32(logitsBytes[i*4 : (i+1)*4])
		allLogits[i] = math.Float32frombits(bits)
	}

	// Create detailed text dump matching Python format
	var sb strings.Builder
	sb.WriteString("# Full Forward Logits\n\n")
	sb.WriteString(fmt.Sprintf("# Shape: [1, %d, %d]\n", seqLen, vocabSize))
	sb.WriteString(fmt.Sprintf("# Layout: (batch=1, seq_len=%d, vocab_size=%d)\n", seqLen, vocabSize))
	sb.WriteString(fmt.Sprintf("# Prompt: '%s'\n", prompt))
	sb.WriteString(fmt.Sprintf("# Tokens: %v\n\n", tokens))

	type logitPair struct {
		tokenID int
		value   float32
	}

	// Process each position
	for pos := 0; pos < seqLen; pos++ {
		// Extract logits for this position
		// Shape is (vocab_size, seq_len), so logits[v*seqLen + pos] gives logit for vocab v at position pos
		posLogits := make([]float32, vocabSize)
		for v := 0; v < vocabSize; v++ {
			posLogits[v] = allLogits[v*seqLen+pos]
		}

		// Find top 10 logits
		pairs := make([]logitPair, len(posLogits))
		for i, v := range posLogits {
			pairs[i] = logitPair{tokenID: i, value: v}
		}
		// Sort by value descending (simple bubble sort for small top-k)
		for i := 0; i < min(10, len(pairs)); i++ {
			for j := i + 1; j < len(pairs); j++ {
				if pairs[j].value > pairs[i].value {
					pairs[i], pairs[j] = pairs[j], pairs[i]
				}
			}
		}

		tokenStr, _ := tp.Decode([]int32{tokens[pos]})
		sb.WriteString(fmt.Sprintf("Position %d (token_id=%d, token='%s'):\n", pos, tokens[pos], tokenStr))
		sb.WriteString("  Top 10 logits:\n")
		for i := 0; i < min(10, len(pairs)); i++ {
			tokStr, _ := tp.Decode([]int32{int32(pairs[i].tokenID)})
			// Pad token string to 20 chars for alignment
			paddedTok := fmt.Sprintf("%-20s", fmt.Sprintf("'%s'", tokStr))
			sb.WriteString(fmt.Sprintf("    %d. token_id=%6d (%s): %f\n", i+1, pairs[i].tokenID, paddedTok, pairs[i].value))
		}

		// First 20 logits
		sb.WriteString("  Full logits (first 20): [")
		for i := 0; i < min(20, len(posLogits)); i++ {
			if i > 0 {
				sb.WriteString(", ")
			}
			sb.WriteString(fmt.Sprintf("%v", posLogits[i]))
		}
		sb.WriteString("]\n")

		// Last 20 logits
		sb.WriteString("  Full logits (last 20):  [")
		start := max(0, len(posLogits)-20)
		for i := start; i < len(posLogits); i++ {
			if i > start {
				sb.WriteString(", ")
			}
			sb.WriteString(fmt.Sprintf("%v", posLogits[i]))
		}
		sb.WriteString("]\n\n")
	}

	if err := os.WriteFile("/tmp/olmo_logits.txt", []byte(sb.String()), 0644); err != nil {
		t.Fatal(err)
	}
	t.Log("Saved /tmp/olmo_logits.txt")

	// Print to console as well
	fmt.Println(sb.String())
}

func TestRoPE(t *testing.T) {
	if args.model == "" {
		t.Skip("no model specified, use -model flag")
	}

	m, err := model.New(blob(t, args.model), ml.BackendParams{AllocMemory: true})
	if err != nil {
		t.Fatal(err)
	}
	if err := m.Backend().Load(t.Context(), func(float32) {}); err != nil {
		t.Fatal(err)
	}

	olmoModel := m.(*Model)

	// Test RoPE on a simple tensor
	headDim := olmoModel.hiddenSize / olmoModel.numHeads
	batchSize := 4
	numHeads := olmoModel.numHeads

	t.Logf("headDim: %d, numHeads: %d", headDim, numHeads)
	t.Logf("ropeBase: %f, ropeScale: %f, originalContextLength: %d",
		olmoModel.ropeBase, olmoModel.ropeScale, olmoModel.originalContextLength)

	// Create test query tensor: (headDim, numHeads, batchSize)
	queryFloats := make([]float32, headDim*numHeads*batchSize)
	for i := range queryFloats {
		queryFloats[i] = float32(i%100) / 100.0
	}

	// Test 1: Dump initial query values (fresh context)
	{
		ctx := m.Backend().NewContext()
		query := ctx.Input().FromFloats(queryFloats, headDim, numHeads, batchSize)
		t.Logf("query shape: %v", query.Shape())
		query = query.Contiguous(ctx)
		ctx.Forward(query).Compute(query)
		dump := ml.Dump(ctx, query, ml.DumpWithPrecision(6), ml.DumpWithThreshold(1000000))
		t.Logf("Query BEFORE RoPE sample values: %s", dump[:min(500, len(dump))])

		// Write to file
		header := fmt.Sprintf("Shape: %v\nDType: %v\n\n", query.Shape(), query.DType())
		if err := os.WriteFile("/tmp/olmo_query_before_rope.txt", []byte(header+dump), 0644); err != nil {
			t.Errorf("Failed to write file: %v", err)
		}
		if err := os.WriteFile("/tmp/olmo_query_before_rope.bin", query.Bytes(), 0644); err != nil {
			t.Errorf("Failed to write binary file: %v", err)
		}
		t.Log("Wrote /tmp/olmo_query_before_rope.txt and .bin")
	}

	// Test 2: SWA RoPE (fresh context)
	{
		ctx := m.Backend().NewContext()
		query := ctx.Input().FromFloats(queryFloats, headDim, numHeads, batchSize)
		positions := ctx.Input().FromInts([]int32{0, 1, 2, 3}, batchSize)
		resultSWA := olmoModel.applyRoPE(ctx, query, positions, headDim, true)
		resultSWA = resultSWA.Contiguous(ctx)
		ctx.Forward(resultSWA).Compute(resultSWA)
		t.Logf("SWA RoPE result shape: %v", resultSWA.Shape())
		dump := ml.Dump(ctx, resultSWA, ml.DumpWithPrecision(6), ml.DumpWithThreshold(1000000))
		t.Logf("Query AFTER SWA RoPE sample values: %s", dump[:min(500, len(dump))])

		// Write to file
		header := fmt.Sprintf("Shape: %v\nDType: %v\nfreqScale: 1.0 (SWA)\n\n", resultSWA.Shape(), resultSWA.DType())
		if err := os.WriteFile("/tmp/olmo_query_after_swa_rope.txt", []byte(header+dump), 0644); err != nil {
			t.Errorf("Failed to write file: %v", err)
		}
		if err := os.WriteFile("/tmp/olmo_query_after_swa_rope.bin", resultSWA.Bytes(), 0644); err != nil {
			t.Errorf("Failed to write binary file: %v", err)
		}
		t.Log("Wrote /tmp/olmo_query_after_swa_rope.txt and .bin")
	}

	// Test 3: Global (non-SWA) RoPE (fresh context)
	{
		ctx := m.Backend().NewContext()
		query := ctx.Input().FromFloats(queryFloats, headDim, numHeads, batchSize)
		positions := ctx.Input().FromInts([]int32{0, 1, 2, 3}, batchSize)
		resultGlobal := olmoModel.applyRoPE(ctx, query, positions, headDim, false)
		resultGlobal = resultGlobal.Contiguous(ctx)
		ctx.Forward(resultGlobal).Compute(resultGlobal)
		t.Logf("Global RoPE result shape: %v", resultGlobal.Shape())
		dump := ml.Dump(ctx, resultGlobal, ml.DumpWithPrecision(6), ml.DumpWithThreshold(1000000))
		t.Logf("Query AFTER Global RoPE sample values: %s", dump[:min(500, len(dump))])

		// Write to file
		header := fmt.Sprintf("Shape: %v\nDType: %v\nfreqScale: %f (Global, 1/ropeScale)\n\n",
			resultGlobal.Shape(), resultGlobal.DType(), 1.0/olmoModel.ropeScale)
		if err := os.WriteFile("/tmp/olmo_query_after_global_rope.txt", []byte(header+dump), 0644); err != nil {
			t.Errorf("Failed to write file: %v", err)
		}
		if err := os.WriteFile("/tmp/olmo_query_after_global_rope.bin", resultGlobal.Bytes(), 0644); err != nil {
			t.Errorf("Failed to write binary file: %v", err)
		}
		t.Log("Wrote /tmp/olmo_query_after_global_rope.txt and .bin")
	}
}
