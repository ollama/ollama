//go:build mlx

package imagegen

import (
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/x/imagegen/cache"
	"github.com/ollama/ollama/x/imagegen/manifest"
	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/imagegen/models/glm4_moe_lite"
	"github.com/ollama/ollama/x/imagegen/tokenizer"
)

// TextModel is the interface for LLM text generation models.
type TextModel interface {
	Forward(tokens *mlx.Array, caches []cache.Cache) *mlx.Array
	NewCache(maxSeqLen int32) []cache.Cache
	Tokenizer() *tokenizer.Tokenizer
	VocabSize() int32
	MaxContextLength() int32
	NumLayers() int
}

// llmState holds the state for LLM generation
type llmState struct {
	model TextModel
}

var llmMu sync.Mutex

// Dedicated stream for generation (like mlx-lm's generation_stream)
var generationStream *mlx.Stream

// withStream runs fn with the generation stream as default
func withStream(fn func()) {
	// Lazy initialization of generationStream
	if generationStream == nil {
		generationStream = mlx.NewStream()
	}
	orig := mlx.GetDefaultStream()
	mlx.SetDefaultStream(generationStream)
	fn()
	mlx.SetDefaultStream(orig)
}

// Decoder wraps model + cache for autoregressive generation.
// This matches the pattern from cmd/engine/generate.go
type Decoder struct {
	model         TextModel
	caches        []cache.Cache
	vocabSize     int32
	temp          float32
	token         *mlx.Array   // Current token (kept across iterations)
	oldCacheState []*mlx.Array // Preallocated slice for old cache state
}

func NewDecoder(m TextModel, temp float32) *Decoder {
	caches := m.NewCache(0)
	return &Decoder{
		model:         m,
		caches:        caches,
		vocabSize:     m.VocabSize(),
		temp:          temp,
		oldCacheState: make([]*mlx.Array, 0, len(caches)*2),
	}
}

func (d *Decoder) prefill(inputIDs []int32) int {
	processed := 0

	// Track old cache state to free after each chunk
	var oldCacheState []*mlx.Array

	// Process all-but-1 tokens in chunks, eval cache state for memory management
	for len(inputIDs) > 1 {
		chunkSize := min(2048, len(inputIDs)-1)
		if chunkSize <= 0 {
			break
		}
		chunk := inputIDs[:chunkSize]

		// Save old cache state before forward
		oldCacheState = oldCacheState[:0]
		for _, c := range d.caches {
			oldCacheState = append(oldCacheState, c.State()...)
		}

		var cacheState []*mlx.Array
		withStream(func() {
			x := mlx.NewArrayInt32(chunk, []int32{1, int32(len(chunk))})
			d.model.Forward(x, d.caches)
			for _, c := range d.caches {
				cacheState = append(cacheState, c.State()...)
			}
		})
		mlx.Eval(cacheState...)

		// Free old cache state
		for _, arr := range oldCacheState {
			if arr != nil {
				arr.Free()
			}
		}

		inputIDs = inputIDs[chunkSize:]
		processed += chunkSize
	}

	// Save old cache state before final step
	oldCacheState = oldCacheState[:0]
	for _, c := range d.caches {
		oldCacheState = append(oldCacheState, c.State()...)
	}

	// Final token + sampling
	withStream(func() {
		x := mlx.NewArrayInt32(inputIDs, []int32{1, int32(len(inputIDs))})
		mlx.Eval(x) // Materialize before any other evals
		logits := d.model.Forward(x, d.caches)
		d.token = sample(logits, d.temp, d.vocabSize)
	})
	// Keep cache state (token auto-kept by AsyncEval)
	for _, c := range d.caches {
		mlx.Keep(c.State()...)
	}
	mlx.AsyncEval(d.token)

	// Free old cache state from before final step
	for _, arr := range oldCacheState {
		if arr != nil {
			arr.Free()
		}
	}

	mlx.ClearCache()

	return processed + len(inputIDs)
}

func (d *Decoder) step() int32 {
	prevToken := d.token

	// Save old cache state (reuse preallocated slice)
	d.oldCacheState = d.oldCacheState[:0]
	for _, c := range d.caches {
		d.oldCacheState = append(d.oldCacheState, c.State()...)
	}

	withStream(func() {
		logits := d.model.Forward(mlx.Reshape(prevToken, 1, 1), d.caches)
		d.token = sample(logits, d.temp, d.vocabSize)
	})
	// Keep token and new cache state so they survive cleanup
	mlx.Keep(d.token)
	for _, c := range d.caches {
		mlx.Keep(c.State()...)
	}
	mlx.AsyncEval(d.token)

	// Sync on previous token (GPU already working on next step)
	val := prevToken.ItemInt32()

	// Free old token and old cache state
	prevToken.Free()
	for _, arr := range d.oldCacheState {
		arr.Free()
	}
	return val
}

// sample samples from logits using temperature scaling
func sample(logits *mlx.Array, temp float32, vocabSize int32) *mlx.Array {
	// Get last position logits: [1, L, vocab] -> [vocab]
	shape := logits.Shape()
	seqLen := shape[1]
	lastLogits := mlx.Slice(logits, []int32{0, seqLen - 1, 0}, []int32{1, seqLen, vocabSize})
	lastLogits = mlx.Reshape(lastLogits, vocabSize)

	if temp <= 0 || temp < 0.01 {
		// Greedy decoding
		return mlx.Argmax(lastLogits, -1, false)
	}

	// Apply temperature scaling
	scaled := mlx.DivScalar(lastLogits, temp)
	return mlx.RandomCategorical(scaled, -1, 1)
}

// loadLLMModel loads a safetensors LLM model and its tokenizer from manifest storage.
func (s *server) loadLLMModel() error {
	// Load the manifest to get model information
	modelManifest, err := manifest.LoadManifest(s.modelName)
	if err != nil {
		return fmt.Errorf("failed to load manifest: %w", err)
	}

	// Detect model architecture from config.json
	configData, err := modelManifest.ReadConfig("config.json")
	if err != nil {
		return fmt.Errorf("failed to read config.json: %w", err)
	}

	var modelConfig struct {
		Architectures []string `json:"architectures"`
		ModelType     string   `json:"model_type"`
	}
	if err := json.Unmarshal(configData, &modelConfig); err != nil {
		return fmt.Errorf("failed to parse config.json: %w", err)
	}

	arch := ""
	if len(modelConfig.Architectures) > 0 {
		arch = modelConfig.Architectures[0]
	}
	if arch == "" {
		arch = modelConfig.ModelType
	}

	slog.Info("detected LLM architecture", "architecture", arch, "model_type", modelConfig.ModelType)

	// Load the appropriate model based on architecture
	var model TextModel
	archLower := strings.ToLower(arch)

	switch {
	case strings.Contains(archLower, "glm4moelite"):
		m, err := glm4_moe_lite.LoadFromManifest(modelManifest)
		if err != nil {
			return fmt.Errorf("failed to load glm4-moe-lite model: %w", err)
		}
		model = m
		slog.Info("loaded glm4-moe-lite model", "vocab_size", m.VocabSize(), "layers", m.NumLayers())

	default:
		return fmt.Errorf("LLM architecture %q is not yet supported. "+
			"Supported architectures: glm4-moe-lite. "+
			"Please convert your model to GGUF format or use a supported architecture", arch)
	}

	s.llmModel = &llmState{
		model: model,
	}

	return nil
}

// handleLLMCompletion handles LLM text generation requests.
func (s *server) handleLLMCompletion(w http.ResponseWriter, r *http.Request, req Request) {
	if s.llmModel == nil {
		http.Error(w, "LLM model not loaded", http.StatusInternalServerError)
		return
	}

	// Serialize generation requests
	llmMu.Lock()
	defer llmMu.Unlock()

	if err := s.llmGenerate(w, r, req); err != nil {
		slog.Error("LLM generation failed", "error", err)
		// Don't send error if we've already started streaming
	}
}

// llmGenerate runs the generation loop using the Decoder pattern from cmd/engine
func (s *server) llmGenerate(w http.ResponseWriter, r *http.Request, req Request) error {
	state := s.llmModel

	// Set up streaming response
	w.Header().Set("Content-Type", "application/x-ndjson")
	w.Header().Set("Transfer-Encoding", "chunked")
	flusher, ok := w.(http.Flusher)
	if !ok {
		return errors.New("streaming not supported")
	}

	tok := state.model.Tokenizer()

	// The prompt is already formatted by the server using the model's renderer
	// (see server/prompt.go renderPrompt), so we don't apply FormatPrompt here.
	prompt := req.Prompt

	// Tokenize the prompt
	inputIDs := tok.Encode(prompt, true)
	slog.Debug("tokenized prompt", "num_tokens", len(inputIDs))

	// Generation parameters
	maxTokens := int(state.model.MaxContextLength())
	if maxTokens <= 0 {
		maxTokens = 4096
	}
	if req.Options != nil && req.Options.NumPredict > 0 {
		maxTokens = req.Options.NumPredict
	}

	temperature := float32(0.7)
	if req.Options != nil && req.Options.Temperature > 0 {
		temperature = float32(req.Options.Temperature)
	}

	// Enable MLX compilation for better performance
	mlx.EnableCompile()

	// Create decoder with fresh caches
	dec := NewDecoder(state.model, temperature)

	prefillStart := time.Now()
	prefillTokens := dec.prefill(inputIDs)
	// Prefill measurement includes time to first token
	firstToken := dec.step()
	prefillDuration := time.Since(prefillStart)
	promptEvalDuration := prefillDuration

	enc := json.NewEncoder(w)
	ctx := r.Context()
	generated := 0
	stopReason := "max_tokens"

	// Handle first token
	generated++
	if tok.IsEOS(firstToken) {
		resp := Response{
			Done:               true,
			StopReason:         fmt.Sprintf("first_token_eos:%d", firstToken),
			PromptEvalCount:    prefillTokens,
			PromptEvalDuration: int(promptEvalDuration.Nanoseconds()),
		}
		enc.Encode(resp)
		flusher.Flush()
		return nil
	}

	text := tok.Decode([]int32{firstToken})
	resp := Response{Content: text}
	enc.Encode(resp)
	flusher.Flush()

	genStart := time.Now()

	// Generation loop
	for n := 1; n < maxTokens; n++ {
		// Check for cancellation
		select {
		case <-ctx.Done():
			stopReason = fmt.Sprintf("context_cancelled:%d", generated)
			break
		default:
		}
		if stopReason != "max_tokens" {
			break
		}

		token := dec.step()
		generated++

		if tok.IsEOS(token) {
			stopReason = fmt.Sprintf("eos_token:%d", token)
			break
		}

		text := tok.Decode([]int32{token})

		// Check for stop sequences
		if req.Options != nil && len(req.Options.Stop) > 0 {
			shouldStop := false
			var matchedStop string
			for _, stop := range req.Options.Stop {
				if strings.Contains(text, stop) {
					text = strings.Split(text, stop)[0]
					shouldStop = true
					matchedStop = stop
					break
				}
			}
			if shouldStop {
				if text != "" {
					resp := Response{Content: text}
					enc.Encode(resp)
					flusher.Flush()
				}
				stopReason = fmt.Sprintf("stop_sequence:%s", matchedStop)
				break
			}
		}

		resp := Response{Content: text}
		enc.Encode(resp)
		flusher.Flush()

		// Periodically clear MLX cache
		if n%256 == 0 {
			mlx.ClearCache()
		}
	}

	// Clean up
	mlx.ClearCache()

	// Send final response with stats
	evalDuration := time.Since(genStart)
	resp = Response{
		Done:               true,
		StopReason:         fmt.Sprintf("%s:generated=%d", stopReason, generated),
		PromptEvalCount:    prefillTokens,
		PromptEvalDuration: int(promptEvalDuration.Nanoseconds()),
		EvalCount:          generated,
		EvalDuration:       int(evalDuration.Nanoseconds()),
	}
	enc.Encode(resp)
	flusher.Flush()

	return nil
}
