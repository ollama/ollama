//go:build mlx

package main

import (
	"context"
	"fmt"
	"time"
	"unicode/utf8"

	"github.com/ollama/ollama/x/imagegen/cache"
	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/imagegen/tokenizer"
)

// Dedicated stream for generation (like mlx-lm's generation_stream)
var generationStream *mlx.Stream

// utf8Streamer buffers decoded text and emits only complete UTF-8 characters.
// This handles cases where tokenizers output partial multi-byte sequences.
type utf8Streamer struct {
	buffer []byte
}

// Write adds decoded text to the buffer and returns complete UTF-8 characters.
func (s *utf8Streamer) Write(text string) string {
	s.buffer = append(s.buffer, text...)

	// Find the last position that ends with a complete UTF-8 character
	validLen := 0
	for i := 0; i < len(s.buffer); {
		r, size := utf8.DecodeRune(s.buffer[i:])
		if r == utf8.RuneError && size == 1 {
			// Invalid or incomplete UTF-8 sequence at this position
			// Check if it could be a valid start of a multi-byte sequence
			if len(s.buffer)-i < 4 {
				// Might be incomplete, keep it in buffer
				break
			}
			// Definitely invalid, skip this byte
			i++
			validLen = i
		} else {
			i += size
			validLen = i
		}
	}

	if validLen == 0 {
		return ""
	}

	result := string(s.buffer[:validLen])
	s.buffer = s.buffer[validLen:]
	return result
}

// Flush returns any remaining buffered bytes (may be incomplete UTF-8).
func (s *utf8Streamer) Flush() string {
	if len(s.buffer) == 0 {
		return ""
	}
	result := string(s.buffer)
	s.buffer = nil
	return result
}

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

type Model interface {
	Tokenizer() *tokenizer.Tokenizer
	VocabSize() int32
	NewCache(maxSeqLen int32) []cache.Cache
	Forward(input *mlx.Array, caches []cache.Cache) *mlx.Array
}

// ChatModel is an optional interface for models that support chat formatting
type ChatModel interface {
	FormatPrompt(prompt string) string
}

// MultimodalModel is for models that support image input
type MultimodalModel interface {
	Model
	FormatPromptWithImage(prompt string) string
	ExpandImageTokens(tokens []int32) []int32
	ForwardWithImage(tokens *mlx.Array, image *mlx.Array, caches []cache.Cache) *mlx.Array
	ImageSize() int32 // Returns expected image size for preprocessing
}

// ImageLoader loads and preprocesses an image for multimodal models
// Returns nil if path is empty
type ImageLoader func(path string, imageSize int32) (*mlx.Array, error)

type input struct {
	Prompt       string
	Image        *mlx.Array // Optional preprocessed image for multimodal models
	MaxTokens    int
	Temperature  float32
	TopP         float32
	TopK         int
	WiredLimitGB int // Metal wired memory limit in GB (default 32)
}

type output struct {
	Text          string
	Done          bool
	PrefillTokSec float64
	GenTokSec     float64
}

// Decoder wraps model + cache for autoregressive generation.
type Decoder struct {
	model         Model
	caches        []cache.Cache
	vocabSize     int32
	temp          float32
	topK          int
	topP          float32
	token         *mlx.Array   // Current token (kept across pools)
	oldCacheState []*mlx.Array // Preallocated slice for old cache state
	image         *mlx.Array   // Optional image for multimodal prefill
}

func NewDecoder(m Model, temp float32, topK int, topP float32) *Decoder {
	caches := m.NewCache(0)
	return &Decoder{
		model:         m,
		caches:        caches,
		vocabSize:     m.VocabSize(),
		temp:          temp,
		topK:          topK,
		topP:          topP,
		oldCacheState: make([]*mlx.Array, 0, len(caches)*2),
	}
}

// SetImage sets the image for multimodal prefill (call before prefill)
func (d *Decoder) SetImage(img *mlx.Array) {
	d.image = img
}

func (d *Decoder) prefill(inputIDs []int32) int {
	processed := 0

	// Track old cache state to free after each chunk
	var oldCacheState []*mlx.Array

	// For multimodal models with an image, we need to process all tokens together
	// in the first forward pass so the image embeddings can be inserted properly.
	// Skip chunking for multimodal prefill.
	isMultimodal := d.image != nil

	// Process all-but-1 tokens in chunks, eval cache state for memory management
	// Skip chunking for multimodal - process everything in the final step
	if !isMultimodal {
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
	}

	// Save old cache state before final step
	oldCacheState = oldCacheState[:0]
	for _, c := range d.caches {
		oldCacheState = append(oldCacheState, c.State()...)
	}

	// Final token + sampling (or all tokens for multimodal)
	withStream(func() {
		x := mlx.NewArrayInt32(inputIDs, []int32{1, int32(len(inputIDs))})
		mlx.Eval(x) // Materialize before any other evals

		var logits *mlx.Array
		// Use ForwardWithImage if we have an image and model supports it
		if d.image != nil {
			if mm, ok := d.model.(MultimodalModel); ok {
				logits = mm.ForwardWithImage(x, d.image, d.caches)
				d.image = nil // Only use image for first forward
			} else {
				logits = d.model.Forward(x, d.caches)
			}
		} else {
			logits = d.model.Forward(x, d.caches)
		}
		d.token = sample(logits, d.temp, d.topK, d.topP, d.vocabSize)
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
		d.token = sample(logits, d.temp, d.topK, d.topP, d.vocabSize)
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

func generate(ctx context.Context, m Model, in input, cb func(output)) error {
	mlx.EnableCompile()
	wiredLimit := in.WiredLimitGB
	if wiredLimit <= 0 {
		wiredLimit = 32 // default 32GB
	}
	mlx.MetalSetWiredLimit(uint64(wiredLimit) << 30)

	temp := in.Temperature
	if temp < 0 {
		temp = 0.7
	}

	tok := m.Tokenizer()
	dec := NewDecoder(m, temp, in.TopK, in.TopP)

	// Apply chat template - use image template if we have an image
	prompt := in.Prompt
	var tokens []int32
	if mm, ok := m.(MultimodalModel); ok && in.Image != nil {
		prompt = mm.FormatPromptWithImage(prompt)
		tokens = tok.Encode(prompt, true)
		tokens = mm.ExpandImageTokens(tokens) // Expand <start_of_image> to 256 image tokens
		dec.SetImage(in.Image)
	} else if cm, ok := m.(ChatModel); ok {
		prompt = cm.FormatPrompt(prompt)
		tokens = tok.Encode(prompt, true)
	} else {
		tokens = tok.Encode(prompt, true)
	}

	prefillStart := time.Now()
	prefillTokens := dec.prefill(tokens)
	// Prefill measurement should include time to first token (like mlx-lm)
	// Step() waits for prefill to complete and returns first token
	firstToken := dec.step()
	prefillTokSec := float64(prefillTokens) / time.Since(prefillStart).Seconds()

	genStart := time.Now()
	maxTokens := max(in.MaxTokens, 100)
	var genTokens int

	// UTF-8 streamer to handle partial multi-byte characters
	streamer := &utf8Streamer{}

	// Handle first token
	genTokens++
	if tok.IsEOS(firstToken) {
		cb(output{Done: true, PrefillTokSec: prefillTokSec, GenTokSec: 0})
		return nil
	}
	if text := streamer.Write(tok.Decode([]int32{firstToken})); text != "" {
		cb(output{Text: text})
	}

	for n := 1; n < maxTokens; n++ {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		token := dec.step()
		genTokens++

		if tok.IsEOS(token) {
			break
		}
		if text := streamer.Write(tok.Decode([]int32{token})); text != "" {
			cb(output{Text: text})
		}

		if n%256 == 0 {
			mlx.ClearCache()
		}
	}

	// Flush any remaining buffered bytes
	if text := streamer.Flush(); text != "" {
		cb(output{Text: text})
	}

	fmt.Printf("\nPeak memory: %.2fGB\n", float64(mlx.MetalGetPeakMemory())/(1<<30))
	cb(output{Done: true, PrefillTokSec: prefillTokSec,
		GenTokSec: float64(genTokens) / time.Since(genStart).Seconds()})
	return nil
}
