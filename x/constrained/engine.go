//go:build mlx

package constrained

import (
	"math"

	"github.com/ollama/ollama/x/imagegen/mlx"
)

// Engine applies grammar constraints to model outputs using MLX.
// It precomputes which tokens are valid for each terminal symbol,
// then builds masks dynamically based on PDA state.
type Engine struct {
	// PDA runtime for tracking state
	runtime *JSONRuntime

	// Token vocabulary from the model
	vocab []string

	// Symbol to token mask mapping
	// Key: terminal symbol (e.g., "{", "STRING", "true")
	// Value: MLX array of shape [vocab_size] with 1.0 for valid tokens, 0.0 for invalid
	symbolMasks map[string]*mlx.Array

	// Cached negative infinity mask for invalid tokens
	negInfMask *mlx.Array

	// Threshold for comparison (0.5 since mask values are 0 or 1)
	threshold *mlx.Array

	// Vocabulary size
	vocabSize int32
}

// NewEngine creates a new constrained decoding engine.
// vocab is the list of token strings from the model's tokenizer.
func NewEngine(vocab []string) (*Engine, error) {
	runtime, err := NewJSONRuntime()
	if err != nil {
		return nil, err
	}

	e := &Engine{
		runtime:     runtime,
		vocab:       vocab,
		symbolMasks: make(map[string]*mlx.Array),
		vocabSize:   int32(len(vocab)),
	}

	// Precompute symbol masks
	e.precomputeSymbolMasks()

	// Create the negative infinity mask and threshold (only if vocab is non-empty)
	if e.vocabSize > 0 {
		e.negInfMask = mlx.FullDtype(float32(math.Inf(-1)), mlx.DtypeFloat32, e.vocabSize)
		mlx.Keep(e.negInfMask)

		e.threshold = mlx.NewScalarArray(0.5)
		mlx.Keep(e.threshold)
	}

	return e, nil
}

// precomputeSymbolMasks builds a mask for each terminal symbol.
// Each mask is [vocab_size] with 1.0 where the token matches the symbol.
func (e *Engine) precomputeSymbolMasks() {
	if e.vocabSize == 0 {
		return // Nothing to precompute for empty vocabulary
	}

	// Get all terminal symbols from the grammar
	pda, _ := GetJSONPDA()
	terminals := pda.Terminals

	// For each terminal, find which tokens match it
	for _, terminal := range terminals {
		maskData := make([]float32, e.vocabSize)

		for i, token := range e.vocab {
			if e.tokenMatchesSymbol(token, terminal) {
				maskData[i] = 1.0
			}
		}

		mask := mlx.NewArray(maskData, []int32{e.vocabSize})
		mlx.Keep(mask)
		e.symbolMasks[terminal] = mask
	}
}

// tokenMatchesSymbol determines if a token can produce a given grammar symbol.
func (e *Engine) tokenMatchesSymbol(token, symbol string) bool {
	tokenType := ClassifyToken(token)
	expectedSymbol := TokenToGrammarSymbol(tokenType)
	return expectedSymbol == symbol
}

// ApplyMask applies grammar constraints to logits.
// Returns logits with invalid tokens set to -inf.
func (e *Engine) ApplyMask(logits *mlx.Array) *mlx.Array {
	// Get valid symbols from current PDA state
	validSymbols := e.runtime.ValidInputs()

	if len(validSymbols) == 0 {
		// No valid tokens - return all -inf
		return mlx.FullDtype(float32(math.Inf(-1)), mlx.DtypeFloat32, e.vocabSize)
	}

	// Build combined mask from valid symbols
	// Start with zeros (all invalid)
	combinedMask := mlx.Zeros([]int32{e.vocabSize})

	// OR together all symbol masks (using Max since values are 0 or 1)
	for _, sym := range validSymbols {
		if mask, ok := e.symbolMasks[sym]; ok {
			combinedMask = mlx.Max(combinedMask, mask)
		}
	}

	// Apply mask: where mask is 1, keep logits; where mask is 0, set to -inf
	// Where(condition, a, b) returns a where condition is true, b otherwise
	// We use mask >= 0.5 since mask values are 0.0 or 1.0
	condition := mlx.GreaterEqual(combinedMask, e.threshold)
	result := mlx.Where(condition, logits, e.negInfMask)

	return result
}

// Accept processes a token and updates the PDA state.
// Returns true if the token was valid and accepted.
func (e *Engine) Accept(tokenID int) bool {
	if tokenID < 0 || tokenID >= len(e.vocab) {
		return false
	}

	token := e.vocab[tokenID]
	return e.runtime.AcceptToken(token)
}

// AcceptString processes a token string directly.
// Returns true if the token was valid and accepted.
func (e *Engine) AcceptString(token string) bool {
	return e.runtime.AcceptToken(token)
}

// IsComplete returns true if the current state is accepting.
func (e *Engine) IsComplete() bool {
	return e.runtime.IsAccepting()
}

// Reset resets the engine to initial state.
func (e *Engine) Reset() {
	e.runtime.Reset()
}

// ValidTokens returns the indices of tokens that are currently valid.
// This is useful for debugging and testing.
func (e *Engine) ValidTokens() []int {
	validSymbols := e.runtime.ValidInputs()
	seen := make(map[int]bool)
	var result []int

	for _, sym := range validSymbols {
		for i, token := range e.vocab {
			if e.tokenMatchesSymbol(token, sym) && !seen[i] {
				seen[i] = true
				result = append(result, i)
			}
		}
	}

	return result
}

// State returns the current PDA state (for debugging).
func (e *Engine) State() State {
	return e.runtime.State()
}

// Close releases MLX resources.
func (e *Engine) Close() {
	for _, mask := range e.symbolMasks {
		mask.Free()
	}
	if e.negInfMask != nil {
		e.negInfMask.Free()
	}
	if e.threshold != nil {
		e.threshold.Free()
	}
}
