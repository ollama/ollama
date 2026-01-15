//go:build mlx

package glm_image

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/ollama/ollama/x/imagegen"
)

// GLMTokenizer implements the GLM tokenizer for the AR model
// This is a BPE-style tokenizer with ignore_merges=true, meaning it does
// greedy longest-match tokenization from the vocab without runtime merging.
type GLMTokenizer struct {
	Vocab        map[string]int32 // token string -> token ID
	VocabReverse map[int32]string // token ID -> token string
	SpecialTokens map[string]int32 // special token strings -> IDs

	// Special token IDs
	SopTokenID  int32 // <sop> = grid_bos_token (167845)
	EopTokenID  int32 // <eop> = grid_eos_token (167846)
	BosTokenID  int32 // <|dit_token_16384|> = visual BOS (16384)
	EosTokenID  int32 // <|dit_token_16385|> = visual EOS (16385)
	PadTokenID  int32

	// Sorted vocab keys by length (longest first) for greedy matching
	sortedTokens []string
}

// tokenizerJSON represents the structure of tokenizer.json
type tokenizerJSON struct {
	Model struct {
		Vocab map[string]int32 `json:"vocab"`
	} `json:"model"`
	AddedTokens []struct {
		ID      int32  `json:"id"`
		Content string `json:"content"`
		Special bool   `json:"special"`
	} `json:"added_tokens"`
}

// NewGLMTokenizer creates a GLM tokenizer from the model manifest
func NewGLMTokenizer(manifest *imagegen.ModelManifest) (*GLMTokenizer, error) {
	// Read tokenizer.json from processor directory in manifest
	data, err := manifest.ReadConfig("processor/tokenizer.json")
	if err != nil {
		return nil, fmt.Errorf("failed to read tokenizer.json from manifest: %w", err)
	}

	var tj tokenizerJSON
	if err := json.Unmarshal(data, &tj); err != nil {
		return nil, fmt.Errorf("failed to parse tokenizer.json: %w", err)
	}

	tok := &GLMTokenizer{
		Vocab:         make(map[string]int32),
		VocabReverse:  make(map[int32]string),
		SpecialTokens: make(map[string]int32),
	}

	// Load vocab from model section
	for token, id := range tj.Model.Vocab {
		tok.Vocab[token] = id
		tok.VocabReverse[id] = token
	}

	// Load added tokens (special tokens including dit_tokens)
	for _, at := range tj.AddedTokens {
		tok.Vocab[at.Content] = at.ID
		tok.VocabReverse[at.ID] = at.Content
		if at.Special {
			tok.SpecialTokens[at.Content] = at.ID
		}
	}

	// Set special token IDs
	tok.SopTokenID = 167845   // <sop>
	tok.EopTokenID = 167846   // <eop>
	tok.BosTokenID = 16384    // <|dit_token_16384|>
	tok.EosTokenID = 16385    // <|dit_token_16385|>
	tok.PadTokenID = 16385    // Same as EOS

	// Build sorted token list for greedy matching (longest first)
	tok.sortedTokens = make([]string, 0, len(tok.Vocab))
	for token := range tok.Vocab {
		tok.sortedTokens = append(tok.sortedTokens, token)
	}
	sort.Slice(tok.sortedTokens, func(i, j int) bool {
		return len(tok.sortedTokens[i]) > len(tok.sortedTokens[j])
	})

	fmt.Printf("Loaded GLM tokenizer with %d tokens\n", len(tok.Vocab))

	return tok, nil
}

// NewGLMTokenizerFromPath creates a GLM tokenizer from a directory path
func NewGLMTokenizerFromPath(modelPath string) (*GLMTokenizer, error) {
	// Read tokenizer.json from processor directory
	tokenizerPath := filepath.Join(modelPath, "processor", "tokenizer.json")
	data, err := os.ReadFile(tokenizerPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read tokenizer.json: %w", err)
	}

	var tj tokenizerJSON
	if err := json.Unmarshal(data, &tj); err != nil {
		return nil, fmt.Errorf("failed to parse tokenizer.json: %w", err)
	}

	tok := &GLMTokenizer{
		Vocab:         make(map[string]int32),
		VocabReverse:  make(map[int32]string),
		SpecialTokens: make(map[string]int32),
	}

	// Load vocab from model section
	for token, id := range tj.Model.Vocab {
		tok.Vocab[token] = id
		tok.VocabReverse[id] = token
	}

	// Load added tokens (special tokens including dit_tokens)
	for _, at := range tj.AddedTokens {
		tok.Vocab[at.Content] = at.ID
		tok.VocabReverse[at.ID] = at.Content
		if at.Special {
			tok.SpecialTokens[at.Content] = at.ID
		}
	}

	// Set special token IDs
	tok.SopTokenID = 167845 // <sop>
	tok.EopTokenID = 167846 // <eop>
	tok.BosTokenID = 16384  // <|dit_token_16384|>
	tok.EosTokenID = 16385  // <|dit_token_16385|>
	tok.PadTokenID = 16385  // Same as EOS

	// Build sorted token list for greedy matching (longest first)
	tok.sortedTokens = make([]string, 0, len(tok.Vocab))
	for token := range tok.Vocab {
		tok.sortedTokens = append(tok.sortedTokens, token)
	}
	sort.Slice(tok.sortedTokens, func(i, j int) bool {
		return len(tok.sortedTokens[i]) > len(tok.sortedTokens[j])
	})

	fmt.Printf("Loaded GLM tokenizer with %d tokens\n", len(tok.Vocab))

	return tok, nil
}

// Encode tokenizes a string into token IDs
// This uses greedy longest-match tokenization with GPT-2 style space handling
func (t *GLMTokenizer) Encode(text string) []int32 {
	if text == "" {
		return []int32{}
	}

	var tokens []int32

	// First, check for and handle special tokens
	// Replace special tokens with placeholders, encode, then restore
	specialReplacements := make(map[string]int32)
	for special, id := range t.SpecialTokens {
		if strings.Contains(text, special) {
			specialReplacements[special] = id
		}
	}

	// Process text character by character with special token handling
	i := 0
	isFirstToken := true

	for i < len(text) {
		// Check for special tokens first
		foundSpecial := false
		for special, id := range specialReplacements {
			if strings.HasPrefix(text[i:], special) {
				tokens = append(tokens, id)
				i += len(special)
				isFirstToken = false
				foundSpecial = true
				break
			}
		}
		if foundSpecial {
			continue
		}

		// Handle regular text with GPT-2 style space prefix
		// "Ġ" (U+0120) represents a space before a token
		remaining := text[i:]

		// Try to find the longest matching token
		matched := false
		for _, token := range t.sortedTokens {
			// Skip special tokens in regular matching
			if _, isSpecial := t.SpecialTokens[token]; isSpecial {
				continue
			}

			// Check if this token matches
			tokenText := token

			// Handle the Ġ prefix (represents space)
			if strings.HasPrefix(token, "Ġ") {
				// This token expects a leading space
				if i > 0 || !isFirstToken {
					// Check if remaining starts with space + token content
					tokenContent := token[len("Ġ"):]
					if strings.HasPrefix(remaining, " "+tokenContent) {
						tokens = append(tokens, t.Vocab[token])
						i += 1 + len(tokenContent) // space + content
						isFirstToken = false
						matched = true
						break
					}
				}
			} else {
				// Regular token without space prefix
				if strings.HasPrefix(remaining, tokenText) {
					tokens = append(tokens, t.Vocab[token])
					i += len(tokenText)
					isFirstToken = false
					matched = true
					break
				}
			}
		}

		if !matched {
			// No token found - skip this character (or use UNK)
			// For now, just skip unknown characters
			i++
		}
	}

	return tokens
}

// EncodeForGeneration encodes a prompt with grid tokens for image generation
// Format: {prompt}<sop>{token_h} {token_w}<eop><sop>{prev_h} {prev_w}<eop><|dit_token_16384|>
//
// Uses GPT-2 style tokenization where " 32" becomes "Ġ32" (a single token with
// space prefix), matching the HuggingFace tokenizer behavior.
func (t *GLMTokenizer) EncodeForGeneration(prompt string, targetHeight, targetWidth int32) []int32 {
	// Calculate grid dimensions
	factor := int32(32)
	height := (targetHeight / factor) * factor
	width := (targetWidth / factor) * factor
	tokenH := height / factor
	tokenW := width / factor

	// Calculate previous grid dimensions
	ratio := float64(tokenH) / float64(tokenW)
	prevTokenH := int32(sqrt(ratio) * 16)
	prevTokenW := int32(sqrt(1.0/ratio) * 16)

	// Encode the prompt text
	promptTokens := t.Encode(prompt)

	// Build the full sequence:
	// [prompt tokens] <sop> [tokenH] [Ġ+tokenW] <eop> <sop> [prevH] [Ġ+prevW] <eop> <bos>
	// Note: HF tokenizer treats " 32" as "Ġ32" (single token), not "Ġ" + "32"
	var tokens []int32
	tokens = append(tokens, promptTokens...)

	// First grid: <sop> H W <eop>
	// First number has no space prefix, second number has space prefix (Ġ)
	tokens = append(tokens, t.SopTokenID)
	tokens = append(tokens, t.encodeNumber(tokenH)...)
	tokens = append(tokens, t.encodeSpaceNumber(tokenW)...) // " W" as Ġ+W
	tokens = append(tokens, t.EopTokenID)

	// Second grid: <sop> prevH prevW <eop>
	tokens = append(tokens, t.SopTokenID)
	tokens = append(tokens, t.encodeNumber(prevTokenH)...)
	tokens = append(tokens, t.encodeSpaceNumber(prevTokenW)...) // " prevW" as Ġ+prevW
	tokens = append(tokens, t.EopTokenID)

	// BOS token (start of image generation)
	tokens = append(tokens, t.BosTokenID)

	return tokens
}

// encodeNumber encodes a number - first tries as a whole token, falls back to digit-by-digit
func (t *GLMTokenizer) encodeNumber(n int32) []int32 {
	s := fmt.Sprintf("%d", n)
	// First try: look up the whole number as a single token
	if id, ok := t.Vocab[s]; ok {
		return []int32{id}
	}
	// Fallback: encode digit by digit
	var tokens []int32
	for _, c := range s {
		if id, ok := t.Vocab[string(c)]; ok {
			tokens = append(tokens, id)
		}
	}
	return tokens
}

// encodeSpaceNumber encodes " N" as "ĠN" (space-prefixed number) matching HF tokenizer
// GPT-2 style: " 32" becomes single token "Ġ32", not "Ġ" + "32"
func (t *GLMTokenizer) encodeSpaceNumber(n int32) []int32 {
	s := fmt.Sprintf("%d", n)

	// First try: look up "Ġ{number}" as a single token (e.g., "Ġ32")
	spaceToken := "Ġ" + s
	if id, ok := t.Vocab[spaceToken]; ok {
		return []int32{id}
	}

	// Fallback: bare space Ġ + number tokens
	var tokens []int32
	if spaceID, ok := t.Vocab["Ġ"]; ok {
		tokens = append(tokens, spaceID)
	}
	tokens = append(tokens, t.encodeNumber(n)...)
	return tokens
}

// sqrt is a helper for float64 sqrt
func sqrt(x float64) float64 {
	if x <= 0 {
		return 0
	}
	// Newton's method
	z := x
	for i := 0; i < 10; i++ {
		z = z - (z*z-x)/(2*z)
	}
	return z
}

// Decode converts token IDs back to a string
func (t *GLMTokenizer) Decode(tokens []int32) string {
	var sb strings.Builder
	for _, id := range tokens {
		if token, ok := t.VocabReverse[id]; ok {
			// Handle Ġ prefix (convert back to space)
			if strings.HasPrefix(token, "Ġ") {
				sb.WriteString(" ")
				sb.WriteString(token[len("Ġ"):])
			} else {
				sb.WriteString(token)
			}
		}
	}
	return sb.String()
}
