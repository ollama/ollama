//go:build mlx

// tokenizer.go - BPE and SentencePiece tokenizer for HuggingFace models
//
// Based on standard BPE algorithm (Sennrich et al. 2015) with:
// - GPT-2 byte-level encoding (OpenAI tiktoken)
// - HuggingFace tokenizer.json pretokenizer patterns
// - SentencePiece ▁-style space handling

package tokenizer

import (
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"unicode"
	"unicode/utf8"
)

// TokenizerType identifies the tokenization algorithm
type TokenizerType int

const (
	TokenizerBPE           TokenizerType = iota // GPT-2 style byte-level BPE
	TokenizerSentencePiece                      // SentencePiece with ▁ for spaces
	TokenizerWordPiece                          // BERT style with ## continuations
)

// Vocabulary holds the tokenizer vocabulary and merges
type Vocabulary struct {
	Values  []string
	Reverse map[string]int32
	Merges  map[string]int

	BOS    int32
	EOS    []int32 // Multiple EOS tokens supported (e.g., Gemma has <eos> and <end_of_turn>)
	PAD    int32   // Padding token (often <|endoftext|> or <pad>)
	AddBOS bool
	AddEOS bool

	// Precomputed byte token IDs for <0xNN> fallback (256 entries, -1 if not found)
	byteTokens [256]int32
}

// Tokenizer handles BPE, SentencePiece, and WordPiece tokenization
type Tokenizer struct {
	vocab         *Vocabulary
	pretokenizer  *regexp.Regexp
	specialTokens map[string]int32 // Special tokens for direct lookup
	typ           TokenizerType    // Algorithm type
	unkToken      int32            // [UNK] token ID for WordPiece fallback
}

// Precomputed GPT-2 byte-level encoding table
// Maps byte values to their encoded rune equivalents
var byteToRune [256]rune

func init() {
	for b := 0; b < 256; b++ {
		r := rune(b)
		switch {
		case r == 0x00ad:
			r = 0x0143
		case r <= 0x0020:
			r = r + 0x0100
		case r >= 0x007f && r <= 0x00a0:
			r = r + 0x00a2
		}
		byteToRune[b] = r
	}
}

// loadSpecialTokenConfig loads special token configuration from HuggingFace companion files.
//
// Loading priority for EOS tokens (can be single int or []int):
//  1. generation_config.json - eos_token_id (preferred, matches HuggingFace generation)
//  2. config.json - eos_token_id (model config fallback)
//  3. tokenizer_config.json - eos_token string + add_bos/add_eos flags
//  4. special_tokens_map.json - final fallback
func loadSpecialTokenConfig(dir string, t *Tokenizer) {
	// Helper to parse eos_token_id which can be int or []int
	parseTokenIDs := func(v interface{}) []int32 {
		switch val := v.(type) {
		case float64:
			return []int32{int32(val)}
		case []interface{}:
			ids := make([]int32, 0, len(val))
			for _, id := range val {
				if f, ok := id.(float64); ok {
					ids = append(ids, int32(f))
				}
			}
			return ids
		}
		return nil
	}

	// Priority 1: generation_config.json (eos_token_id can be int or []int)
	if data, err := os.ReadFile(dir + "generation_config.json"); err == nil {
		var config struct {
			EOSTokenID interface{} `json:"eos_token_id"`
			BOSTokenID interface{} `json:"bos_token_id"`
		}
		if err := json.Unmarshal(data, &config); err == nil {
			if ids := parseTokenIDs(config.EOSTokenID); len(ids) > 0 {
				t.vocab.EOS = ids
			}
			if ids := parseTokenIDs(config.BOSTokenID); len(ids) > 0 {
				t.vocab.BOS = ids[0]
			}
		}
	}

	// Priority 2: config.json (model config, same format)
	if len(t.vocab.EOS) == 0 || t.vocab.BOS < 0 {
		if data, err := os.ReadFile(dir + "config.json"); err == nil {
			var config struct {
				EOSTokenID interface{} `json:"eos_token_id"`
				BOSTokenID interface{} `json:"bos_token_id"`
			}
			if err := json.Unmarshal(data, &config); err == nil {
				if len(t.vocab.EOS) == 0 {
					if ids := parseTokenIDs(config.EOSTokenID); len(ids) > 0 {
						t.vocab.EOS = ids
					}
				}
				if t.vocab.BOS < 0 {
					if ids := parseTokenIDs(config.BOSTokenID); len(ids) > 0 {
						t.vocab.BOS = ids[0]
					}
				}
			}
		}
	}

	// Priority 3: tokenizer_config.json (token strings + add_bos/add_eos flags)
	if data, err := os.ReadFile(dir + "tokenizer_config.json"); err == nil {
		var config struct {
			BOSToken    interface{} `json:"bos_token"`
			EOSToken    interface{} `json:"eos_token"`
			PADToken    interface{} `json:"pad_token"`
			AddBOSToken *bool       `json:"add_bos_token"`
			AddEOSToken *bool       `json:"add_eos_token"`
		}
		if err := json.Unmarshal(data, &config); err == nil {
			if t.vocab.BOS < 0 {
				if bosStr := extractTokenString(config.BOSToken); bosStr != "" {
					if id, ok := t.specialTokens[bosStr]; ok {
						t.vocab.BOS = id
					}
				}
			}
			if len(t.vocab.EOS) == 0 {
				if eosStr := extractTokenString(config.EOSToken); eosStr != "" {
					if id, ok := t.specialTokens[eosStr]; ok {
						t.vocab.EOS = []int32{id}
					}
				}
			}
			if t.vocab.PAD < 0 {
				if padStr := extractTokenString(config.PADToken); padStr != "" {
					if id, ok := t.specialTokens[padStr]; ok {
						t.vocab.PAD = id
					}
				}
			}
			if config.AddBOSToken != nil {
				t.vocab.AddBOS = *config.AddBOSToken
			}
			if config.AddEOSToken != nil {
				t.vocab.AddEOS = *config.AddEOSToken
			}
		}
	}

	// Priority 4: special_tokens_map.json (final fallback)
	if data, err := os.ReadFile(dir + "special_tokens_map.json"); err == nil {
		var tokensMap map[string]interface{}
		if err := json.Unmarshal(data, &tokensMap); err == nil {
			if t.vocab.BOS < 0 {
				if bosStr := extractTokenString(tokensMap["bos_token"]); bosStr != "" {
					if id, ok := t.specialTokens[bosStr]; ok {
						t.vocab.BOS = id
					}
				}
			}
			if len(t.vocab.EOS) == 0 {
				if eosStr := extractTokenString(tokensMap["eos_token"]); eosStr != "" {
					if id, ok := t.specialTokens[eosStr]; ok {
						t.vocab.EOS = []int32{id}
					}
				}
			}
			if t.vocab.PAD < 0 {
				if padStr := extractTokenString(tokensMap["pad_token"]); padStr != "" {
					if id, ok := t.specialTokens[padStr]; ok {
						t.vocab.PAD = id
					}
				}
			}
		}
	}
}

// extractTokenString extracts the token string from various formats used in HuggingFace configs.
// Tokens can be represented as:
//   - string: "token"
//   - object: {"content": "token", ...}
func extractTokenString(v interface{}) string {
	if v == nil {
		return ""
	}
	// Direct string
	if s, ok := v.(string); ok {
		return s
	}
	// Object with content field
	if m, ok := v.(map[string]interface{}); ok {
		if content, ok := m["content"].(string); ok {
			return content
		}
	}
	return ""
}

// rewritePatternForRE2 rewrites HuggingFace pretokenizer regex patterns to be
// compatible with Go's regexp package (RE2). HuggingFace patterns use PCRE features:
//   - (?!\S) negative lookahead - RE2 doesn't support this
//   - (?i:...) inline case-insensitive groups - RE2 doesn't support this
//
// We replace \s+(?!\S)|\s+ with \s+ and fix whitespace boundaries in encodeWithRegex().
// The lookahead version splits "a  b" into ["a", " ", " b"] (space prepended to word).
// Simple \s+ would give ["a", "  ", "b"]. We post-process to match Python's behavior.
func rewritePatternForRE2(pattern string) string {
	// Replace lookahead pattern with simple \s+ - we fix boundaries in encodeWithRegex()
	pattern = strings.ReplaceAll(pattern, `\s+(?!\S)|\s+`, `\s+`)

	// Handle the pattern when it appears with a ? suffix (optional contractions in GPT-4o style)
	// IMPORTANT: Must be done before the non-optional version to avoid partial replacement
	pattern = strings.ReplaceAll(pattern,
		`(?i:'s|'t|'re|'ve|'m|'ll|'d)?`,
		`(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?`)

	// Expand case-insensitive contraction pattern to explicit alternations
	// (?i:'s|'t|'re|'ve|'m|'ll|'d) -> '[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD]
	pattern = strings.ReplaceAll(pattern,
		`(?i:'s|'t|'re|'ve|'m|'ll|'d)`,
		`(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])`)

	return pattern
}

// LoadFromBytes loads a tokenizer from tokenizer.json bytes.
// This is useful when loading from blob storage where the file content is already in memory.
// Note: This won't load special token config from companion files. Use LoadFromBytesWithConfig
// to provide tokenizer_config.json data for proper PAD/EOS token loading.
func LoadFromBytes(data []byte) (*Tokenizer, error) {
	return loadFromTokenizerJSON(data, "")
}

// TokenizerConfig holds optional configuration data that can be passed to LoadFromBytesWithConfig.
type TokenizerConfig struct {
	TokenizerConfigJSON   []byte // tokenizer_config.json content
	GenerationConfigJSON  []byte // generation_config.json content
	SpecialTokensMapJSON  []byte // special_tokens_map.json content
	ConfigJSON            []byte // config.json content
}

// LoadFromBytesWithConfig loads a tokenizer from tokenizer.json bytes with additional config files.
// This is useful when loading from blob storage where companion config files are also blobs.
func LoadFromBytesWithConfig(data []byte, config *TokenizerConfig) (*Tokenizer, error) {
	t, err := loadFromTokenizerJSON(data, "")
	if err != nil {
		return nil, err
	}

	if config == nil {
		return t, nil
	}

	// Apply special token configs from provided data
	loadSpecialTokenConfigFromBytes(t, config)

	return t, nil
}

// loadSpecialTokenConfigFromBytes loads special token configuration from byte slices.
func loadSpecialTokenConfigFromBytes(t *Tokenizer, config *TokenizerConfig) {
	// Helper to parse eos_token_id which can be int or []int
	parseTokenIDs := func(v interface{}) []int32 {
		switch val := v.(type) {
		case float64:
			return []int32{int32(val)}
		case []interface{}:
			ids := make([]int32, 0, len(val))
			for _, id := range val {
				if f, ok := id.(float64); ok {
					ids = append(ids, int32(f))
				}
			}
			return ids
		}
		return nil
	}

	// Priority 1: generation_config.json
	if len(config.GenerationConfigJSON) > 0 {
		var genConfig struct {
			EOSTokenID interface{} `json:"eos_token_id"`
			BOSTokenID interface{} `json:"bos_token_id"`
		}
		if err := json.Unmarshal(config.GenerationConfigJSON, &genConfig); err == nil {
			if ids := parseTokenIDs(genConfig.EOSTokenID); len(ids) > 0 {
				t.vocab.EOS = ids
			}
			if ids := parseTokenIDs(genConfig.BOSTokenID); len(ids) > 0 {
				t.vocab.BOS = ids[0]
			}
		}
	}

	// Priority 2: config.json
	if len(config.ConfigJSON) > 0 && (len(t.vocab.EOS) == 0 || t.vocab.BOS < 0) {
		var modelConfig struct {
			EOSTokenID interface{} `json:"eos_token_id"`
			BOSTokenID interface{} `json:"bos_token_id"`
		}
		if err := json.Unmarshal(config.ConfigJSON, &modelConfig); err == nil {
			if len(t.vocab.EOS) == 0 {
				if ids := parseTokenIDs(modelConfig.EOSTokenID); len(ids) > 0 {
					t.vocab.EOS = ids
				}
			}
			if t.vocab.BOS < 0 {
				if ids := parseTokenIDs(modelConfig.BOSTokenID); len(ids) > 0 {
					t.vocab.BOS = ids[0]
				}
			}
		}
	}

	// Priority 3: tokenizer_config.json
	if len(config.TokenizerConfigJSON) > 0 {
		var tokConfig struct {
			BOSToken    interface{} `json:"bos_token"`
			EOSToken    interface{} `json:"eos_token"`
			PADToken    interface{} `json:"pad_token"`
			AddBOSToken *bool       `json:"add_bos_token"`
			AddEOSToken *bool       `json:"add_eos_token"`
		}
		if err := json.Unmarshal(config.TokenizerConfigJSON, &tokConfig); err == nil {
			if t.vocab.BOS < 0 {
				if bosStr := extractTokenString(tokConfig.BOSToken); bosStr != "" {
					if id, ok := t.specialTokens[bosStr]; ok {
						t.vocab.BOS = id
					}
				}
			}
			if len(t.vocab.EOS) == 0 {
				if eosStr := extractTokenString(tokConfig.EOSToken); eosStr != "" {
					if id, ok := t.specialTokens[eosStr]; ok {
						t.vocab.EOS = []int32{id}
					}
				}
			}
			if t.vocab.PAD < 0 {
				if padStr := extractTokenString(tokConfig.PADToken); padStr != "" {
					if id, ok := t.specialTokens[padStr]; ok {
						t.vocab.PAD = id
					}
				}
			}
			if tokConfig.AddBOSToken != nil {
				t.vocab.AddBOS = *tokConfig.AddBOSToken
			}
			if tokConfig.AddEOSToken != nil {
				t.vocab.AddEOS = *tokConfig.AddEOSToken
			}
		}
	}

	// Priority 4: special_tokens_map.json
	if len(config.SpecialTokensMapJSON) > 0 {
		var tokensMap map[string]interface{}
		if err := json.Unmarshal(config.SpecialTokensMapJSON, &tokensMap); err == nil {
			if t.vocab.BOS < 0 {
				if bosStr := extractTokenString(tokensMap["bos_token"]); bosStr != "" {
					if id, ok := t.specialTokens[bosStr]; ok {
						t.vocab.BOS = id
					}
				}
			}
			if len(t.vocab.EOS) == 0 {
				if eosStr := extractTokenString(tokensMap["eos_token"]); eosStr != "" {
					if id, ok := t.specialTokens[eosStr]; ok {
						t.vocab.EOS = []int32{id}
					}
				}
			}
			if t.vocab.PAD < 0 {
				if padStr := extractTokenString(tokensMap["pad_token"]); padStr != "" {
					if id, ok := t.specialTokens[padStr]; ok {
						t.vocab.PAD = id
					}
				}
			}
		}
	}
}

// Load loads a tokenizer from a path which can be:
// - A tokenizer.json file
// - A directory containing tokenizer.json or vocab.json + merges.txt
func Load(path string) (*Tokenizer, error) {
	// Check if path is a directory
	if info, err := os.Stat(path); err == nil && info.IsDir() {
		dir := strings.TrimSuffix(path, "/") + "/"
		// Try tokenizer.json first
		if data, err := os.ReadFile(dir + "tokenizer.json"); err == nil {
			return loadFromTokenizerJSON(data, dir)
		}
		// Fall back to vocab.json + merges.txt
		return LoadVocabMerges(path)
	}

	// It's a file - read it directly
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read tokenizer: %w", err)
	}

	// Get directory for loading companion files
	dir := ""
	if idx := strings.LastIndex(path, "/"); idx >= 0 {
		dir = path[:idx+1]
	}
	return loadFromTokenizerJSON(data, dir)
}

// loadFromTokenizerJSON parses a tokenizer.json file
func loadFromTokenizerJSON(data []byte, dir string) (*Tokenizer, error) {

	var raw struct {
		Model struct {
			Type   string           `json:"type"` // "BPE" or "WordPiece"
			Vocab  map[string]int32 `json:"vocab"`
			Merges json.RawMessage  `json:"merges"` // Can be []string or [][]string (BPE only)
		} `json:"model"`
		PreTokenizer json.RawMessage `json:"pre_tokenizer"`
		Decoder      json.RawMessage `json:"decoder"`
		AddedTokens  []struct {
			ID      int32  `json:"id"`
			Content string `json:"content"`
			Special bool   `json:"special"`
		} `json:"added_tokens"`
	}

	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("failed to parse tokenizer: %w", err)
	}

	// Parse merges - can be []string (Llama) or [][]string (GPT-OSS)
	// WordPiece models don't have merges
	var mergesStrings []string
	if raw.Model.Type != "WordPiece" && raw.Model.Merges != nil {
		var mergesArrays [][]string
		if err := json.Unmarshal(raw.Model.Merges, &mergesStrings); err != nil {
			// Try array of arrays format
			if err := json.Unmarshal(raw.Model.Merges, &mergesArrays); err != nil {
				return nil, fmt.Errorf("failed to parse merges: %w", err)
			}
			// Convert [][]string to []string
			mergesStrings = make([]string, len(mergesArrays))
			for i, pair := range mergesArrays {
				mergesStrings[i] = pair[0] + " " + pair[1]
			}
		}
	}

	// Build tokenizer
	t := &Tokenizer{
		vocab: &Vocabulary{
			Values:  make([]string, len(raw.Model.Vocab)),
			Reverse: raw.Model.Vocab,
			Merges:  make(map[string]int, len(mergesStrings)),
			BOS:     -1,
			PAD:     -1,
		},
		specialTokens: make(map[string]int32),
	}

	// Build values array
	for token, id := range raw.Model.Vocab {
		if int(id) >= len(t.vocab.Values) {
			newValues := make([]string, id+1)
			copy(newValues, t.vocab.Values)
			t.vocab.Values = newValues
		}
		t.vocab.Values[id] = token
	}

	// Build merges map
	for i, merge := range mergesStrings {
		t.vocab.Merges[merge] = i
	}

	// Add all added_tokens to vocabulary and special tokens map.
	// HuggingFace treats ALL added_tokens as special for tokenization purposes -
	// they bypass BPE and get their own token ID. The "special" flag just indicates
	// if it's a "truly special" token like BOS/EOS/PAD, but for tokenization we need
	// to treat all added_tokens as special to match HuggingFace behavior.
	for _, tok := range raw.AddedTokens {
		if int(tok.ID) >= len(t.vocab.Values) {
			newValues := make([]string, tok.ID+1)
			copy(newValues, t.vocab.Values)
			t.vocab.Values = newValues
		}
		t.vocab.Values[tok.ID] = tok.Content
		t.specialTokens[tok.Content] = tok.ID // Add ALL added_tokens to special tokens
	}

	// Load special token configuration from companion files
	loadSpecialTokenConfig(dir, t)

	// Precompute byte token IDs for <0xNN> fallback
	initByteTokens(t)

	// Determine tokenizer type
	switch {
	case raw.Model.Type == "WordPiece":
		t.typ = TokenizerWordPiece
	case detectSentencePiece(raw.Decoder):
		t.typ = TokenizerSentencePiece
	default:
		t.typ = TokenizerBPE
	}

	// Parse and compile pretokenizer pattern (BPE only - SentencePiece doesn't use pretokenizer)
	if t.typ == TokenizerBPE {
		pattern := extractPretokenizer(raw.PreTokenizer)
		if pattern == "" {
			pattern = `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
		}
		re, err := regexp.Compile(rewritePatternForRE2(pattern))
		if err != nil {
			return nil, fmt.Errorf("failed to compile pretokenizer regex %q: %w", pattern, err)
		}
		t.pretokenizer = re
	}

	return t, nil
}

// detectSentencePiece checks if the decoder uses SentencePiece-style (▁ for spaces)
// vs GPT-2 byte-level encoding
func detectSentencePiece(data json.RawMessage) bool {
	if data == nil {
		return false
	}

	// Check for Sequence decoder with Replace step (SentencePiece style)
	var seq struct {
		Type     string `json:"type"`
		Decoders []struct {
			Type    string `json:"type"`
			Pattern struct {
				String string `json:"String"`
			} `json:"pattern"`
		} `json:"decoders"`
	}
	if err := json.Unmarshal(data, &seq); err == nil {
		if seq.Type == "Sequence" {
			for _, dec := range seq.Decoders {
				// Look for Replace decoder that converts ▁ to space
				if dec.Type == "Replace" && dec.Pattern.String == "▁" {
					return true
				}
			}
		}
	}

	// Check for direct ByteLevel decoder (GPT-2 style)
	var simple struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(data, &simple); err == nil {
		if simple.Type == "ByteLevel" {
			return false
		}
	}

	return false
}

// initByteTokens precomputes byte token IDs for <0xNN> fallback encoding
func initByteTokens(t *Tokenizer) {
	for i := range t.vocab.byteTokens {
		t.vocab.byteTokens[i] = -1
	}
	for b := 0; b < 256; b++ {
		token := fmt.Sprintf("<0x%02X>", b)
		if id, ok := t.vocab.Reverse[token]; ok {
			t.vocab.byteTokens[b] = id
		}
	}
}

// extractPretokenizer extracts the regex pattern from the pre_tokenizer config
func extractPretokenizer(data json.RawMessage) string {
	if data == nil {
		return ""
	}

	// Try to parse as a single Split pretokenizer
	var single struct {
		Type    string `json:"type"`
		Pattern struct {
			Regex string `json:"Regex"`
		} `json:"pattern"`
	}
	if err := json.Unmarshal(data, &single); err == nil && single.Pattern.Regex != "" {
		return single.Pattern.Regex
	}

	// Try to parse as Sequence of pretokenizers - use first Split pattern
	var seq struct {
		Type          string `json:"type"`
		Pretokenizers []struct {
			Type    string `json:"type"`
			Pattern struct {
				Regex string `json:"Regex"`
			} `json:"pattern"`
		} `json:"pretokenizers"`
	}
	if err := json.Unmarshal(data, &seq); err == nil && seq.Type == "Sequence" {
		for _, pt := range seq.Pretokenizers {
			if pt.Type == "Split" && pt.Pattern.Regex != "" {
				return pt.Pattern.Regex
			}
		}
	}

	return ""
}

// isNonNewlineWhitespace returns true if s contains only whitespace characters (no newlines)
func isNonNewlineWhitespace(s string) bool {
	if s == "" {
		return false
	}
	for _, r := range s {
		if r == '\n' || r == '\r' {
			return false
		}
		if !unicode.IsSpace(r) {
			return false
		}
	}
	return true
}

// splitBySpecialTokens splits text into parts, keeping special tokens as separate elements
func (t *Tokenizer) splitBySpecialTokens(s string) []string {
	if len(t.specialTokens) == 0 {
		return []string{s}
	}

	// Sort special tokens by length (longest first) to match greedily
	tokens := make([]string, 0, len(t.specialTokens))
	for tok := range t.specialTokens {
		tokens = append(tokens, tok)
	}
	sort.Slice(tokens, func(i, j int) bool {
		return len(tokens[i]) > len(tokens[j])
	})

	var result []string
	remaining := s

	for len(remaining) > 0 {
		found := false
		for _, tok := range tokens {
			if strings.HasPrefix(remaining, tok) {
				result = append(result, tok)
				remaining = remaining[len(tok):]
				found = true
				break
			}
		}
		if !found {
			// Find next special token position
			nextPos := len(remaining)
			for _, tok := range tokens {
				if idx := strings.Index(remaining, tok); idx != -1 && idx < nextPos {
					nextPos = idx
				}
			}
			if nextPos > 0 {
				result = append(result, remaining[:nextPos])
			}
			remaining = remaining[nextPos:]
		}
	}

	return result
}

// Encode tokenizes text to token IDs. Parallelizes for large inputs (>10KB).
func (t *Tokenizer) Encode(s string, addBOS bool) []int32 {
	// First: split by special tokens
	parts := t.splitBySpecialTokens(s)

	// Second: collect all pretokenizer chunks
	type chunk struct {
		text      string
		isSpecial bool
	}
	var allChunks []chunk

	if t.pretokenizer != nil {
		re := t.pretokenizer
		for _, part := range parts {
			if _, ok := t.specialTokens[part]; ok {
				allChunks = append(allChunks, chunk{part, true})
				continue
			}

			// Split by pretokenizer regex
			type match struct{ start, end int }
			var matches []match
			offset := 0
			for offset < len(part) {
				loc := re.FindStringIndex(part[offset:])
				if loc == nil {
					break
				}
				matches = append(matches, match{offset + loc[0], offset + loc[1]})
				offset += loc[1]
			}

			// Apply whitespace boundary fix for Python regex compatibility
			for i := 0; i < len(matches)-1; i++ {
				m := part[matches[i].start:matches[i].end]
				next := part[matches[i+1].start:matches[i+1].end]

				if isNonNewlineWhitespace(m) && len(next) > 0 {
					firstRune, _ := utf8.DecodeRuneInString(next)
					if unicode.IsLetter(firstRune) {
						lastSpaceStart := matches[i].end
						for j := matches[i].end; j > matches[i].start; {
							r, size := utf8.DecodeLastRuneInString(part[matches[i].start:j])
							if unicode.IsSpace(r) {
								lastSpaceStart = j - size
								break
							}
							j -= size
						}
						if lastSpaceStart > matches[i].start {
							matches[i].end = lastSpaceStart
							matches[i+1].start = lastSpaceStart
						} else {
							matches[i+1].start = matches[i].start
							matches[i].end = matches[i].start
						}
					}
				}
			}

			for _, m := range matches {
				if m.end > m.start {
					allChunks = append(allChunks, chunk{part[m.start:m.end], false})
				}
			}
		}
	} else {
		// No pretokenizer - treat each part as a single chunk
		for _, part := range parts {
			if _, ok := t.specialTokens[part]; ok {
				allChunks = append(allChunks, chunk{part, true})
			} else {
				allChunks = append(allChunks, chunk{part, false})
			}
		}
	}

	// Encode chunks - parallel for large inputs (>4KB), sequential otherwise
	var ids []int32
	if len(s) < 4096 {
		for _, c := range allChunks {
			if c.isSpecial {
				if id, ok := t.specialTokens[c.text]; ok {
					ids = append(ids, id)
				}
			} else {
				ids = t.encodeChunkInto(c.text, ids)
			}
		}
	} else {
		numWorkers := runtime.GOMAXPROCS(0)
		if numWorkers > len(allChunks) {
			numWorkers = len(allChunks)
		}

		chunksPer := (len(allChunks) + numWorkers - 1) / numWorkers
		results := make([][]int32, numWorkers)
		var wg sync.WaitGroup

		for i := 0; i < numWorkers; i++ {
			start := i * chunksPer
			end := start + chunksPer
			if end > len(allChunks) {
				end = len(allChunks)
			}
			if start >= end {
				continue
			}

			wg.Add(1)
			go func(i int, chunks []chunk) {
				defer wg.Done()
				var r []int32
				for _, c := range chunks {
					if c.isSpecial {
						if id, ok := t.specialTokens[c.text]; ok {
							r = append(r, id)
						}
					} else {
						r = t.encodeChunkInto(c.text, r)
					}
				}
				results[i] = r
			}(i, allChunks[start:end])
		}
		wg.Wait()

		for _, r := range results {
			ids = append(ids, r...)
		}
	}

	if addBOS && t.vocab.BOS >= 0 {
		ids = append([]int32{t.vocab.BOS}, ids...)
	}
	return ids
}

// encodeChunkInto appends encoded tokens to ids and returns the extended slice
// Uses BPE merge algorithm when merges are available, otherwise longest-match
func (t *Tokenizer) encodeChunkInto(s string, ids []int32) []int32 {
	if t.typ == TokenizerWordPiece {
		return t.encodeWordPieceInto(s, ids)
	}

	if s == "" {
		return ids
	}

	// Apply encoding transformation
	// SentencePiece: replace space with ▁
	// BPE: convert bytes using precomputed table (GPT-2 byte-level encoding)
	var encoded string
	if t.typ == TokenizerSentencePiece {
		encoded = strings.ReplaceAll(s, " ", "▁")
	} else {
		var sb strings.Builder
		sb.Grow(len(s) * 2)
		for i := 0; i < len(s); i++ {
			sb.WriteRune(byteToRune[s[i]])
		}
		encoded = sb.String()
	}

	// Fast path: check if entire chunk is a single token
	if id, ok := t.vocab.Reverse[encoded]; ok {
		return append(ids, id)
	}

	return t.encodeBPEMerge(encoded, ids)
}

// encodeBPEMerge encodes using BPE merge algorithm.
// Repeatedly merges the pair with lowest rank until no more merges possible.
// Works correctly with empty merges (falls back to individual rune/byte encoding).
func (t *Tokenizer) encodeBPEMerge(encoded string, ids []int32) []int32 {
	// Start with individual runes as parts
	runes := []rune(encoded)
	parts := make([]string, len(runes))
	for i, r := range runes {
		parts[i] = string(r)
	}

	// Repeatedly merge lowest-rank pair
	for len(parts) > 1 {
		minRank := int(0x7FFFFFFF)
		minIdx := -1

		for i := 0; i < len(parts)-1; i++ {
			// Merge key format: "token1 token2" (space-separated)
			mergeKey := parts[i] + " " + parts[i+1]
			if rank, ok := t.vocab.Merges[mergeKey]; ok {
				if rank < minRank {
					minRank = rank
					minIdx = i
				}
			}
		}

		if minIdx < 0 {
			break // No more merges possible
		}

		// Merge the pair
		parts[minIdx] = parts[minIdx] + parts[minIdx+1]
		parts = append(parts[:minIdx+1], parts[minIdx+2:]...)
	}

	// Convert parts to token IDs
	for _, part := range parts {
		if id, ok := t.vocab.Reverse[part]; ok {
			ids = append(ids, id)
		} else {
			// Byte fallback for unknown parts
			for _, b := range []byte(part) {
				if id := t.vocab.byteTokens[b]; id >= 0 {
					ids = append(ids, id)
				}
			}
		}
	}

	return ids
}

// encodeWordPieceInto appends WordPiece tokens to ids and returns extended slice
// Uses greedy longest-match with ## prefix for continuation tokens
func (t *Tokenizer) encodeWordPieceInto(s string, ids []int32) []int32 {
	if s == "" {
		return ids
	}

	// Check if entire string is in vocabulary (common case)
	if id, ok := t.vocab.Reverse[s]; ok {
		return append(ids, id)
	}

	runes := []rune(s)
	start := 0

	for start < len(runes) {
		end := len(runes)
		found := false

		// Greedy longest-match
		for end > start {
			substr := string(runes[start:end])
			if start > 0 {
				// Continuation token: prefix with ##
				substr = "##" + substr
			}

			if id, ok := t.vocab.Reverse[substr]; ok {
				ids = append(ids, id)
				found = true
				start = end
				break
			}
			end--
		}

		if !found {
			// No match found - use [UNK] token or skip
			if t.unkToken >= 0 {
				ids = append(ids, t.unkToken)
			}
			start++
		}
	}

	return ids
}

// Decode converts token IDs back to text
func (t *Tokenizer) Decode(ids []int32) string {
	var sb strings.Builder

	for _, id := range ids {
		if int(id) >= len(t.vocab.Values) {
			continue
		}

		token := t.vocab.Values[id]

		switch t.typ {
		case TokenizerWordPiece:
			// WordPiece style: strip ## prefix from continuation tokens
			if strings.HasPrefix(token, "##") {
				sb.WriteString(token[2:])
			} else {
				sb.WriteString(token)
			}
		case TokenizerSentencePiece:
			// SentencePiece style: replace ▁ with space, decode byte tokens
			token = strings.ReplaceAll(token, "▁", " ")
			// Handle byte fallback tokens like <0x0D>
			if len(token) == 6 && token[0] == '<' && token[1] == '0' && token[2] == 'x' && token[5] == '>' {
				if v, err := strconv.ParseUint(token[3:5], 16, 8); err == nil {
					sb.WriteByte(byte(v))
					continue
				}
			}
			sb.WriteString(token)
		default:
			// GPT-2 BPE style: decode byte-level encoding
			for _, r := range token {
				switch {
				case r == 0x0100:
					// NULL byte (0x00 encoded as 0x0100)
					sb.WriteByte(0)
					continue
				case r == 0x0143:
					r = 0x00ad
				case r > 0x0100 && r <= 0x0120:
					r = r - 0x0100
				case r > 0x0120 && r <= 0x0142:
					r = r - 0x00a2
				}

				// Write as byte, not UTF-8 encoded rune
				sb.WriteByte(byte(r))
			}
		}
	}

	return sb.String()
}

// VocabSize returns the vocabulary size
func (t *Tokenizer) VocabSize() int {
	return len(t.vocab.Values)
}

// BOS returns the beginning of sequence token ID
func (t *Tokenizer) BOS() int32 {
	return t.vocab.BOS
}

// EOS returns the first end of sequence token ID (for backwards compatibility)
func (t *Tokenizer) EOS() int32 {
	if len(t.vocab.EOS) > 0 {
		return t.vocab.EOS[0]
	}
	return -1
}

// EOSTokens returns all end of sequence token IDs
func (t *Tokenizer) EOSTokens() []int32 {
	return t.vocab.EOS
}

// PAD returns the padding token ID, or -1 if not set
func (t *Tokenizer) PAD() int32 {
	return t.vocab.PAD
}

// IsEOS returns true if the token ID is an end of sequence token
func (t *Tokenizer) IsEOS(id int32) bool {
	for _, eos := range t.vocab.EOS {
		if id == eos {
			return true
		}
	}
	return false
}

// GetSpecialToken returns the token ID for a special token string
func (t *Tokenizer) GetSpecialToken(name string) (int32, bool) {
	id, ok := t.specialTokens[name]
	return id, ok
}

// LoadVocabMerges loads a tokenizer from vocab.json + merges.txt format (GPT-style)
func LoadVocabMerges(dir string) (*Tokenizer, error) {
	vocabPath := dir + "/vocab.json"
	mergesPath := dir + "/merges.txt"
	addedTokensPath := dir + "/added_tokens.json"

	// Load vocab
	vocabData, err := os.ReadFile(vocabPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read vocab.json: %w", err)
	}

	vocabMap := make(map[string]int32)
	if err := json.Unmarshal(vocabData, &vocabMap); err != nil {
		return nil, fmt.Errorf("failed to parse vocab.json: %w", err)
	}

	// Load merges
	mergesData, err := os.ReadFile(mergesPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read merges.txt: %w", err)
	}

	mergesLines := strings.Split(string(mergesData), "\n")
	var mergesStrings []string
	for _, line := range mergesLines {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		mergesStrings = append(mergesStrings, line)
	}

	// Build tokenizer
	t := &Tokenizer{
		vocab: &Vocabulary{
			Values:  make([]string, len(vocabMap)),
			Reverse: vocabMap,
			Merges:  make(map[string]int, len(mergesStrings)),
			BOS:     -1,
			PAD:     -1,
		},
		specialTokens: make(map[string]int32),
	}

	// Load added tokens if exists
	if addedData, err := os.ReadFile(addedTokensPath); err == nil {
		addedMap := make(map[string]int32)
		if err := json.Unmarshal(addedData, &addedMap); err == nil {
			for token, id := range addedMap {
				vocabMap[token] = id
				t.specialTokens[token] = id
			}
		}
	}

	// Build values array
	for token, id := range vocabMap {
		if int(id) >= len(t.vocab.Values) {
			newValues := make([]string, id+1)
			copy(newValues, t.vocab.Values)
			t.vocab.Values = newValues
		}
		t.vocab.Values[id] = token
	}

	// Build merges map
	for i, merge := range mergesStrings {
		t.vocab.Merges[merge] = i
	}

	// Load special token configuration from companion files
	loadSpecialTokenConfig(dir+"/", t)

	// Precompute byte token IDs for <0xNN> fallback
	initByteTokens(t)

	// GPT-2/tiktoken pretokenizer pattern
	pattern := `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
	re, err := regexp.Compile(rewritePatternForRE2(pattern))
	if err != nil {
		return nil, fmt.Errorf("failed to compile pretokenizer regex: %w", err)
	}
	t.pretokenizer = re

	return t, nil
}
