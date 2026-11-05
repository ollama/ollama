package tokenizer

import (
	"encoding/json"
	"fmt"
)

// TokenizerConfig holds optional configuration data that can be passed to LoadFromBytesWithConfig.
type TokenizerConfig struct {
	TokenizerConfigJSON  []byte // tokenizer_config.json content
	GenerationConfigJSON []byte // generation_config.json content
	SpecialTokensMapJSON []byte // special_tokens_map.json content
	ConfigJSON           []byte // config.json content
}

// LoadFromBytes loads a tokenizer from tokenizer.json bytes.
// This is useful when loading from blob storage where the file content is already in memory.
// Note: This won't load special token config from companion files. Use LoadFromBytesWithConfig
// to provide tokenizer_config.json data for proper PAD/EOS token loading.
func LoadFromBytes(data []byte) (*Tokenizer, error) {
	return loadFromTokenizerJSON(data)
}

// LoadFromBytesWithConfig loads a tokenizer from tokenizer.json bytes with additional config files.
// This is useful when loading from blob storage where companion config files are also blobs.
func LoadFromBytesWithConfig(data []byte, config *TokenizerConfig) (*Tokenizer, error) {
	t, err := loadFromTokenizerJSON(data)
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

// loadFromTokenizerJSON parses tokenizer.json content from bytes.
func loadFromTokenizerJSON(data []byte) (*Tokenizer, error) {
	var raw struct {
		Model struct {
			Type         string           `json:"type"` // "BPE"
			IgnoreMerges bool             `json:"ignore_merges"`
			Vocab        map[string]int32 `json:"vocab"`
			Merges       json.RawMessage  `json:"merges"` // Can be []string or [][]string (BPE only)
		} `json:"model"`
		PreTokenizer json.RawMessage `json:"pre_tokenizer"`
		Normalizer   struct {
			Type    string `json:"type"`
			Pattern struct {
				String *string `json:"String"`
			} `json:"pattern"`
			Content string `json:"content"`
		} `json:"normalizer"`
		Decoder     json.RawMessage `json:"decoder"`
		AddedTokens []struct {
			ID         int32  `json:"id"`
			Content    string `json:"content"`
			Special    bool   `json:"special"`
			Normalized bool   `json:"normalized"`
		} `json:"added_tokens"`
	}

	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("failed to parse tokenizer: %w", err)
	}

	// Covers SentencePiece and BPE models
	if raw.Model.Type != "BPE" {
		return nil, fmt.Errorf("unsupported tokenizer type: %s", raw.Model.Type)
	}

	// Parse merges - can be []string (Llama) or [][]string (GPT-OSS).
	var mergesStrings []string
	if raw.Model.Merges != nil {
		var mergesArrays [][]string
		if err := json.Unmarshal(raw.Model.Merges, &mergesStrings); err != nil {
			// Try array of arrays format
			if err := json.Unmarshal(raw.Model.Merges, &mergesArrays); err != nil {
				return nil, fmt.Errorf("failed to parse merges: %w", err)
			}
			// Convert [][]string to []string
			mergesStrings = make([]string, len(mergesArrays))
			for i, pair := range mergesArrays {
				if len(pair) != 2 {
					return nil, fmt.Errorf("failed to parse merges: expected merge pair of length 2, got %d", len(pair))
				}
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
		// Hugging Face ignores empty added tokens; they cannot consume input.
		if tok.Content == "" {
			continue
		}
		if int(tok.ID) >= len(t.vocab.Values) {
			newValues := make([]string, tok.ID+1)
			copy(newValues, t.vocab.Values)
			t.vocab.Values = newValues
		}
		t.vocab.Values[tok.ID] = tok.Content
		t.specialTokens[tok.Content] = tok.ID // Add ALL added_tokens to special tokens
	}

	// Precompute byte token IDs for <0xNN> fallback
	initByteTokens(t)

	// Determine tokenizer type
	switch {
	case detectSentencePiece(raw.Decoder):
		t.typ = TokenizerSentencePiece
	default:
		t.typ = TokenizerBPE
	}

	t.normalizeNFC = raw.Normalizer.Type == "NFC"
	t.normalizeSpaces = raw.Normalizer.Type == "Replace" && raw.Normalizer.Pattern.String != nil && *raw.Normalizer.Pattern.String == " " && raw.Normalizer.Content == "▁"
	t.ignoreMerges = raw.Model.IgnoreMerges
	var pre struct {
		Type string `json:"type"`
		metaspace
	}
	if len(raw.PreTokenizer) > 0 {
		if err := json.Unmarshal(raw.PreTokenizer, &pre); err != nil {
			return nil, fmt.Errorf("failed to parse pre_tokenizer: %w", err)
		}
	}
	if pre.Type == "Metaspace" {
		if pre.Replacement != "▁" || (pre.PrependScheme != "always" && pre.PrependScheme != "first" && pre.PrependScheme != "never") {
			return nil, fmt.Errorf("unsupported Metaspace pretokenizer: %s", raw.PreTokenizer)
		}
		t.metaspace = &pre.metaspace
		t.typ = TokenizerSentencePiece
	}
	if t.typ == TokenizerBPE {
		var err error
		t.pretokenizer, err = loadPretokenizers(raw.PreTokenizer)
		if err != nil {
			return nil, err
		}
	}

	// Hugging Face gives special tokens precedence when normalization maps
	// different added-token spellings to the same pattern.
	for _, special := range []bool{true, false} {
		for _, tok := range raw.AddedTokens {
			if tok.Special != special {
				continue
			}
			if tok.Normalized {
				t.normalizedTokenMatcher.add(t.normalize(tok.Content), tok.Content)
			} else {
				t.specialTokenMatcher.add(tok.Content, tok.Content)
			}
		}
	}

	return t, nil
}

type specialTokenConfigData struct {
	tokenizerConfigJSON  []byte
	generationConfigJSON []byte
	specialTokensMapJSON []byte
	configJSON           []byte
}

func applySpecialTokenConfig(t *Tokenizer, config specialTokenConfigData) {
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
	if len(config.generationConfigJSON) > 0 {
		var genConfig struct {
			EOSTokenID interface{} `json:"eos_token_id"`
			BOSTokenID interface{} `json:"bos_token_id"`
		}
		if err := json.Unmarshal(config.generationConfigJSON, &genConfig); err == nil {
			if ids := parseTokenIDs(genConfig.EOSTokenID); len(ids) > 0 {
				t.vocab.EOS = ids
			}
			if ids := parseTokenIDs(genConfig.BOSTokenID); len(ids) > 0 {
				t.vocab.BOS = ids[0]
			}
		}
	}

	// Priority 2: config.json
	if len(config.configJSON) > 0 && (len(t.vocab.EOS) == 0 || t.vocab.BOS < 0) {
		var modelConfig struct {
			EOSTokenID interface{} `json:"eos_token_id"`
			BOSTokenID interface{} `json:"bos_token_id"`
		}
		if err := json.Unmarshal(config.configJSON, &modelConfig); err == nil {
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
	if len(config.tokenizerConfigJSON) > 0 {
		var tokConfig struct {
			BOSToken    interface{} `json:"bos_token"`
			EOSToken    interface{} `json:"eos_token"`
			PADToken    interface{} `json:"pad_token"`
			AddBOSToken *bool       `json:"add_bos_token"`
			AddEOSToken *bool       `json:"add_eos_token"`
		}
		if err := json.Unmarshal(config.tokenizerConfigJSON, &tokConfig); err == nil {
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
	if len(config.specialTokensMapJSON) > 0 {
		var tokensMap map[string]interface{}
		if err := json.Unmarshal(config.specialTokensMapJSON, &tokensMap); err == nil {
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

// loadSpecialTokenConfigFromBytes loads special token configuration from byte slices.
func loadSpecialTokenConfigFromBytes(t *Tokenizer, config *TokenizerConfig) {
	applySpecialTokenConfig(t, specialTokenConfigData{
		tokenizerConfigJSON:  config.TokenizerConfigJSON,
		generationConfigJSON: config.GenerationConfigJSON,
		specialTokensMapJSON: config.SpecialTokensMapJSON,
		configJSON:           config.ConfigJSON,
	})
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
	for b := range 256 {
		token := fmt.Sprintf("<0x%02X>", b)
		if id, ok := t.vocab.Reverse[token]; ok {
			t.vocab.byteTokens[b] = id
		}
	}
}
