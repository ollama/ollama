package convert

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"log/slog"
	"maps"
	"os"
	"slices"
	"strings"
)

const (
	_ int32 = iota
	tokenTypeNormal
	tokenTypeUnknown
	tokenTypeControl
	tokenTypeUserDefined
	tokenTypeUnused
	tokenTypeByte
)

type Tokenizer struct {
	*Vocabulary
	SpecialVocabulary []*SpecialVocabulary
	Merges            []string

	Pre      string
	Template string
}

func parseTokenizer(fsys fs.FS, specialTokenTypes []string) (*Tokenizer, error) {
	v, err := parseVocabulary(fsys)
	if err != nil {
		return nil, err
	}

	t := &Tokenizer{
		Vocabulary: v,
		Pre:        "default",
	}

	addedTokens := make(map[string]token)
	if f, err := fsys.Open("tokenizer.json"); errors.Is(err, os.ErrNotExist) {
	} else if err != nil {
		return nil, err
	} else {
		defer f.Close()

		var tt tokenizer
		if err := json.NewDecoder(f).Decode(&tt); err != nil {
			return nil, err
		}

		for _, t := range tt.AddedTokens {
			addedTokens[t.Content] = t
		}

		if len(tt.Model.Merges) == 0 {
			// noop; merges is empty
		} else if err := json.Unmarshal(tt.Model.Merges, &t.Merges); err == nil {
			// noop; merges is []string
		} else if merges, err := func() ([][]string, error) {
			var merges [][]string
			if err := json.Unmarshal(tt.Model.Merges, &merges); err != nil {
				return nil, err
			}

			return merges, nil
		}(); err == nil {
			t.Merges = make([]string, len(merges))
			for i := range merges {
				t.Merges[i] = strings.Join(merges[i], " ")
			}
		} else {
			return nil, fmt.Errorf("could not parse tokenizer merges. expected []string or [][]string: %w", err)
		}

		sha256sum := sha256.New()
		for _, pt := range tt.PreTokenizer.PreTokenizers {
			switch pt.Type {
			case "Split":
				if pt.Pattern.Regex != "" {
					// create a checksum of all Split pretokenizers which should be sufficient
					// to identify the pretokenizer
					sha256sum.Write([]byte(pt.Pattern.Regex))
				}
			}
		}

		switch digest := hex.EncodeToString(sha256sum.Sum(nil)); digest {
		case "d98f9631be1e9607a9848c26c1f9eac1aa9fc21ac6ba82a2fc0741af9780a48f":
			t.Pre = "llama-bpe"
		case "03df5c5863ad70781dcfdef491ead25140f895fe8010964be0daefe27be32b02":
			t.Pre = "deepseek-llm"
		case "21cde974d587f0d54dc8d56b183cc1e6239600172035c68fbd6d4b9f8da0576e":
			t.Pre = "deepseek-coder"
		case "1ff7f41064896984db5d1bb6ff64fa4bc29007d08c1b439e505b7392777a319e":
			t.Pre = "qwen2"
		case "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855":
			// noop, empty pretokenizer
		default:
			slog.Warn("unknown pretokenizer, using default", "digest", digest)
		}
	}

	if f, err := fsys.Open("tokenizer_config.json"); errors.Is(err, os.ErrNotExist) {
		// noop
	} else if err != nil {
		return nil, err
	} else {
		defer f.Close()

		var p map[string]json.RawMessage
		if err := json.NewDecoder(f).Decode(&p); err != nil {
			return nil, err
		}

		if template, ok := p["chat_template"]; ok {
			var s []struct {
				Name     string `json:"name"`
				Template string `json:"template"`
			}
			if err := json.Unmarshal(template, &t.Template); err == nil {
				// noop
			} else if err := json.Unmarshal(template, &s); err == nil {
				for _, e := range s {
					if e.Name == "default" {
						t.Template = e.Template
						break
					}
				}
			} else {
				return nil, fmt.Errorf("invalid chat_template: %w", err)
			}
		}

		for _, st := range specialTokenTypes {
			sv := SpecialVocabulary{Type: st}
			if bts, ok := p[fmt.Sprintf("add_%s_token", st)]; ok {
				if err := json.Unmarshal(bts, &sv.AddToken); err != nil {
					return nil, err
				}
			}

			if bts, ok := p[fmt.Sprintf("%s_token", st)]; ok {
				var content string
				if err := json.Unmarshal(bts, &content); err != nil {
					var mm map[string]any
					if err := json.Unmarshal(bts, &mm); err != nil {
						continue
					}

					content, ok = mm["content"].(string)
					if !ok {
						continue
					}
				}

				sv.Content = content
			}

			if id, ok := addedTokens[sv.Content]; ok {
				sv.ID = id.ID
				t.SpecialVocabulary = append(t.SpecialVocabulary, &sv)
			}
		}
	}

	if f, err := fsys.Open("generation_config.json"); errors.Is(err, os.ErrNotExist) {
	} else if err != nil {
		return nil, err
	} else {
		defer f.Close()

		var p map[string]json.RawMessage
		if err := json.NewDecoder(f).Decode(&p); err != nil {
			return nil, err
		}

		for _, st := range specialTokenTypes {
			if bts, ok := p[fmt.Sprintf("%s_token_id", st)]; ok {
				var ids []int32
				if err := json.Unmarshal(bts, &ids); err != nil {
					// value is not a list so the existing ID is used
					continue
				}

				if i := slices.IndexFunc(t.SpecialVocabulary, func(sv *SpecialVocabulary) bool {
					return sv.Type == st
				}); i >= 0 {
					t.SpecialVocabulary[i].IDs = ids
				}
			}
		}
	}

	return t, nil
}

type tokenizer struct {
	AddedTokens []token `json:"added_tokens"`
	Model       struct {
		Type   string          `json:"type"`
		Vocab  map[string]int  `json:"vocab"`
		Merges json.RawMessage `json:"merges"`
	} `json:"model"`

	PreTokenizer struct {
		PreTokenizers []struct {
			Type    string `json:"type"`
			Pattern struct {
				Regex string `json:"Regex"`
			} `json:"pattern"`
		} `json:"pretokenizers"`
	} `json:"pre_tokenizer"`
}

type token struct {
	ID          int    `json:"id"`
	Content     string `json:"content"`
	Special     bool   `json:"special"`
	UserDefined bool
}

type Vocabulary struct {
	Model  string
	Tokens []string
	Scores []float32
	Types  []int32
}

func parseVocabularyFromTokenizer(fsys fs.FS) (*Vocabulary, error) {
	f, err := fsys.Open("tokenizer.json")
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var t tokenizer
	if err := json.NewDecoder(f).Decode(&t); err != nil {
		return nil, err
	}

	tokens := make(map[int]token, len(t.Model.Vocab))
	for k, v := range t.Model.Vocab {
		tokens[v] = token{
			ID:      v,
			Content: k,
		}
	}

	for _, token := range t.AddedTokens {
		token.UserDefined = true
		tokens[token.ID] = token
	}

	v := Vocabulary{Model: "gpt2"}
	for _, k := range slices.Sorted(maps.Keys(tokens)) {
		token := tokens[k]
		v.Tokens = append(v.Tokens, token.Content)
		v.Scores = append(v.Scores, float32(token.ID))

		switch {
		case token.Special:
			v.Types = append(v.Types, tokenTypeControl)
		case token.UserDefined:
			v.Types = append(v.Types, tokenTypeUserDefined)
		default:
			v.Types = append(v.Types, tokenTypeNormal)
		}
	}

	return &v, nil
}

func parseVocabulary(fsys fs.FS) (*Vocabulary, error) {
	patterns := []struct {
		Pattern string
		Func    func(fs.FS) (*Vocabulary, error)
	}{
		{"tokenizer.model", parseSentencePiece},
		{"tokenizer.json", parseVocabularyFromTokenizer},
	}

	for _, pattern := range patterns {
		if _, err := fs.Stat(fsys, pattern.Pattern); errors.Is(err, os.ErrNotExist) {
			continue
		} else if err != nil {
			return nil, err
		}

		return pattern.Func(fsys)
	}

	return nil, errors.New("unknown tokenizer format")
}

type SpecialVocabulary struct {
	Type     string
	ID       int
	Content  string
	AddToken bool

	// IDs is populated by generation_config.json
	IDs []int32
}

func (sv SpecialVocabulary) Key() string {
	switch t := sv.Type; t {
	case "bos", "eos", "cls", "mask":
		return t
	case "unk":
		return "unknown"
	case "sep":
		//nolint:misspell // this is an upstream typo
		return "seperator"
	case "pad":
		return "padding"
	}

	panic("unknown special vocabulary type")
}
