package convert

import (
	"cmp"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"slices"

	"golang.org/x/exp/maps"
)

type Tokenizer struct {
	Version     string         `json:"version"`
	AddedTokens []Token        `json:"added_tokens"`
	Model       TokenizerModel `json:"model"`

	PreTokenizer struct {
		PreTokenizers []struct {
			Type    string `json:"type"`
			Pattern struct {
				Regex string `json:"Regex"`
			} `json:"pattern"`
		} `json:"pretokenizers"`
	} `json:"pre_tokenizer"`
}

type TokenizerModel struct {
	Type   string         `json:"type"`
	Vocab  map[string]int `json:"vocab"`
	Merges []string       `json:"merges"`
	Tokens []Token
}

type Token struct {
	ID          int    `json:"id"`
	Content     string `json:"content"`
	Special     bool   `json:"special"`
	UserDefined bool
}

func (t *Token) Type() int32 {
	switch {
	case t.Special:
		return tokenTypeControl
	case t.UserDefined:
		return tokenTypeUserDefined
	default:
		return tokenTypeNormal
	}
}

func (t *Tokenizer) maxID() int {
	return max(
		slices.Max(maps.Values(t.Model.Vocab)),
		slices.MaxFunc(t.AddedTokens, func(a, b Token) int {
			return cmp.Compare(a.ID, b.ID)
		}).ID,
	)
}

func parseTokens(dirpath string) (pre string, tokens []Token, merges []string, err error) {
	f, err := os.Open(dirpath)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	var t Tokenizer
	if err := json.NewDecoder(f).Decode(&t); err != nil {
		return "", nil, nil, err
	}

	tokens = make([]Token, t.maxID()+1)
	for k, v := range t.Model.Vocab {
		tokens[v] = Token{ID: v, Content: k, Special: false, UserDefined: false}
	}

	for _, v := range t.AddedTokens {
		v.UserDefined = true
		tokens[v.ID] = v
	}

	sha256sum := sha256.New()
	for _, pt := range t.PreTokenizer.PreTokenizers {
		switch pt.Type {
		case "Split":
			if pt.Pattern.Regex != "" {
				sha256sum.Write([]byte(pt.Pattern.Regex))
			}
		}
	}

	switch digest := fmt.Sprintf("%x", sha256sum.Sum(nil)); digest {
	case "d98f9631be1e9607a9848c26c1f9eac1aa9fc21ac6ba82a2fc0741af9780a48f":
		pre = "llama-bpe"
	case "03df5c5863ad70781dcfdef491ead25140f895fe8010964be0daefe27be32b02":
		pre = "deepseek-llm"
	case "21cde974d587f0d54dc8d56b183cc1e6239600172035c68fbd6d4b9f8da0576e":
		pre = "deepseek-coder"
	default:
		slog.Warn("unknown pretokenizer, using default", "digest", digest)
		pre = "default"
	}

	return pre, tokens, t.Model.Merges, nil
}
