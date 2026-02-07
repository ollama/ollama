package tokenizer

import (
	"encoding/json"
	"errors"
	"io"
	"os"

	"github.com/ollama/ollama/types/model"
)

const (
	TOKEN_TYPE_NORMAL = iota + 1
	TOKEN_TYPE_UNKNOWN
	TOKEN_TYPE_CONTROL
	TOKEN_TYPE_USER_DEFINED
	TOKEN_TYPE_UNUSED
	TOKEN_TYPE_BYTE
)

type Tokenizer interface {
	Encode(s string, addSpecial bool) ([]int32, error)
	Decode([]int32) (string, error)
	Is(int32, Special) bool
	Vocabulary() *Vocabulary
}

func New(root *model.Root) (Tokenizer, error) {
	f, err := root.Open("tokenizer.json")
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var tokenizer struct {
		Model struct {
			Type   string           `json:"type"`
			Vocab  map[string]int32 `json:"vocab"`
			Merges json.RawMessage  `json:"merges"`
		} `json:"model"`

		PreTokenizer json.RawMessage `json:"pre_tokenizer"`
		Decoder      json.RawMessage `json:"decoder"`

		AddedTokens []struct {
			ID      int32  `json:"id"`
			Content string `json:"content"`
			Special bool   `json:"special"`
		} `json:"added_tokens"`
	}

	if err := json.NewDecoder(f).Decode(&tokenizer); err != nil {
		return nil, err
	}

	special := make(map[int32]struct{})
	for _, token := range tokenizer.AddedTokens {
		tokenizer.Model.Vocab[token.Content] = token.ID
		special[token.ID] = struct{}{}
	}

	vocab, err := specialTokens(root, tokenizer.Model.Vocab)
	if err != nil {
		return nil, err
	}

	vocab.Values = make([]string, len(tokenizer.Model.Vocab))
	vocab.Scores = make([]float32, len(tokenizer.Model.Vocab))
	vocab.Types = make([]int32, len(tokenizer.Model.Vocab))
	for content, id := range tokenizer.Model.Vocab {
		vocab.Values[id] = content
		vocab.Scores[id] = float32(id)
		vocab.Types[id] = TOKEN_TYPE_NORMAL
		if _, ok := special[id]; ok {
			vocab.Types[id] = TOKEN_TYPE_USER_DEFINED
		}
	}

	if tokenizer.Model.Merges != nil {
		var pairs [][]string
		if err := json.Unmarshal(tokenizer.Model.Merges, &pairs); err == nil {
			vocab.Merges = make([]string, len(pairs))
			for i, pair := range pairs {
				vocab.Merges[i] = pair[0] + " " + pair[1]
			}
		} else if err := json.Unmarshal(tokenizer.Model.Merges, &vocab.Merges); err != nil {
			return nil, err
		}
	}

	vocab.valuesOnce.Do(func() {})
	vocab.values = tokenizer.Model.Vocab

	if tokenizer.Model.Type == "WordPiece" {
		return NewWordPiece(vocab, true), nil
	}

	if tokenizer.Decoder != nil {
		var decoder struct {
			Type     string `json:"type"`
			Decoders []struct {
				Type    string `json:"type"`
				Pattern struct {
					String string `json:"string"`
				} `json:"pattern"`
			} `json:"decoders"`
		}

		if err := json.Unmarshal(tokenizer.Decoder, &decoder); err != nil {
			return nil, err
		}

		if decoder.Type == "Sequence" {
			for _, d := range decoder.Decoders {
				if d.Type == "Replace" && d.Pattern.String == "▁" {
					return NewSentencePiece(vocab), nil
				}
			}
		}
	}

	var pretokenizers []string
	if tokenizer.PreTokenizer != nil {
		var pretokenizer struct {
			Type          string `json:"type"`
			Pretokenizers []struct {
				Type    string `json:"type"`
				Pattern struct {
					Regex string
				} `json:"pattern"`
				IndividualDigits bool `json:"individual_digits"`
			}
		}

		if err := json.Unmarshal(tokenizer.PreTokenizer, &pretokenizer); err != nil {
			return nil, err
		}

		if pretokenizer.Type == "Sequence" {
			for _, pretokenizer := range pretokenizer.Pretokenizers {
				switch pretokenizer.Type {
				case "Digits":
					if pretokenizer.IndividualDigits {
						pretokenizers = append(pretokenizers, `\d`)
					} else {
						pretokenizers = append(pretokenizers, `\d+`)
					}
				case "Punctuation":
					pretokenizers = append(pretokenizers, `[^\p{L}\p{N}]+`)
				case "Split":
					pretokenizers = append(pretokenizers, pretokenizer.Pattern.Regex)
				case "WhitespaceSplit":
					pretokenizers = append(pretokenizers, `\s+(?!\S)|\s+`)
				}
			}
		}
	}

	return NewBytePairEncoding(vocab, pretokenizers...), nil
}

// valueOrValues is a type that can unmarshal from either a single value or an array of values.
type valueOrValues[E any] []E

func (m *valueOrValues[E]) UnmarshalJSON(data []byte) error {
	var s []E
	if err := json.Unmarshal(data, &s); err != nil {
		var e E
		if err := json.Unmarshal(data, &e); err != nil {
			return err
		}
		s = []E{e}
	}
	*m = valueOrValues[E](s)
	return nil
}

type specialTokenIDs struct {
	BOSTokenID valueOrValues[int32] `json:"bos_token_id"`
	EOSTokenID valueOrValues[int32] `json:"eos_token_id"`
}

// stringOrContent is a type that can unmarshal from either a string or an object with a "content" field.
type stringOrContent string

func (t *stringOrContent) UnmarshalJSON(data []byte) error {
	var s string
	if err := json.Unmarshal(data, &s); err != nil {
		var m map[string]any
		if err := json.Unmarshal(data, &m); err != nil {
			return err
		}
		if content, ok := m["content"].(string); ok {
			s = content
		}
	}
	*t = stringOrContent(s)
	return nil
}

func specialTokens(root *model.Root, values map[string]int32) (*Vocabulary, error) {
	var vocab Vocabulary
	for _, c := range []struct {
		name string
		fn   func(io.Reader) error
	}{
		{
			name: "generation_config.json",
			fn: func(r io.Reader) error {
				var c specialTokenIDs
				if err := json.NewDecoder(r).Decode(&c); err != nil {
					return err
				}

				vocab.BOS = c.BOSTokenID
				vocab.EOS = c.EOSTokenID
				return nil
			},
		},
		{
			name: "config.json",
			fn: func(r io.Reader) error {
				var c specialTokenIDs
				if err := json.NewDecoder(r).Decode(&c); err != nil {
					return err
				}

				if len(vocab.BOS) == 0 {
					vocab.BOS = c.BOSTokenID
				}

				if len(vocab.EOS) == 0 {
					vocab.EOS = c.EOSTokenID
				}

				return nil
			},
		},
		{
			name: "tokenizer_config.json",
			fn: func(r io.Reader) error {
				var c struct {
					BOSToken    stringOrContent `json:"bos_token"`
					EOSToken    stringOrContent `json:"eos_token"`
					PADToken    stringOrContent `json:"pad_token"`
					AddBOSToken bool            `json:"add_bos_token"`
					AddEOSToken bool            `json:"add_eos_token"`
				}
				if err := json.NewDecoder(r).Decode(&c); err != nil {
					return err
				}

				if len(vocab.BOS) == 0 && c.BOSToken != "" {
					if id, ok := values[string(c.BOSToken)]; ok {
						vocab.BOS = []int32{id}
					}
				}

				if len(vocab.EOS) == 0 && c.EOSToken != "" {
					if id, ok := values[string(c.EOSToken)]; ok {
						vocab.EOS = []int32{id}
					}
				}

				vocab.AddBOS = c.AddBOSToken
				vocab.AddEOS = c.AddEOSToken
				return nil
			},
		},
		{
			name: "special_tokens_map.json",
			fn: func(r io.Reader) error {
				var c map[string]stringOrContent
				if err := json.NewDecoder(r).Decode(&c); err != nil {
					return err
				}

				if bos, ok := c["bos_token"]; ok && len(vocab.BOS) == 0 {
					if id, ok := values[string(bos)]; ok {
						vocab.BOS = []int32{id}
					}
				}

				if eos, ok := c["eos_token"]; ok && len(vocab.EOS) == 0 {
					if id, ok := values[string(eos)]; ok {
						vocab.EOS = []int32{id}
					}
				}

				return nil
			},
		},
	} {
		if err := func() error {
			f, err := root.Open(c.name)
			if errors.Is(err, os.ErrNotExist) {
				return nil
			} else if err != nil {
				return err
			}
			defer f.Close()

			return c.fn(f)
		}(); err != nil {
			return nil, err
		}
	}

	return &vocab, nil
}
