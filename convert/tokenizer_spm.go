package convert

import (
	"cmp"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"log/slog"
	"os"
	"reflect"
	"slices"

	"google.golang.org/protobuf/proto"

	"github.com/ollama/ollama/convert/sentencepiece"
)

func parseSentencePiece(fsys fs.FS) (*Vocabulary, error) {
	slog.Debug("using spm vocabulary")

	ast, err := parseAdditionalSpecialTokens(fsys)
	if err != nil {
		return nil, err
	}

	bts, err := fs.ReadFile(fsys, "tokenizer.model")
	if err != nil {
		return nil, err
	}

	var spm sentencepiece.ModelProto
	if err := proto.Unmarshal(bts, &spm); err != nil {
		return nil, err
	}

	v := Vocabulary{Model: "llama"}
	for _, piece := range spm.GetPieces() {
		v.Tokens = append(v.Tokens, piece.GetPiece())
		v.Scores = append(v.Scores, piece.GetScore())

		switch t := piece.GetType(); t {
		case sentencepiece.ModelProto_SentencePiece_UNKNOWN,
			sentencepiece.ModelProto_SentencePiece_CONTROL,
			sentencepiece.ModelProto_SentencePiece_UNUSED,
			sentencepiece.ModelProto_SentencePiece_BYTE:
			v.Types = append(v.Types, int32(t))
		default:
			tt := int32(sentencepiece.ModelProto_SentencePiece_NORMAL)

			// temporary fix to handle gemma3 broken configs
			// TODO(parthsareen): allow reading of tokenizer.json to allow managing special tokens when using spm
			if slices.Contains([]string{"<end_of_turn>", "<start_of_turn>", "<start_function_declaration>", "<end_function_declaration>", "<start_function_call>", "<end_function_call>", "<start_function_response>", "<end_function_response>", "<escape>"}, piece.GetPiece()) {
				tt = int32(sentencepiece.ModelProto_SentencePiece_CONTROL)
			}

			for _, t := range ast {
				if t.Content == piece.GetPiece() {
					tt = int32(sentencepiece.ModelProto_SentencePiece_CONTROL)
					break
				}
			}

			v.Types = append(v.Types, tt)
		}
	}

	f, err := fsys.Open("added_tokens.json")
	if errors.Is(err, os.ErrNotExist) {
		return &v, nil
	} else if err != nil {
		return nil, err
	}
	defer f.Close()

	var atm map[string]int
	if err := json.NewDecoder(f).Decode(&atm); err != nil {
		return nil, err
	}

	type t struct {
		id      int
		content string
	}

	var ts []t
	for content, id := range atm {
		ts = append(ts, t{id, content})
	}

	slices.SortFunc(ts, func(i, j t) int {
		return cmp.Compare(i.id, j.id)
	})

	for _, t := range ts {
		if t.id < len(v.Tokens) {
			if v.Tokens[t.id] == t.content {
				slog.Warn("tokenizer", "duplicate token", t.content, "id", t.id)
				continue
			}
			return nil, fmt.Errorf("token mismatch: %s != %s at pos [%d]", t.content, v.Tokens[t.id], t.id)
		}
		if t.id != len(v.Tokens) {
			return nil, fmt.Errorf("invalid token id: [%d] as pos [%d]", t.id, len(v.Tokens))
		}

		v.Tokens = append(v.Tokens, t.content)
		v.Scores = append(v.Scores, -1000.0)
		v.Types = append(v.Types, tokenTypeUserDefined)
	}

	return &v, nil
}

type specialToken struct {
	Content    string `json:"content"`
	Lstrip     bool   `json:"lstrip"`
	Normalized bool   `json:"normalized"`
	Rstrip     bool   `json:"rstrip"`
	SingleWord bool   `json:"single_word"`
}

func parseAdditionalSpecialTokens(fsys fs.FS) ([]specialToken, error) {
	f, err := fsys.Open("special_tokens_map.json")
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	} else if err != nil {
		return nil, err
	}
	defer f.Close()

	var m struct {
		AdditionalSpecialTokens any `json:"additional_special_tokens"`
	}

	if err := json.NewDecoder(f).Decode(&m); err != nil {
		return nil, err
	}

	var ast []specialToken

	switch st := m.AdditionalSpecialTokens.(type) {
	case []string:
		for _, s := range st {
			ast = append(ast, specialToken{Content: s})
		}
	case []any:
		for _, s := range st {
			// marshal and unmarshal the object to get the special token
			tMap := s.(map[string]any)
			data, err := json.Marshal(tMap)
			if err != nil {
				return nil, err
			}

			var token specialToken
			err = json.Unmarshal(data, &token)
			if err != nil {
				return nil, err
			}

			ast = append(ast, token)
		}

	default:
		slog.Warn("special token", "unknown token", reflect.TypeOf(st))
	}

	slog.Debug("spm tokenizer", "additional tokens", ast)

	return ast, nil
}
