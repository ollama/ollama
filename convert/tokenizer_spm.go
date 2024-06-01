package convert

import (
	"cmp"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"

	"google.golang.org/protobuf/proto"

	"github.com/ollama/ollama/convert/sentencepiece"
)

func parseSentencePiece(d string) (*Vocabulary, error) {
	bts, err := os.ReadFile(filepath.Join(d, "tokenizer.model"))
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
			v.Types = append(v.Types, int32(sentencepiece.ModelProto_SentencePiece_NORMAL))
		}
	}

	f, err := os.Open(filepath.Join(d, "added_tokens.json"))
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

	n := len(v.Tokens)
	for i, t := range ts {
		if t.id != i+n {
			return nil, fmt.Errorf("invalid token id: %d", t.id)
		}

		v.Tokens = append(v.Tokens, t.content)
		v.Scores = append(v.Scores, -1000.0)
		v.Types = append(v.Types, tokenTypeUserDefined)
	}

	return &v, nil
}
