package convert

import (
	"encoding/json"
	"io/ioutil"
	"os"
)

type Tokenizer struct {
	Version     string         `json:"version"`
	AddedTokens []Token        `json:"added_tokens"`
	Model       TokenizerModel `json:"model"`
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

func (t *Tokenizer) getMaxID() int {
	var maxID int
	for _, v := range t.Model.Vocab {
		maxID = max(maxID, v)
	}

	for _, v := range t.AddedTokens {
		maxID = max(maxID, v.ID)
	}
	return maxID
}

func newTokenizer(dirpath string) (*Tokenizer, error) {
	f, err := os.Open(dirpath)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	data, err := ioutil.ReadAll(f)
	if err != nil {
		return nil, err
	}

	var tdata Tokenizer

	if err := json.Unmarshal(data, &tdata); err != nil {
		return nil, err
	}

	maxID := tdata.getMaxID()
	tdata.Model.Tokens = make([]Token, maxID+1)

	for k, v := range tdata.Model.Vocab {
		tdata.Model.Tokens[v] = Token{ID: v, Content: k, Special: false, UserDefined: false}
	}

	for _, v := range tdata.AddedTokens {
		v.UserDefined = true
		tdata.Model.Tokens[v.ID] = v
	}

	return &tdata, nil
}
