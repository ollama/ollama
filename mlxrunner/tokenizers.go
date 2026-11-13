package mlxrunner

import (
	"context"
	"fmt"
	"math"
	"net/http"
	"slices"
	"sync"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/mlxrunner/model"
	"github.com/ollama/ollama/mlxrunner/tokenizer"
	modeltypes "github.com/ollama/ollama/types/model"
)

// ErrNoTokenizer reports a model whose architecture has no text tokenizer.
var ErrNoTokenizer = model.ErrNoTokenizer

// tokenizerCacheSize bounds the tokenizers a [Tokenizers] keeps.
const tokenizerCacheSize = 4

// Tokenizers builds MLX tokenizers in the calling process with the loader the
// runner uses, so tokenizing never starts a runner. The zero value is ready.
type Tokenizers struct {
	mu      sync.Mutex        // guards entries
	entries []*tokenizerEntry // most recently used last
}

type tokenizerEntry struct {
	digest string
	once   sync.Once
	tok    *Tokenizer // set by once
	err    error      // set by once
}

// Tokenizer returns the tokenizer of the manifest that name and runner
// select, which must still be manifestDigest.
func (c *Tokenizers) Tokenizer(name, runner, manifestDigest string) (*Tokenizer, error) {
	c.mu.Lock()
	i := slices.IndexFunc(c.entries, func(e *tokenizerEntry) bool { return e.digest == manifestDigest })
	var e *tokenizerEntry
	if i >= 0 {
		e = c.entries[i]
		c.entries = append(slices.Delete(c.entries, i, i+1), e)
	} else {
		e = &tokenizerEntry{digest: manifestDigest}
		c.entries = append(c.entries, e)
		if len(c.entries) > tokenizerCacheSize {
			c.entries = slices.Delete(c.entries, 0, 1)
		}
	}
	c.mu.Unlock()

	e.once.Do(func() {
		e.tok, e.err = loadTokenizer(name, runner, manifestDigest)
	})
	if e.err != nil {
		c.mu.Lock()
		c.entries = slices.DeleteFunc(c.entries, func(x *tokenizerEntry) bool { return x == e })
		c.mu.Unlock()
	}
	return e.tok, e.err
}

func loadTokenizer(name, runner, manifestDigest string) (*Tokenizer, error) {
	mf, err := manifest.ParseNamedManifestForRunner(modeltypes.ParseName(name), runner)
	if err != nil {
		return nil, err
	}
	digest := mf.SelectedDigest()
	if digest == "" {
		digest = mf.Digest()
	}
	if digest != manifestDigest {
		return nil, fmt.Errorf("model %q changed while tokenizing; retry the request", name)
	}
	tok, err := model.LoadTokenizer(mf)
	if err != nil {
		return nil, err
	}
	return &Tokenizer{tok: tok}, nil
}

// Tokenizer converts between text and an MLX model's token IDs.
type Tokenizer struct {
	tok *tokenizer.Tokenizer
}

// TokenizePrompt encodes content. addSpecial adds BOS as the runner does
// for a raw prompt.
func (t *Tokenizer) TokenizePrompt(_ context.Context, content string, addSpecial bool) ([]int, error) {
	ids := t.tok.Encode(content, addSpecial && t.tok.AddBOS())
	tokens := make([]int, len(ids))
	for i, id := range ids {
		tokens[i] = int(id)
	}
	return tokens, nil
}

// Detokenize decodes token IDs.
func (t *Tokenizer) Detokenize(_ context.Context, tokens []int) (string, error) {
	ids := make([]int32, len(tokens))
	for i, id := range tokens {
		if id < 0 || id >= t.tok.VocabSize() || id > math.MaxInt32 {
			return "", api.StatusError{StatusCode: http.StatusBadRequest, ErrorMessage: fmt.Sprintf("token ID %d is outside the vocabulary", id)}
		}
		ids[i] = int32(id)
	}
	return t.tok.Decode(ids), nil
}
