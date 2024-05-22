package convert

import (
	"cmp"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"google.golang.org/protobuf/proto"

	"github.com/uppercaveman/ollama-server/convert/sentencepiece"
	"github.com/uppercaveman/ollama-server/llm"
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

type Params struct {
	Architectures     []string `json:"architectures"`
	VocabSize         int      `json:"vocab_size"`
	HiddenSize        int      `json:"hidden_size"`       // n_embd
	HiddenLayers      int      `json:"num_hidden_layers"` // n_layer
	ContextSize       int      `json:"max_position_embeddings"`
	IntermediateSize  int      `json:"intermediate_size"`
	AttentionHeads    int      `json:"num_attention_heads"` // n_head
	KeyValHeads       int      `json:"num_key_value_heads"`
	NormEPS           float64  `json:"rms_norm_eps"`
	BoSTokenID        int      `json:"bos_token_id"`
	EoSTokenID        int      `json:"eos_token_id"`
	HeadDimension     int      `json:"head_dim"`
	PaddingTokenID    int      `json:"pad_token_id"`
	RopeFrequencyBase float64  `json:"rope_theta"`

	Experts     int `json:"num_local_experts"`
	ExpertsUsed int `json:"num_experts_per_tok"`

	PreTokenizer string

	ByteOrder
}

type ByteOrder interface {
	binary.ByteOrder
	binary.AppendByteOrder
}

type ModelArch interface {
	GetTensors() error
	LoadVocab() error
	WriteGGUF(io.WriteSeeker) error
}

type ModelFormat interface {
	GetLayerName(string) (string, error)
	GetTensors(string, *Params) ([]llm.Tensor, error)
	GetParams(string) (*Params, error)
	GetModelArch(string, string, *Params) (ModelArch, error)
}

type ModelData struct {
	Path    string
	Name    string
	Params  *Params
	Vocab   *Vocab
	Tensors []llm.Tensor
	Format  ModelFormat
}

func GetModelFormat(dirname string) (ModelFormat, error) {
	files, err := filepath.Glob(filepath.Join(dirname, "*"))
	if err != nil {
		return nil, err
	}

	for _, fn := range files {
		if strings.HasSuffix(fn, ".safetensors") {
			return &SafetensorFormat{}, nil
		} else if strings.HasSuffix(fn, ".bin") || strings.HasSuffix(fn, ".pth") {
			slog.Debug("model is torch")
			return &TorchFormat{}, nil
		}
	}

	return nil, fmt.Errorf("couldn't determine model format")
}

// Details on gguf's tokenizer can be found at:
// https://github.com/ggerganov/ggml/blob/master/docs/gguf.md#tokenizer
type Vocab struct {
	Tokens []string
	Scores []float32
	Types  []int32
	Merges []string
}

func LoadSentencePieceTokens(dirpath string, params *Params) (*Vocab, error) {
	slog.Info(fmt.Sprintf("reading vocab from %s", filepath.Join(dirpath, "tokenizer.model")))
	in, err := os.ReadFile(filepath.Join(dirpath, "tokenizer.model"))
	if err != nil {
		return nil, err
	}

	// To regenerate sentencepiece from the protobufs use:
	// protoc -I=./ --go_out=./ sentencepiece_model.proto
	modelProto := &sentencepiece.ModelProto{}
	if err := proto.Unmarshal(in, modelProto); err != nil {
		return nil, err
	}

	v := &Vocab{
		Tokens: make([]string, 0),
		Scores: make([]float32, 0),
		Types:  make([]int32, 0),
	}

	pieces := modelProto.GetPieces()
	for _, p := range pieces {
		v.Tokens = append(v.Tokens, p.GetPiece())
		v.Scores = append(v.Scores, p.GetScore())
		t := p.GetType()
		switch t {
		case sentencepiece.ModelProto_SentencePiece_UNKNOWN:
		case sentencepiece.ModelProto_SentencePiece_CONTROL:
		case sentencepiece.ModelProto_SentencePiece_UNUSED:
		case sentencepiece.ModelProto_SentencePiece_BYTE:
		default:
			t = sentencepiece.ModelProto_SentencePiece_NORMAL
		}
		v.Types = append(v.Types, int32(t))
	}

	slog.Info(fmt.Sprintf("vocab size: %d", len(v.Tokens)))

	// add any additional tokens
	addIn, err := os.ReadFile(filepath.Join(dirpath, "added_tokens.json"))
	if os.IsNotExist(err) {
		return v, nil
	} else if err != nil {
		return nil, err
	}

	slog.Info("reading user defined tokens")

	var extraTokenData map[string]int
	if err := json.Unmarshal(addIn, &extraTokenData); err != nil {
		return nil, err
	}

	type token struct {
		key string
		pos int
	}

	extraTokens := make([]token, 0)
	for k, id := range extraTokenData {
		extraTokens = append(extraTokens, token{k, id})
	}

	slices.SortFunc(extraTokens, func(a, b token) int {
		return cmp.Compare(a.pos, b.pos)
	})

	numToks := len(v.Tokens)

	for cnt, t := range extraTokens {
		// the token id should match the specific index for the total number of tokens
		if t.pos != cnt+numToks {
			return nil, fmt.Errorf("token ID '%d' for '%s' doesn't match total token size", t.pos, t.key)
		}
		v.Tokens = append(v.Tokens, t.key)
		v.Scores = append(v.Scores, -1000.0)
		v.Types = append(v.Types, tokenTypeUserDefined)
	}
	slog.Info(fmt.Sprintf("vocab size w/ extra tokens: %d", len(v.Tokens)))

	if params.VocabSize > len(v.Tokens) {
		missingTokens := params.VocabSize - len(v.Tokens)
		slog.Warn(fmt.Sprintf("vocab is missing %d tokens", missingTokens))
		for cnt := 0; cnt < missingTokens; cnt++ {
			v.Tokens = append(v.Tokens, fmt.Sprintf("<dummy%05d>", cnt+1))
			v.Scores = append(v.Scores, -1)
			v.Types = append(v.Types, tokenTypeUserDefined)
		}
	}

	return v, nil
}
