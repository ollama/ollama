package convert

import (
	"cmp"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"slices"

	"google.golang.org/protobuf/proto"

	"github.com/jmorganca/ollama/convert/sentencepiece"
	"github.com/jmorganca/ollama/llm"
)

type Params struct {
	Architectures    []string `json:"architectures"`
	VocabSize        int      `json:"vocab_size"`
	HiddenSize       int      `json:"hidden_size"`       // n_embd
	HiddenLayers     int      `json:"num_hidden_layers"` // n_layer
	ContextSize      int      `json:"max_position_embeddings"`
	IntermediateSize int      `json:"intermediate_size"`
	AttentionHeads   int      `json:"num_attention_heads"` // n_head
	KeyValHeads      int      `json:"num_key_value_heads"`
	NormEPS          float64  `json:"rms_norm_eps"`
	RopeFreqBase     float64  `json:"rope_theta"`
	BoSTokenID       int      `json:"bos_token_id"`
	EoSTokenID       int      `json:"eos_token_id"`

	ByteOrder
}

type ByteOrder interface {
	binary.ByteOrder
	binary.AppendByteOrder
}

type MetaData struct {
	Type    string `mapstructure:"dtype"`
	Shape   []int  `mapstructure:"shape"`
	Offsets []int  `mapstructure:"data_offsets"`
}

func GetSafeTensors(dirpath string, params *Params) (tensors []*llm.Tensor, err error) {
	files, err := filepath.Glob(filepath.Join(dirpath, "/model-*.safetensors"))
	if err != nil {
		return nil, err
	}

	for _, f := range files {
		ts, err := ParseSafetensor(f, params)
		if err != nil {
			slog.Error("%v", err)
			return nil, err
		}

		tensors = append(tensors, ts...)
	}

	return tensors, nil
}

func GetParams(dirpath string) (*Params, error) {
	f, err := os.Open(filepath.Join(dirpath, "config.json"))
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var params Params

	d := json.NewDecoder(f)
	err = d.Decode(&params)
	if err != nil {
		return nil, err
	}

	params.ByteOrder = binary.LittleEndian
	return &params, nil
}

// Details on gguf's tokenizer can be found at:
// https://github.com/ggerganov/ggml/blob/master/docs/gguf.md#tokenizer
type Vocab struct {
	Tokens []string
	Scores []float32
	Types  []int32
}

func LoadTokens(dirpath string) (*Vocab, error) {
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
		v.Types = append(v.Types, int32(llm.GGUFTokenUserDefined))
	}
	slog.Info(fmt.Sprintf("vocab size w/ extra tokens: %d", len(v.Tokens)))

	return v, nil
}

func WriteGGUF(name string, tensors []*llm.Tensor, params *Params, vocab *Vocab) (string, error) {
	kv := llm.KV{
		"general.architecture":                   "llama",
		"general.name":                           name,
		"llama.context_length":                   uint32(params.ContextSize),
		"llama.embedding_length":                 uint32(params.HiddenSize),
		"llama.block_count":                      uint32(params.HiddenLayers),
		"llama.feed_forward_length":              uint32(params.IntermediateSize),
		"llama.rope.dimension_count":             uint32(128),
		"llama.attention.head_count":             uint32(params.AttentionHeads),
		"llama.attention.head_count_kv":          uint32(params.KeyValHeads),
		"llama.attention.layer_norm_rms_epsilon": float32(params.NormEPS),
		"llama.rope.freq_base":                   float32(params.RopeFreqBase),
		"general.file_type":                      uint32(1),
		"tokenizer.ggml.model":                   "llama",

		"tokenizer.ggml.tokens":     vocab.Tokens,
		"tokenizer.ggml.scores":     vocab.Scores,
		"tokenizer.ggml.token_type": vocab.Types,

		"tokenizer.ggml.bos_token_id":     uint32(params.BoSTokenID),
		"tokenizer.ggml.eos_token_id":     uint32(params.EoSTokenID),
		"tokenizer.ggml.unknown_token_id": uint32(0),
		"tokenizer.ggml.add_bos_token":    true,
		"tokenizer.ggml.add_eos_token":    false,

		// llamacpp sets the chat template, however we don't need to set it since we pass it in through a layer
		// "tokenizer.chat_template": "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}" // XXX removeme,
	}

	f, err := os.CreateTemp("", "ollama-gguf")
	if err != nil {
		return "", err
	}
	defer f.Close()

	m := llm.NewGGUFV3(params.ByteOrder)
	if err := m.Encode(f, kv, tensors); err != nil {
		return "", err
	}

	return f.Name(), nil
}
