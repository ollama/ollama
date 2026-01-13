package convert

import (
	"cmp"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"iter"
	"log/slog"
	"maps"
	"os"
	"slices"
	"strings"

	ofs "github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/fs/ggml"
)

type ModelParameters struct {
	Architectures []string `json:"architectures"`
	VocabSize     uint32   `json:"vocab_size"`

	// TODO is this needed?
	ModelType string `json:"model_type"`

	TextModel struct {
		VocabSize  uint32 `json:"vocab_size"`
		HiddenSize uint32 `json:"hidden_size"`
		ModelType  string `json:"model_type"`
	} `json:"text_config"`
}

type AdapterParameters struct {
	Alpha          uint32 `json:"lora_alpha"`
	LoraLayers     uint32 `json:"lora_layers"`
	LoraParameters struct {
		Rank  uint32  `json:"rank"`
		Alpha float32 `json:"alpha"`
		Scale float32 `json:"scale"`
	} `json:"lora_parameters"`
}

type KV map[string]any

func (kv KV) Architecture() string {
	return kv.String("general.architecture", "unknown")
}

type valueTypes interface {
	uint8 | int8 | uint16 | int16 |
		uint32 | int32 | uint64 | int64 |
		string | float32 | float64 | bool
}

type arrayValueTypes interface {
	[]uint8 | []int8 | []uint16 | []int16 |
		[]uint32 | []int32 | []uint64 | []int64 |
		[]string | []float32 | []float64 | []bool
}

func keyValue[T valueTypes | arrayValueTypes](kv KV, key string, defaultValue ...T) (T, bool) {
	if !strings.HasPrefix(key, "tokenizer.") && !strings.HasPrefix(key, "general.") {
		key = kv.Architecture() + "." + key
	}

	if val, ok := kv[key].(T); ok {
		return val, true
	}
	return defaultValue[0], false
}

func (kv KV) String(key string, defaultValue ...string) string {
	val, _ := keyValue(kv, key, append(defaultValue, "")...)
	return val
}

func (kv KV) Uint(key string, defaultValue ...uint32) uint32 {
	val, _ := keyValue(kv, key, append(defaultValue, 0)...)
	return val
}

func (kv KV) Float(key string, defaultValue ...float32) float32 {
	val, _ := keyValue(kv, key, append(defaultValue, 0)...)
	return val
}

func (kv KV) Bool(key string, defaultValue ...bool) bool {
	val, _ := keyValue(kv, key, append(defaultValue, false)...)
	return val
}

func (kv KV) Strings(key string, defaultValue ...[]string) []string {
	val, _ := keyValue(kv, key, append(defaultValue, []string{""})...)
	return val
}

func (kv KV) Ints(key string, defaultValue ...[]int32) []int32 {
	val, _ := keyValue(kv, key, append(defaultValue, []int32{0})...)
	return val
}

func (kv KV) Uints(key string, defaultValue ...[]uint32) []uint32 {
	val, _ := keyValue(kv, key, append(defaultValue, []uint32{0})...)
	return val
}

func (kv KV) Floats(key string, defaultValue ...[]float32) []float32 {
	val, _ := keyValue(kv, key, append(defaultValue, []float32{0})...)
	return val
}

func (kv KV) Bools(key string, defaultValue ...[]bool) []bool {
	val, _ := keyValue(kv, key, append(defaultValue, []bool{false})...)
	return val
}

func (kv KV) Len() int {
	return len(kv)
}

func (kv KV) Keys() iter.Seq[string] {
	return maps.Keys(kv)
}

func (kv KV) Value(key string) any {
	return kv[key]
}

func (ModelParameters) KV(t *Tokenizer) KV {
	kv := KV{
		"general.file_type":            uint32(1),
		"general.quantization_version": uint32(2),
		"tokenizer.ggml.pre":           t.Pre,
		"tokenizer.ggml.model":         t.Vocabulary.Model,
		"tokenizer.ggml.tokens":        t.Vocabulary.Tokens,
		"tokenizer.ggml.scores":        t.Vocabulary.Scores,
		"tokenizer.ggml.token_type":    t.Vocabulary.Types,
	}

	if len(t.Merges) > 0 {
		kv["tokenizer.ggml.merges"] = t.Merges
	}

	if t.Template != "" {
		kv["tokenizer.chat_template"] = t.Template
	}

	for _, sv := range t.SpecialVocabulary {
		kv[fmt.Sprintf("tokenizer.ggml.add_%s_token", sv.Key())] = sv.AddToken
		kv[fmt.Sprintf("tokenizer.ggml.%s_token_id", sv.Key())] = uint32(sv.ID)
		if len(sv.IDs) > 0 {
			kv[fmt.Sprintf("tokenizer.ggml.%s_token_ids", sv.Key())] = sv.IDs
		}
	}

	return kv
}

func (p AdapterParameters) KV() KV {
	var alpha float32
	if p.LoraParameters.Alpha == 0 {
		alpha = float32(p.Alpha)
	} else {
		alpha = p.LoraParameters.Alpha
	}

	kv := KV{
		"adapter.lora.alpha": alpha,
		"adapter.type":       "lora",
		"general.file_type":  uint32(1),
		"general.type":       "adapter",
		"general.version":    "v0.2",
	}

	return kv
}

func (ModelParameters) specialTokenTypes() []string {
	return []string{
		"bos", "eos", "unk", "sep", "pad", "cls", "mask",
	}
}

type ModelKV interface {
	// KV maps parameters to LLM key-values
	KV(*Tokenizer) KV
}

type ModelConverter interface {
	ModelKV

	// Tensors maps input tensors to LLM tensors. Model specific modifications can be done here.
	Tensors([]Tensor) []*ggml.Tensor
	// Replacements returns a list of string pairs to replace in tensor names.
	// See [strings.Replacer](https://pkg.go.dev/strings#Replacer) for details
	Replacements() []string

	// specialTokenTypes returns any special token types the model uses
	specialTokenTypes() []string
}

type moreParser interface {
	parseMore(fs.FS) error
}

type AdapterConverter interface {
	// KV maps parameters to LLM key-values
	KV(ofs.Config) KV
	// Tensors maps input tensors to LLM tensors. Adapter specific modifications can be done here.
	Tensors([]Tensor) []*ggml.Tensor
	// Replacements returns a list of string pairs to replace in tensor names.
	// See [strings.Replacer](https://pkg.go.dev/strings#Replacer) for details
	Replacements() []string
}

func ConvertAdapter(fsys fs.FS, f *os.File, baseKV ofs.Config) error {
	bts, err := fs.ReadFile(fsys, "adapter_config.json")
	if err != nil {
		return err
	}

	var p AdapterParameters
	if err := json.Unmarshal(bts, &p); err != nil {
		return err
	}

	arch := baseKV.Architecture()
	if arch == "" {
		return errors.New("architecture not set for the base model")
	}

	var conv AdapterConverter
	switch arch {
	case "llama":
		conv = &llamaAdapter{}
	case "gemma2":
		conv = &gemma2Adapter{}
	default:
		return errors.New("unsupported architecture")
	}

	ts, err := parseTensors(fsys, strings.NewReplacer(conv.Replacements()...))
	if err != nil {
		return err
	}

	if err := json.Unmarshal(bts, conv); err != nil {
		return err
	}

	return writeFile(f, conv.KV(baseKV), conv.Tensors(ts))
}

func LoadModelMetadata(fsys fs.FS) (ModelKV, *Tokenizer, error) {
	bts, err := fs.ReadFile(fsys, "config.json")
	if err != nil {
		return nil, nil, err
	}

	var p ModelParameters
	if err := json.Unmarshal(bts, &p); err != nil {
		return nil, nil, err
	}

	if len(p.Architectures) < 1 {
		return nil, nil, errors.New("unknown architecture")
	}

	var conv ModelConverter
	switch p.Architectures[0] {
	case "LlamaForCausalLM":
		conv = &llamaModel{}
	case "MllamaForConditionalGeneration":
		conv = &mllamaModel{}
	case "Llama4ForConditionalGeneration":
		conv = &llama4Model{}
	case "Mistral3ForConditionalGeneration":
		conv = &mistral3Model{}
	case "Ministral3ForCausalLM":
		conv = &mistral3CausalModel{}
	case "MixtralForCausalLM":
		conv = &mixtralModel{}
	case "GemmaForCausalLM":
		conv = &gemmaModel{}
	case "Gemma2ForCausalLM":
		conv = &gemma2Model{}
	case "Gemma3ForCausalLM", "Gemma3ForConditionalGeneration":
		conv = &gemma3Model{Architecture: p.Architectures[0]}
	case "Gemma3nForConditionalGeneration":
		conv = &gemma3nModel{}
	case "Phi3ForCausalLM":
		conv = &phi3Model{}
	case "Qwen2ForCausalLM":
		conv = &qwen2Model{}
	case "Qwen2_5_VLForConditionalGeneration":
		conv = &qwen25VLModel{}
	case "Qwen3VLForConditionalGeneration", "Qwen3VLMoeForConditionalGeneration":
		conv = &qwen3VLModel{}
	case "Olmo3ForCausalLM":
		conv = &olmoModel{}
	case "BertModel":
		conv = &bertModel{}
	case "NomicBertModel", "NomicBertMoEModel":
		conv = &nomicbertModel{}
	case "CohereForCausalLM":
		conv = &commandrModel{}
	case "GptOssForCausalLM":
		conv = &gptossModel{}
	case "DeepseekOCRForCausalLM":
		conv = &deepseekocr{}
	case "DeepseekV3ForCausalLM":
		conv = &deepseek2Model{}
	default:
		return nil, nil, fmt.Errorf("unsupported architecture %q", p.Architectures[0])
	}

	if err := json.Unmarshal(bts, conv); err != nil {
		return nil, nil, err
	}

	if t, ok := conv.(moreParser); ok {
		if err := t.parseMore(fsys); err != nil {
			return nil, nil, err
		}
	}

	t, err := parseTokenizer(fsys, conv.specialTokenTypes())
	if err != nil {
		return nil, nil, err
	}

	vocabSize := int(cmp.Or(p.VocabSize, p.TextModel.VocabSize))

	switch {
	case vocabSize == 0:
		slog.Debug("vocabulary size was not explicitly set by the model", "default size", len(t.Vocabulary.Tokens))
	case vocabSize > len(t.Vocabulary.Tokens):
		slog.Debug("vocabulary is smaller than expected, padding with dummy tokens", "expect", vocabSize, "actual", len(t.Vocabulary.Tokens))
		for i := range vocabSize - len(t.Vocabulary.Tokens) {
			t.Vocabulary.Tokens = append(t.Vocabulary.Tokens, fmt.Sprintf("[PAD%d]", i))
			t.Vocabulary.Scores = append(t.Vocabulary.Scores, -1)
			t.Vocabulary.Types = append(t.Vocabulary.Types, tokenTypeUserDefined)
		}
	case vocabSize < len(t.Vocabulary.Tokens):
		slog.Debug("vocabulary is larger than expected", "want", vocabSize, "got", len(t.Vocabulary.Tokens))
		p.VocabSize = uint32(len(t.Vocabulary.Tokens))
		p.TextModel.VocabSize = uint32(len(t.Vocabulary.Tokens))
	default:
		slog.Debug("vocabulary", "size", len(t.Vocabulary.Tokens))
	}
	return conv, t, nil
}

// Convert writes an Ollama compatible model to the provided io.WriteSeeker based on configurations
// and files it finds in the input path.
// Supported input model formats include safetensors.
// Supported input tokenizers files include tokenizer.json (preferred) and tokenizer.model.
func ConvertModel(fsys fs.FS, f *os.File) error {
	kv, t, err := LoadModelMetadata(fsys)
	if err != nil {
		return err
	}
	conv := kv.(ModelConverter)

	ts, err := parseTensors(fsys, strings.NewReplacer(conv.Replacements()...))
	if err != nil {
		return err
	}

	return writeFile(f, conv.KV(t), conv.Tensors(ts))
}

func writeFile(f *os.File, kv KV, ts []*ggml.Tensor) error {
	for i := range ts {
		ts[i].Shape = slices.Clone(ts[i].Shape)
		slices.Reverse(ts[i].Shape)
	}
	return ggml.WriteGGUF(f, kv, ts)
}
