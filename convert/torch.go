package convert

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/nlpodyssey/gopickle/pytorch"
	"github.com/nlpodyssey/gopickle/types"
	"github.com/x448/float16"

	"github.com/ollama/ollama/llm"
)

type torchWriterTo struct {
	t *llm.Tensor

	params *Params
	bo     ByteOrder

	storage pytorch.StorageInterface
	handler func(w io.Writer, r torchWriterTo) error
}

type TorchFormat struct{}

func (tf *TorchFormat) GetTensors(dirpath string, params *Params) ([]llm.Tensor, error) {
	slog.Debug("getting torch tensors")

	//files, err := filepath.Glob(filepath.Join(dirpath, "pytorch_model-*.bin"))
	files, err := filepath.Glob(filepath.Join(dirpath, "consolidatedr.*.pth"))
	if err != nil {
		slog.Error("didn't find any torch files")
		return nil, err
	}

	var offset uint64

	var tensors []llm.Tensor
	for _, fn := range files {
		m, err := pytorch.Load(fn)
		if err != nil {
			slog.Error(fmt.Sprintf("error unpickling: %q", err))
			return []llm.Tensor{}, err
		}

		for _, k := range m.(*types.Dict).Keys() {
			if strings.HasSuffix(k.(string), "self_attn.rotary_emb.inv_freq") {
				continue
			}

			t, _ := m.(*types.Dict).Get(k)
			tshape := t.(*pytorch.Tensor).Size

			var size uint64
			var kind uint32
			switch len(tshape) {
			case 0:
				continue
			case 1:
				// convert to float32
				kind = 0
				size = uint64(tshape[0] * 4)
			case 2:
				// convert to float16
				kind = 1
				size = uint64(tshape[0] * tshape[1] * 2)
			}

			ggufName, err := tf.GetLayerName(k.(string))
			if err != nil {
				slog.Error(err.Error())
				return nil, err
			}
			slog.Debug(fmt.Sprintf("finding name for '%s' -> '%s'", k.(string), ggufName))

			shape := []uint64{0, 0, 0, 0}
			for i := range tshape {
				shape[i] = uint64(tshape[i])
			}

			tensor := llm.Tensor{
				Name:   ggufName,
				Kind:   kind,
				Offset: offset, // calculate the offset
				Shape:  shape[:],
			}

			tensor.WriterTo = torchWriterTo{
				t:       &tensor,
				params:  params,
				bo:      params.ByteOrder,
				storage: t.(*pytorch.Tensor).Source,
			}

			tensors = append(tensors, tensor)
			offset += size
		}
	}

	return tensors, nil

}

func getAltParams(dirpath string) (*Params, error) {
	f, err := os.Open(filepath.Join(dirpath, "params.json"))
	if err != nil {
		slog.Error("no params.json")
		return nil, err
	}
	defer f.Close()

	type TorchParams struct {
		HiddenSize     int     `json:"dim"`
		AttentionHeads int     `json:"n_heads"`
		KeyValHeads    int     `json:"n_kv_heads"`
		HiddenLayers   int     `json:"n_layers"`
		RopeTheta      float64 `json:"rope_theta"`
		NormEPS        float64 `json:"norm_eps"`
	}

	var tparams TorchParams

	d := json.NewDecoder(f)
	err = d.Decode(&tparams)
	if err != nil {
		return nil, err
	}

	params := &Params{
		Architectures:  []string{"LlamaForCausalLM"},
		HiddenSize:     tparams.HiddenSize,
		AttentionHeads: tparams.AttentionHeads,
		KeyValHeads:    tparams.KeyValHeads,
		HiddenLayers:   tparams.HiddenLayers,
		NormEPS:        tparams.NormEPS,
	}

	switch {
	case tparams.RopeTheta == 1000000:
		// Codellama
		params.ContextSize = 16384
	case tparams.NormEPS == 1e-06:
		// llama2
		slog.Debug("Found llama2 - setting context size to 4096")
		params.ContextSize = 4096
	default:
		params.ContextSize = 2048
	}

	params.ByteOrder = binary.LittleEndian
	return params, nil
}

func (m *TorchFormat) GetParams(dirpath string) (*Params, error) {
	f, err := os.Open(filepath.Join(dirpath, "config.json"))
	if err != nil {
		if os.IsNotExist(err) {
			// try params.json instead
			return getAltParams(dirpath)
		} else {
			return nil, err
		}
	}

	var params Params
	d := json.NewDecoder(f)
	err = d.Decode(&params)
	if err != nil {
		return nil, err
	}

	params.ByteOrder = binary.LittleEndian
	return &params, nil
}

func (m *TorchFormat) GetLayerName(n string) (string, error) {
	directMap := map[string]string{
		"tok_embeddings.weight":     "token_embd.weight",
		"output.weight":             "output.weight",
		"norm.weight":               "output_norm.weight",
		"rope.freqs":                "rope_freqs.weight",
		"model.embed_tokens.weight": "token_embd.weight",
		"lm_head.weight":            "output.weight",
		"model.norm.weight":         "output_norm.weight",
	}

	lMap := map[string]string{
		"layers.(\\d+).attention_norm.weight":                 "blk.$1.attn_norm.weight",
		"layers.(\\d+).attention_output_norm.weight":          "blk.$1.attn_norm.weight",
		"layers.(\\d+).feed_forward.w2.weight":                "blk.$1.ffn_down.weight",
		"layers.(\\d+).feed_forward.w1.weight":                "blk.$1.ffn_gate.weight",
		"layers.(\\d+).feed_forward.w3.weight":                "blk.$1.ffn_up.weight",
		"layers.(\\d+).ffn_norm.weight":                       "blk.$1.ffn_norm.weight",
		"layers.(\\d+).attention.wk.weight":                   "blk.$1.attn_k.weight",
		"layers.(\\d+).attention.wo.weight":                   "blk.$1.attn_output.weight",
		"layers.(\\d+).attention.wq.weight":                   "blk.$1.attn_q.weight",
		"layers.(\\d+).attention.wv.weight":                   "blk.$1.attn_v.weight",
		"model.layers.(\\d+).input_layernorm.weight":          "blk.$1.attn_norm.weight",
		"model.layers.(\\d+).mlp.down_proj.weight":            "blk.$1.ffn_down.weight",
		"model.layers.(\\d+).mlp.gate_proj.weight":            "blk.$1.ffn_gate.weight",
		"model.layers.(\\d+).mlp.up_proj.weight":              "blk.$1.ffn_up.weight",
		"model.layers.(\\d+).post_attention_layernorm.weight": "blk.$1.ffn_norm.weight",
		"model.layers.(\\d+).self_attn.k_proj.weight":         "blk.$1.attn_k.weight",
		"model.layers.(\\d+).self_attn.o_proj.weight":         "blk.$1.attn_output.weight",
		"model.layers.(\\d+).self_attn.q_proj.weight":         "blk.$1.attn_q.weight",
		"model.layers.(\\d+).self_attn.v_proj.weight":         "blk.$1.attn_v.weight",
	}

	v, ok := directMap[n]
	if ok {
		return v, nil
	}

	// quick hack to rename the layers to gguf format
	for k, v := range lMap {
		re := regexp.MustCompile(k)
		newName := re.ReplaceAllString(n, v)
		if newName != n {
			return newName, nil
		}
	}

	return "", fmt.Errorf("couldn't find a layer name for '%s'", n)
}

func (r torchWriterTo) WriteTo(w io.Writer) (n int64, err error) {
	// use the handler if one is present
	if r.handler != nil {
		return 0, r.handler(w, r)
	}

	switch r.storage.(type) {
	case *pytorch.FloatStorage:
		slog.Warn(fmt.Sprintf("unexpected storage found for layer '%s'; skipping", r.t.Name))
		return 0, nil
	case *pytorch.HalfStorage:
		switch r.t.Kind {
		case 0:
			data := r.storage.(*pytorch.HalfStorage).Data
			slog.Debug(fmt.Sprintf("%35s F32 (%d)", r.t.Name, len(data)))
			if err := binary.Write(w, r.bo, data); err != nil {
				return 0, err
			}
		case 1:
			data := r.storage.(*pytorch.HalfStorage).Data
			tData := make([]uint16, len(data))
			for cnt, v := range data {
				tData[cnt] = uint16(float16.Fromfloat32(v))
			}
			slog.Debug(fmt.Sprintf("%35s F16 (%d)", r.t.Name, len(tData)))
			if err := binary.Write(w, r.bo, tData); err != nil {
				return 0, err
			}
		}
	}

	return 0, nil
}

func (m *TorchFormat) GetModelArch(name, dirPath string, params *Params) (ModelArch, error) {
	switch len(params.Architectures) {
	case 0:
		return nil, fmt.Errorf("No architecture specified to convert")
	case 1:
		switch params.Architectures[0] {
		case "LlamaForCausalLM":
			return &LlamaModel{
				ModelData{
					Name:   name,
					Path:   dirPath,
					Params: params,
					Format: m,
				},
			}, nil
		default:
			return nil, fmt.Errorf("Models based on '%s' are not yet supported", params.Architectures[0])
		}
	}

	return nil, fmt.Errorf("Unknown error")
}
