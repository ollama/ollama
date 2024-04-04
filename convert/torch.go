package convert

import (
	//"encoding/binary"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"regexp"

	"github.com/nlpodyssey/gopickle/pytorch"
	"github.com/nlpodyssey/gopickle/types"

	"github.com/ollama/ollama/llm"
)

type torchWriterTo struct {
	t *llm.Tensor
}

type TorchFormat struct{}

func (m *TorchFormat) GetTensors(dirpath string, params *Params) ([]llm.Tensor, error) {
	slog.Debug("getting torch tensors\n")

	files, err := filepath.Glob(filepath.Join(dirpath, "consolidated.*.pth"))
	if err != nil {
		return nil, err
	}

	var tensors []llm.Tensor
	for _, fn := range files {
		m, err := pytorch.Load(fn)
		if err != nil {
			slog.Error(fmt.Sprintf("error unpickling: %q", err))
			return []llm.Tensor{}, err
		}

		for _, k := range m.(*types.Dict).Keys() {
			slog.Debug(fmt.Sprintf("layer name: %s\n", k))
			t, _ := m.(*types.Dict).Get(k)
			slog.Debug(fmt.Sprintf("tensor = %#v", t.(*pytorch.Tensor)))
			tshape := t.(*pytorch.Tensor).Size

			//var size uint64
			var kind uint32
			switch len(tshape) {
			case 0:
				continue
			case 1:
				// convert to float32
				kind = 0
				//size = uint64(tshape[0] * 4)
			case 2:
				// convert to float16
				kind = 1
				//size = uint64(tshape[0] * tshape[1] * 2)
			}

			shape := []uint64{0, 0, 0, 0}
			for i := range tshape {
				shape[i] = uint64(tshape[i])
			}

			tensor := llm.Tensor{
				Name: k.(string), // replace w/ gguf name
				Kind: kind,
				//Offset: offset, // calculate the offset
				Shape: shape[:],
			}

			tensors = append(tensors, tensor)

		}
		//slog.Debug(fmt.Sprintf("model = %#v\n", m))
	}

	return tensors, nil

}

func (m *TorchFormat) GetParams(dirpath string) (*Params, error) {
	f, err := os.Open(filepath.Join(dirpath, "params.json"))
	if err != nil {
		return nil, err
	}
	defer f.Close()

	type TorchParams struct {
		HiddenSize     int     `json:"dim"`
		AttentionHeads int     `json:"n_heads"`
		KeyValHeads    int     `json:"n_kv_heads"`
		HiddenLayers   int     `json:"n_layers"`
		RopeTheta      int     `json:"rope_theta"`
		NormEPS        float64 `json:"norm_eps"`
	}

	var tparams TorchParams

	d := json.NewDecoder(f)
	err = d.Decode(&tparams)
	if err != nil {
		return nil, err
	}

	params := &Params{
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

	return params, nil
}

func (m *TorchFormat) GetLayerName(n string) (string, error) {
	lMap := map[string]string{
		"tok_embeddings.weight":                "token_embd.weight",
		"layers.(\\d+).attention_norm.weight":  "blk.$1.attn_norm.weight",
		"layers.(\\d+).feed_forward.w2.weight": "blk.$1.ffn_down.weight",
		"layers.(\\d+).feed_forward.w1.weight": "blk.$1.ffn_gate.weight",
		"layers.(\\d+).feed_forward.w3.weight": "blk.$1.ffn_up.weight",
		"layers.(\\d+).ffn_norm.weight":        "blk.$1.ffn_norm.weight",
		"layers.(\\d+).attention.wk.weight":    "blk.$1.attn_k.weight",
		"layers.(\\d+).attention.wo.weight":    "blk.$1.attn_output.weight",
		"layers.(\\d+).attention.wq.weight":    "blk.$1.attn_q.weight",
		"layers.(\\d+).attention.wv.weight":    "blk.$1.attn_v.weight",
		"output.weight":                        "output.weight",
		"norm.weight":                          "output_norm.weight",
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

func (m *TorchFormat) GetModelArch(name, dirPath string, params *Params) (ModelArch, error) {
	return &LlamaModel{
		ModelData{
			Name:   name,
			Path:   dirPath,
			Params: params,
		},
	}, nil
}
