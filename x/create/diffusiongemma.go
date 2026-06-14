package create

import (
	"strings"

	"github.com/ollama/ollama/x/safetensors"
)

// diffusionGemmaImportTransform adapts a DiffusionGemma checkpoint to the
// canonical Gemma 4 tensor layout the MLX loader expects, then defers to the
// Gemma 4 transform for MoE expert splitting and per-tensor quantization.
//
// DiffusionGemma checkpoints carry two transformer stacks that share weights
// (an "encoder" used for the causal prompt prefill and a "decoder" used for the
// bidirectional canvas denoising) plus a small self-conditioning MLP and a
// vision tower. Mirroring the reference converter (conversion/diffusion_gemma.py
// in llama.cpp PR #24427), we keep the decoder stack as the canonical model.*
// tensors, drop the duplicate encoder stack (keeping only its per-layer output
// scale), map the self-conditioning MLP to self_cond_* names, and drop vision.
type diffusionGemmaImportTransform struct {
	gemma4ImportTransform
}

func newDiffusionGemmaImportTransform(modelDir string, cfg sourceModelConfig) (tensorImportTransform, error) {
	g, err := newGemma4ImportTransform(modelDir, cfg)
	if err != nil {
		return nil, err
	}
	return diffusionGemmaImportTransform{gemma4ImportTransform: g.(gemma4ImportTransform)}, nil
}

func (t diffusionGemmaImportTransform) skipTensor(name string) bool {
	// Drop the entire encoder stack: the vision tower and the encoder per-layer
	// output scale (model.encoder.language_model.layers.N.layer_scalar), which the
	// reference loads but never applies. The decoder stack carries the canonical
	// text weights (including its own layer_scalar) and the self-conditioning MLP.
	if isDiffusionGemmaVisionTensor(name) || strings.Contains(name, "model.encoder.") {
		return true
	}
	return t.gemma4ImportTransform.skipTensor(name)
}

func (t diffusionGemmaImportTransform) transformTensor(td *safetensors.TensorData) ([]*safetensors.TensorData, error) {
	if td == nil {
		return nil, nil
	}

	mapped := remapDiffusionGemmaName(td.Name)
	if mapped == "" {
		return nil, nil
	}
	td.Name = mapped

	// Keep the MoE experts STACKED. DiffusionGemma ships pre-quantized experts as
	// one tensor per projection spanning all experts (experts.gate_up_proj /
	// experts.down_proj, plus companion .scales/.biases). gemma4's transform splits
	// the .weight per-expert but leaves the 3-D .scales/.biases stacked, orphaning
	// them — the loader would then find a per-expert weight with no matching scale
	// and treat the 4-bit weight as float (garbage MoE). The MLX loader reads the
	// stacked quantized experts directly via GatherQMM, so don't split them.
	if strings.Contains(mapped, ".experts.") {
		return []*safetensors.TensorData{td}, nil
	}

	// Defer to the Gemma 4 transform for any other stacked-MoE handling on the
	// now-canonical names.
	return t.gemma4ImportTransform.transformTensor(td)
}

func isDiffusionGemmaVisionTensor(name string) bool {
	return strings.Contains(name, "vision") || strings.Contains(name, "embed_vision")
}

// remapDiffusionGemmaName rewrites a DiffusionGemma decoder tensor name to the
// canonical Gemma 4 name the MLX loader resolves, or returns "" to drop it. The
// encoder stack is dropped earlier by skipTensor.
func remapDiffusionGemmaName(name string) string {
	// Self-conditioning MLP: model.decoder.self_conditioning.<part>.<suffix> ->
	// self_cond_<name>.<suffix>. The <suffix> (.weight / .scales / .biases) MUST
	// be preserved — the MLP is quantized, so the scale/bias companions ride along.
	if i := strings.Index(name, "decoder.self_conditioning."); i >= 0 {
		sub := name[i+len("decoder.self_conditioning."):] // e.g. "gate_proj.scales"
		dot := strings.IndexByte(sub, '.')
		if dot < 0 {
			return ""
		}
		part, suffix := sub[:dot], sub[dot:] // "gate_proj", ".scales"
		switch part {
		case "pre_norm":
			return "self_cond_pre_norm" + suffix
		case "gate_proj":
			return "self_cond_gate" + suffix
		case "up_proj":
			return "self_cond_up" + suffix
		case "down_proj":
			return "self_cond_down" + suffix
		}
		return "" // unknown self-conditioning tensor
	}

	// Decoder stack -> canonical model.* (strip the "decoder." segment).
	if i := strings.Index(name, "model.decoder."); i >= 0 {
		return name[:i] + "model." + name[i+len("model.decoder."):]
	}

	return name
}
