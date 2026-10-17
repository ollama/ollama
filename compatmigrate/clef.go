package compatmigrate

import (
	_ "embed"
	"fmt"
	"math"
	"slices"
	"strings"

	"github.com/ollama/ollama/fs/gguf"
)

type clefMigrator struct{}

func (clefMigrator) NeedsMigration(src *SourceModel) bool {
	return src.GGUF.KeyValue("general.architecture").String() == "qwen35" && src.GGUF.KeyValue("decision.type").String() == "clef"
}

func (clefMigrator) Migrate(src *SourceModel) (*Result, error) {
	if (qwen35Migrator{}).NeedsMigration(src) {
		return nil, fmt.Errorf("Clef requires the published native Qwen3.5 backbone: %w", errUnsupportedFamily)
	}
	if src.Config.Parser != "" || (src.Config.Renderer != "" && src.Config.Renderer != "clef") {
		return nil, fmt.Errorf("Clef has a custom parser or renderer: %w", errUnsupportedFamily)
	}
	read := func(key string) uint64 { return src.GGUF.KeyValue("decision." + key).Uint() }
	hidden, width, heads, ff := read("hidden_size"), read("width"), read("heads"), read("feedforward")
	routing, joint := read("routing_layers"), read("layers")
	if hidden == 0 || hidden > 16384 || width == 0 || width > 4096 || heads == 0 || width%heads != 0 ||
		ff == 0 || ff > 65536 || routing == 0 || routing > 32 || joint == 0 || joint > 32 {
		return nil, fmt.Errorf("invalid Clef head dimensions")
	}

	tensors, err := readAllSourceTensors(src)
	if err != nil {
		return nil, err
	}
	head := map[string]*sourceTensor{}
	result := &Result{ModelKV: outKV{}, PreserveProjector: src.ProjectorGGUF != nil, ClearRenderer: true}
	for _, t := range tensors {
		if strings.HasPrefix(t.name, "clef.") {
			head[strings.TrimPrefix(t.name, "clef.")] = t
		} else {
			// The quantized backbone is already native. Preserve its bytes exactly.
			result.ModelTensors = append(result.ModelTensors, &outTensor{
				Name: t.name, Kind: uint32(t.info.Type), Shape: slices.Clone(t.shape), WriterTo: t.Clone(),
			})
		}
	}
	// Match conversion/clef.py in llama.cpp v0.6.0. Source dimensions are
	// GGUF order (innermost first); fused QKV consists of three contiguous parts.
	copyHead := func(from, to string, shape ...uint64) error {
		t, ok := head[from]
		if !ok || t.info.Type != gguf.TensorTypeF32 || !slices.Equal(t.shape, shape) {
			return fmt.Errorf("Clef tensor %q must be F32 with shape %v", from, shape)
		}
		result.ModelTensors = append(result.ModelTensors, copyTensor(to, t))
		delete(head, from)
		return nil
	}
	copyNorm := func(from, to string, size uint64) error {
		for _, suffix := range []string{".weight", ".bias"} {
			if err := copyHead(from+suffix, to+suffix, size); err != nil {
				return err
			}
		}
		return nil
	}
	copyAttention := func(from, to string) error {
		for _, suffix := range []string{"weight", "bias"} {
			name := from + ".in_proj_" + suffix
			t, ok := head[name]
			shape := []uint64{width, 3 * width}
			if suffix == "bias" {
				shape = []uint64{3 * width}
			}
			if !ok || t.info.Type != gguf.TensorTypeF32 || !slices.Equal(t.shape, shape) {
				return fmt.Errorf("Clef tensor %q must be F32 with shape %v", name, shape)
			}
			shape[len(shape)-1] /= 3
			for i, part := range []string{"q", "k", "v"} {
				writer := t.Clone()
				writer.info.Shape = slices.Clone(shape)
				writer.shape = slices.Clone(shape)
				writer.info.Offset += uint64(i) * uint64(writer.info.NumBytes())
				result.ModelTensors = append(result.ModelTensors, &outTensor{
					Name: to + "_" + part + "." + suffix, Kind: uint32(gguf.TensorTypeF32), Shape: slices.Clone(shape), WriterTo: writer,
				})
			}
			delete(head, name)
		}
		if err := copyHead(from+".out_proj.weight", to+"_o.weight", width, width); err != nil {
			return err
		}
		return copyHead(from+".out_proj.bias", to+"_o.bias", width)
	}
	for _, projection := range []string{"memory", "question", "option_question", "global", "option_context", "option_lexical"} {
		if err := copyHead(projection+"_projection.weight", "decision.proj_"+projection+".weight", hidden, width); err != nil {
			return nil, err
		}
	}
	for _, norm := range []string{"hidden_norm", "option_summary_norm", "field_norm", "option_norm"} {
		size := width
		if norm == "hidden_norm" {
			size = hidden
		}
		if err := copyNorm(norm, "decision."+norm, size); err != nil {
			return nil, err
		}
	}
	// Flash retains the scorer's singleton output dimension; the 27B export
	// squeezes it. Both store the same F32 row vector.
	scorerShape := []uint64{width}
	if t := head["residual_scorer.3.weight"]; t != nil && len(t.shape) == 2 {
		scorerShape = append(scorerShape, 1)
	}
	for _, t := range []struct {
		from, to string
		shape    []uint64
	}{
		{"type_embedding.weight", "token_types.weight", []uint64{width, 3}},
		{"residual_scorer.0.weight", "decision.scorer.weight", []uint64{4 * width, width}},
		{"residual_scorer.0.bias", "decision.scorer.bias", []uint64{width}},
		{"residual_scorer.3.weight", "decision.scorer_out.weight", scorerShape},
		{"residual_scorer.3.bias", "decision.scorer_out.bias", []uint64{1}},
	} {
		if err := copyHead(t.from, t.to, t.shape...); err != nil {
			return nil, err
		}
	}
	for i := range routing + joint {
		to := fmt.Sprintf("dec.blk.%d.", i)
		from := fmt.Sprintf("evidence_layers.%d.", i)
		cross, norms, up, down := "attention", []string{"query_norm", "memory_norm", "feedforward_norm"}, "feedforward.0", "feedforward.3"
		if i >= routing {
			from = fmt.Sprintf("layers.%d.", i-routing)
			cross, norms, up, down = "multihead_attn", []string{"norm2", "norm1", "norm3"}, "linear1", "linear2"
			if err := copyAttention(from+"self_attn", to+"attn"); err != nil {
				return nil, err
			}
		}
		nativeNorms := []string{"cross_attn_norm", "cross_attn_norm_kv", "ffn_norm"}
		if i >= routing {
			nativeNorms[1] = "attn_norm"
		}
		for j, norm := range norms {
			if err := copyNorm(from+norm, to+nativeNorms[j], width); err != nil {
				return nil, err
			}
		}
		if err := copyAttention(from+cross, to+"cross_attn"); err != nil {
			return nil, err
		}
		for _, t := range []struct {
			from, to string
			shape    []uint64
		}{
			{up + ".weight", "ffn_up.weight", []uint64{width, ff}},
			{up + ".bias", "ffn_up.bias", []uint64{ff}},
			{down + ".weight", "ffn_down.weight", []uint64{ff, width}},
			{down + ".bias", "ffn_down.bias", []uint64{width}},
		} {
			if err := copyHead(from+t.from, to+t.to, t.shape...); err != nil {
				return nil, err
			}
		}
	}
	var scales []float32
	for _, name := range []string{"prior_logit_scale", "joint_logit_scale", "residual_gate"} {
		t, ok := head[name]
		if !ok || t.info.Type != gguf.TensorTypeF32 || !slices.Equal(t.shape, []uint64{1}) {
			return nil, fmt.Errorf("invalid Clef scalar %q", name)
		}
		values, err := readSourceTensorFloatData(t)
		if err != nil {
			return nil, err
		}
		v := float64(values[0])
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return nil, fmt.Errorf("non-finite Clef scalar %q", name)
		}
		if name == "residual_gate" {
			v = 1 / (1 + math.Exp(-v))
		} else {
			v = math.Exp(math.Min(v, math.Log(100)))
		}
		scales = append(scales, float32(v))
		delete(head, name)
	}
	if len(head) != 0 {
		return nil, fmt.Errorf("Clef contains unrecognized head tensors")
	}
	result.ModelTensors = append(result.ModelTensors, f32Tensor("decision.scales", []uint64{3}, scales))
	for _, entry := range src.GGUF.KeyValues() {
		if entry.Valid() && !strings.HasPrefix(entry.Key, "qwen35.decision.") {
			key := strings.Replace(entry.Key, "qwen35.", "clef.", 1)
			result.ModelKV[key] = normalizeGGUFValue(entry.Any())
		}
	}
	result.ModelKV["general.architecture"] = "clef"
	result.ModelKV["clef.decision.type"] = "clef"
	result.ModelKV["clef.decision.routing_block_count"] = uint32(routing)
	result.ModelKV["clef.decision.block_count"] = uint32(joint)
	result.ModelKV["clef.decision.head_count"] = uint32(heads)
	result.ModelKV["clef.attention.layer_norm_epsilon"] = float32(1e-5)
	result.ModelKV["tokenizer.chat_template.systemone"] = clefSchemaTemplate
	result.Requires, err = decisionMigrationVersion(src.Config.Requires)
	return result, err
}

// From conversion/clef.py in llama.cpp v0.6.0.
//
//go:embed templates/clef.jinja
var clefSchemaTemplate string
