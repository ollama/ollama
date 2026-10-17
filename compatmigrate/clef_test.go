package compatmigrate

import (
	"bytes"
	"context"
	"errors"
	"io"
	"math"
	"slices"
	"testing"
	"testing/synctest"

	"github.com/ollama/ollama/fs/gguf"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

func clefFixture() (outKV, []*outTensor) {
	kv := outKV{
		"general.architecture": "qwen35", "qwen35.decision.type": "clef",
		"qwen35.decision.hidden_size": uint32(4), "qwen35.decision.width": uint32(2),
		"qwen35.decision.heads": uint32(1), "qwen35.decision.feedforward": uint32(8),
		"qwen35.decision.routing_layers": uint32(1), "qwen35.decision.layers": uint32(1),
		"qwen35.rope.dimension_sections": []int32{11, 11, 10, 0},
		"qwen35.context_length":          uint32(4096), "tokenizer.ggml.model": "gpt2",
	}
	tensors := []*outTensor{fixtureTensor("token_embd.weight", gguf.TensorTypeQ8_0, []uint64{32, 2})}
	add := func(name string, shape ...uint64) {
		tensors = append(tensors, f32Tensor("clef."+name, shape, sequence(int(tensorElementCount(shape)))))
	}
	for _, name := range []string{"memory", "question", "option_question", "global", "option_context", "option_lexical"} {
		add(name+"_projection.weight", 4, 2)
	}
	for _, name := range []string{"hidden_norm", "option_summary_norm", "field_norm", "option_norm"} {
		size := uint64(2)
		if name == "hidden_norm" {
			size = 4
		}
		add(name+".weight", size)
		add(name+".bias", size)
	}
	add("type_embedding.weight", 2, 3)
	add("residual_scorer.0.weight", 8, 2)
	add("residual_scorer.0.bias", 2)
	add("residual_scorer.3.weight", 2)
	add("residual_scorer.3.bias", 1)
	tensors = append(tensors, f32Tensor("clef.prior_logit_scale", []uint64{1}, []float32{0}), f32Tensor("clef.joint_logit_scale", []uint64{1}, []float32{10}), f32Tensor("clef.residual_gate", []uint64{1}, []float32{0}))
	for _, block := range []struct {
		prefix           string
		norms, attention []string
		up, down         string
	}{
		{"evidence_layers.0.", []string{"query_norm", "memory_norm", "feedforward_norm"}, []string{"attention"}, "feedforward.0", "feedforward.3"},
		{"layers.0.", []string{"norm1", "norm2", "norm3"}, []string{"self_attn", "multihead_attn"}, "linear1", "linear2"},
	} {
		for _, norm := range block.norms {
			add(block.prefix+norm+".weight", 2)
			add(block.prefix+norm+".bias", 2)
		}
		for _, attention := range block.attention {
			name := block.prefix + attention
			add(name+".in_proj_weight", 2, 6)
			add(name+".in_proj_bias", 6)
			add(name+".out_proj.weight", 2, 2)
			add(name+".out_proj.bias", 2)
		}
		add(block.prefix+block.up+".weight", 2, 8)
		add(block.prefix+block.up+".bias", 8)
		add(block.prefix+block.down+".weight", 8, 2)
		add(block.prefix+block.down+".bias", 2)
	}
	return kv, tensors
}

func TestClefMigrationTensorContract(t *testing.T) {
	kv, tensors := clefFixture()
	source := fixtureSourceModel(t, kv, tensors)
	result, err := (clefMigrator{}).Migrate(source)
	if err != nil {
		t.Fatal(err)
	}
	if result.ModelKV["general.architecture"] != "clef" || result.ModelKV["clef.context_length"] != uint32(4096) || result.ModelKV["qwen35.context_length"] != nil {
		t.Fatal("architecture metadata not migrated")
	}
	if result.ModelKV["clef.decision.block_count"] != uint32(1) || result.ModelKV["clef.decision.routing_block_count"] != uint32(1) || result.ModelKV["tokenizer.chat_template.systemone"] == "" {
		t.Fatal("missing native head metadata")
	}
	outputs := map[string]*outTensor{}
	for _, tensor := range result.ModelTensors {
		outputs[tensor.Name] = tensor
	}
	for _, prefix := range []string{"dec.blk.0.cross_attn", "dec.blk.1.cross_attn", "dec.blk.1.attn"} {
		for i, part := range []string{"q", "k", "v"} {
			for _, suffix := range []string{"weight", "bias"} {
				name := prefix + "_" + part + "." + suffix
				output := outputs[name]
				if output == nil {
					t.Fatalf("missing %s", name)
				}
				n := 4
				shape := []uint64{2, 2}
				if suffix == "bias" {
					n = 2
					shape = []uint64{2}
				}
				var b bytes.Buffer
				if _, err := output.WriteTo(&b); err != nil {
					t.Fatal(err)
				}
				got, err := decodeFloatTensor(gguf.TensorTypeF32, b.Bytes())
				if err != nil {
					t.Fatal(err)
				}
				want := sequence(3 * n)[i*n : (i+1)*n]
				if !slices.Equal(got, want) || !slices.Equal(output.Shape, shape) {
					t.Fatalf("%s: %v shape %v; want %v shape %v", name, got, output.Shape, want, shape)
				}
			}
		}
	}
	var b bytes.Buffer
	_, err = outputs["decision.scales"].WriteTo(&b)
	if err != nil {
		t.Fatal(err)
	}
	scales, err := decodeFloatTensor(gguf.TensorTypeF32, b.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(scales, []float32{1, 100, .5}) {
		t.Fatalf("scales %v", scales)
	}
	before := source.GGUF.TensorInfo("token_embd.weight")
	data, err := io.ReadAll(io.NewSectionReader(source.GGUFData, source.GGUFDataOffset+int64(before.Offset), before.NumBytes()))
	if err != nil {
		t.Fatal(err)
	}
	b.Reset()
	_, err = outputs[before.Name].WriteTo(&b)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(data, b.Bytes()) || outputs[before.Name].Kind != uint32(before.Type) {
		t.Fatal("quantized backbone changed")
	}
}

func TestClefMigrationRejectsUnknownHeads(t *testing.T) {
	for _, mode := range []string{"missing", "unknown", "shape", "dtype", "nonfinite", "dimensions"} {
		t.Run(mode, func(t *testing.T) {
			kv, tensors := clefFixture()
			switch mode {
			case "missing":
				tensors = tensors[:len(tensors)-1]
			case "unknown":
				tensors = append(tensors, f32Tensor("clef.unknown", []uint64{1}, []float32{0}))
			case "shape":
				tensors[1].Shape = []uint64{2, 4}
			case "dtype":
				tensors[1] = fixtureTensor(tensors[1].Name, gguf.TensorTypeF16, tensors[1].Shape)
			case "nonfinite":
				for i, v := range tensors {
					if v.Name == "clef.residual_gate" {
						tensors[i] = f32Tensor(v.Name, []uint64{1}, []float32{float32(math.NaN())})
					}
				}
			case "dimensions":
				kv["qwen35.decision.heads"] = uint32(3)
			}
			if _, err := (clefMigrator{}).Migrate(fixtureSourceModel(t, kv, tensors)); err == nil {
				t.Fatal("invalid source accepted")
			}
		})
	}
}

func TestClefFlashScorerShape(t *testing.T) {
	for _, outputs := range []uint64{1, 2} {
		kv, tensors := clefFixture()
		for i, tensor := range tensors {
			if tensor.Name == "clef.residual_scorer.3.weight" {
				tensors[i] = f32Tensor(tensor.Name, []uint64{2, outputs}, sequence(int(2*outputs)))
			}
		}
		result, err := (clefMigrator{}).Migrate(fixtureSourceModel(t, kv, tensors))
		if outputs != 1 {
			if err == nil {
				t.Fatal("accepted scorer with multiple outputs")
			}
			continue
		}
		if err != nil {
			t.Fatal(err)
		}
		for _, tensor := range result.ModelTensors {
			if tensor.Name == "decision.scorer_out.weight" && !slices.Equal(tensor.Shape, []uint64{2, 1}) {
				t.Fatalf("scorer shape changed: %v", tensor.Shape)
			}
		}
	}
}

func TestClefRequiredMigration(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	kv, tensors := clefFixture()
	name := model.ParseName("clef-migration:test")
	writeSourceManifest(t, name, sourceManifestInput{config: model.ConfigV2{ModelFormat: "gguf", ModelFamily: "qwen35", Capabilities: []string{"decision", "vision"}}, modelKV: kv, modelTensors: tensors, projectorKV: outKV{"general.architecture": "clip"}, projectorTensors: []*outTensor{fixtureTensor("v.weight", gguf.TensorTypeF16, []uint64{2, 2})}})
	before, err := manifest.ReadManifestData(name)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := WaitLocalCompatibilityMigration(t.Context(), name, ""); err != nil {
		t.Fatal(err)
	}
	native, err := manifest.ParseNamedManifestForRunner(name, manifest.RunnerLlamaCPP)
	if err != nil {
		t.Fatal(err)
	}
	src, err := loadSourceModelFromManifest(name, native)
	if err != nil {
		t.Fatal(err)
	}
	defer src.Close()
	if src.GGUF.KeyValue("general.architecture").String() != "clef" || src.ProjectorGGUF == nil {
		t.Fatal("conversion lost head or projector")
	}
	first, err := manifest.ReadManifestData(name)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Equal(before, first) {
		t.Fatal("manifest was not converted")
	}
	if _, err := WaitLocalCompatibilityMigration(t.Context(), name, ""); err != nil {
		t.Fatal(err)
	}
	second, err := manifest.ReadManifestData(name)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(first, second) {
		t.Fatal("second load rewrote manifest")
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := WaitLocalCompatibilityMigration(ctx, name, ""); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation: %v", err)
	}
}

func TestWaitLocalCompatibilityMigrationCancellation(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		name := model.ParseName("clef-migration:wait")
		migration := &localMigration{done: make(chan struct{})}
		migrationInFlight.Store(name.String()+":", migration)
		t.Cleanup(func() { migrationInFlight.Delete(name.String() + ":") })
		ctx, cancel := context.WithCancel(t.Context())
		defer cancel()
		canceled, remaining := make(chan error, 1), make(chan error, 1)
		go func() { _, err := WaitLocalCompatibilityMigration(ctx, name, ""); canceled <- err }()
		go func() { _, err := WaitLocalCompatibilityMigration(t.Context(), name, ""); remaining <- err }()
		synctest.Wait()
		cancel()
		synctest.Wait()
		if err := <-canceled; !errors.Is(err, context.Canceled) {
			t.Fatalf("canceled waiter: %v", err)
		}
		select {
		case err := <-remaining:
			t.Fatalf("canceling one waiter interrupted another: %v", err)
		default:
		}
		want := errors.New("conversion failed")
		migration.err = want
		close(migration.done)
		if err := <-remaining; !errors.Is(err, want) {
			t.Fatalf("remaining waiter: %v; want %v", err, want)
		}
	})
}
