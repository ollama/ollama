package convert

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

func TestHybridPatternUnmarshal(t *testing.T) {
	t.Run("string", func(t *testing.T) {
		var p hybridPattern
		if err := json.Unmarshal([]byte(`"MEM*"`), &p); err != nil {
			t.Fatal(err)
		}
		if got, want := string(p), "MEM*"; got != want {
			t.Fatalf("unexpected pattern: got %q want %q", got, want)
		}
	})

	t.Run("array", func(t *testing.T) {
		var p hybridPattern
		if err := json.Unmarshal([]byte(`["M","E","M","*"]`), &p); err != nil {
			t.Fatal(err)
		}
		if got, want := string(p), "MEM*"; got != want {
			t.Fatalf("unexpected pattern: got %q want %q", got, want)
		}
	})
}

func TestNemotronHLayerArrays(t *testing.T) {
	m := &nemotronHModel{
		NumHiddenLayers:       5,
		NumAttentionHeads:     32,
		NumKeyValueHeads:      8,
		HybridOverridePattern: "MEM*E",
		NRoutedExperts:        128,
		NumExpertsPerTok:      6,
		MoEIntermediateSize:   1856,
	}

	headsKV, ffn, err := m.layerArrays()
	if err != nil {
		t.Fatal(err)
	}

	if got, want := headsKV, []uint32{0, 0, 0, 8, 0}; !slices.Equal(got, want) {
		t.Fatalf("unexpected head_count_kv: got %v want %v", got, want)
	}
	if got, want := ffn, []uint32{0, 1856, 0, 0, 1856}; !slices.Equal(got, want) {
		t.Fatalf("unexpected feed_forward_length: got %v want %v", got, want)
	}
}

func TestNemotronHKV(t *testing.T) {
	m := &nemotronHModel{
		MaxPositionEmbeddings:       1048576,
		HiddenSize:                  2688,
		NumHiddenLayers:             5,
		NumAttentionHeads:           32,
		NumKeyValueHeads:            2,
		HeadDim:                     128,
		LayerNormEpsilon:            1e-5,
		RopeTheta:                   10000,
		PartialRotaryFactor:         0.5,
		ConvKernel:                  4,
		SSMStateSize:                128,
		MambaNumHeads:               64,
		MambaHeadDim:                64,
		NGroups:                     8,
		HybridOverridePattern:       "MEM*E",
		NRoutedExperts:              128,
		NSharedExperts:              1,
		NumExpertsPerTok:            6,
		MoEIntermediateSize:         1856,
		MoESharedExpertIntermediate: 3712,
		NormTopKProb:                true,
		RoutedScalingFactor:         2.5,
	}
	if err := m.parseMore(nil); err != nil {
		t.Fatal(err)
	}

	kv := m.KV(&Tokenizer{Vocabulary: &Vocabulary{}})
	if got, want := kv["general.architecture"], "nemotron_h_moe"; got != want {
		t.Fatalf("unexpected architecture: got %v want %v", got, want)
	}

	headCountKV, ok := kv["attention.head_count_kv"].([]uint32)
	if !ok {
		t.Fatalf("attention.head_count_kv has unexpected type: %T", kv["attention.head_count_kv"])
	}
	if got, want := headCountKV, []uint32{0, 0, 0, 2, 0}; !slices.Equal(got, want) {
		t.Fatalf("unexpected attention.head_count_kv: got %v want %v", got, want)
	}

	ffnLength, ok := kv["feed_forward_length"].([]uint32)
	if !ok {
		t.Fatalf("feed_forward_length has unexpected type: %T", kv["feed_forward_length"])
	}
	if got, want := ffnLength, []uint32{0, 1856, 0, 0, 1856}; !slices.Equal(got, want) {
		t.Fatalf("unexpected feed_forward_length: got %v want %v", got, want)
	}
}

func TestNemotronHTensorsTransforms(t *testing.T) {
	m := &nemotronHModel{NGroups: 8}
	in := []Tensor{
		&fakeTensor{
			name:  "blk.0.ssm_a",
			shape: []uint64{4},
			data:  []float32{0, 1, 2, 3},
		},
		&fakeTensor{
			name:  "blk.0.ssm_d",
			shape: []uint64{4},
			data:  []float32{0, 1, 2, 3},
		},
		&fakeTensor{
			name:  "blk.0.ssm_norm.weight",
			shape: []uint64{16},
			data:  make([]float32, 16),
		},
		&fakeTensor{
			name:  "blk.0.ssm_conv1d.weight",
			shape: []uint64{10, 1, 4},
			data:  make([]float32, 40),
		},
	}

	out := m.Tensors(in)
	if len(out) != len(in) {
		t.Fatalf("unexpected output tensor count: got %d want %d", len(out), len(in))
	}

	got := map[string]struct {
		shape  []uint64
		writer io.WriterTo
	}{}
	for _, t := range out {
		got[t.Name] = struct {
			shape  []uint64
			writer io.WriterTo
		}{shape: t.Shape, writer: t.WriterTo}
	}

	if shape := got["blk.0.ssm_a"].shape; !slices.Equal(shape, []uint64{4, 1}) {
		t.Fatalf("unexpected ssm_a shape: %v", shape)
	}
	if shape := got["blk.0.ssm_d"].shape; !slices.Equal(shape, []uint64{4, 1}) {
		t.Fatalf("unexpected ssm_d shape: %v", shape)
	}
	if shape := got["blk.0.ssm_norm.weight"].shape; !slices.Equal(shape, []uint64{8, 2}) {
		t.Fatalf("unexpected ssm_norm shape: %v", shape)
	}
	if shape := got["blk.0.ssm_conv1d.weight"].shape; !slices.Equal(shape, []uint64{10, 4}) {
		t.Fatalf("unexpected ssm_conv1d shape: %v", shape)
	}

	var b bytes.Buffer
	if _, err := got["blk.0.ssm_a"].writer.WriteTo(&b); err != nil {
		t.Fatal(err)
	}
	values := make([]float32, 4)
	if err := binary.Read(&b, binary.LittleEndian, &values); err != nil {
		t.Fatal(err)
	}
	// 0 -> -exp(0) == -1
	if values[0] != -1 {
		t.Fatalf("unexpected transformed ssm_a[0]: got %v want -1", values[0])
	}
}

func TestNemotronHLoadModelMetadata(t *testing.T) {
	tempDir := t.TempDir()

	config := `{
		"architectures": ["NemotronHForCausalLM"],
		"model_type": "nemotron_h",
		"num_hidden_layers": 4,
		"hidden_size": 512,
		"max_position_embeddings": 32768,
		"num_attention_heads": 8,
		"num_key_value_heads": 2,
		"head_dim": 64,
		"layer_norm_epsilon": 1e-5,
		"conv_kernel": 4,
		"ssm_state_size": 128,
		"mamba_num_heads": 16,
		"mamba_head_dim": 32,
		"n_groups": 8,
		"hybrid_override_pattern": "ME*M",
		"n_routed_experts": 16,
		"num_experts_per_tok": 4,
		"moe_intermediate_size": 256
	}`

	if err := os.WriteFile(filepath.Join(tempDir, "config.json"), []byte(config), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(tempDir, "tokenizer.json"), []byte(`{}`), 0o644); err != nil {
		t.Fatal(err)
	}

	kv, _, err := LoadModelMetadata(os.DirFS(tempDir))
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := kv.(*nemotronHModel); !ok {
		t.Fatalf("unexpected converter type: %T", kv)
	}
}

func TestNemotronHNanoVLLoadModelMetadata(t *testing.T) {
	tempDir := t.TempDir()

	config := `{
		"architectures": ["NemotronH_Nano_VL_V2"],
		"model_type": "NemotronH_Nano_VL_V2",
		"max_sequence_length": 131072,
		"force_image_size": 512,
		"downsample_ratio": 0.5,
		"patch_size": 16,
		"use_thumbnail": true,
		"img_context_token_id": 18,
		"img_context_token": "<image>",
		"img_start_token": "<img>",
		"img_end_token": "</img>",
		"sound_context_token_id": 27,
		"sound_context_token": "<so_embedding>",
		"vit_hidden_size": 1280,
		"projector_hidden_size": 20480,
		"norm_mean": [0.48145466, 0.4578275, 0.40821073],
		"norm_std": [0.26862954, 0.26130258, 0.27577711],
		"vision_config": {
			"version": "radio_v2.5-h",
			"patch_size": 16,
			"max_resolution": 2048,
			"separate_video_embedder": true
		},
		"sound_config": {
			"model_type": "parakeet",
			"hidden_size": 1024,
			"num_attention_heads": 8,
			"num_hidden_layers": 24,
			"intermediate_size": 4096,
			"conv_kernel_size": 9,
			"subsampling_conv_channels": 256,
			"subsampling_conv_kernel_size": 3,
			"subsampling_conv_stride": 2,
			"subsampling_factor": 8,
			"num_mel_bins": 128,
			"projection_hidden_size": 4096,
			"sampling_rate": 16000
		},
		"llm_config": {
			"architectures": ["NemotronHForCausalLM"],
			"model_type": "nemotron_h",
			"num_hidden_layers": 4,
			"hidden_size": 512,
			"max_position_embeddings": 262144,
			"num_attention_heads": 8,
			"num_key_value_heads": 2,
			"head_dim": 64,
			"layer_norm_epsilon": 1e-5,
			"conv_kernel": 4,
			"ssm_state_size": 128,
			"mamba_num_heads": 16,
			"mamba_head_dim": 32,
			"n_groups": 8,
			"hybrid_override_pattern": "ME*M",
			"n_routed_experts": 16,
			"num_experts_per_tok": 4,
			"moe_intermediate_size": 256
		}
	}`

	if err := os.WriteFile(filepath.Join(tempDir, "config.json"), []byte(config), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(tempDir, "preprocessor_config.json"), []byte(`{
		"image_size": 512,
		"patch_size": 16,
		"downsample_ratio": 0.5,
		"max_num_tiles": 12,
		"use_thumbnail": true,
		"norm_mean": [0.48145466, 0.4578275, 0.40821073],
		"norm_std": [0.26862954, 0.26130258, 0.27577711]
	}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(tempDir, "tokenizer.json"), []byte(`{}`), 0o644); err != nil {
		t.Fatal(err)
	}

	conv, tokenizer, err := LoadModelMetadata(os.DirFS(tempDir))
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := conv.(*nemotronHNanoVLModel); !ok {
		t.Fatalf("unexpected converter type: %T", conv)
	}

	kv := conv.KV(tokenizer)
	if got, want := kv["general.architecture"], "nemotron_h_omni"; got != want {
		t.Fatalf("unexpected architecture: got %v want %v", got, want)
	}
	if got, want := kv["context_length"], uint32(131072); got != want {
		t.Fatalf("unexpected context length: got %v want %v", got, want)
	}
	if got, want := kv["vision.block_count"], uint32(32); got != want {
		t.Fatalf("unexpected vision block count: got %v want %v", got, want)
	}
	if got, want := kv["vision.image_size"], uint32(512); got != want {
		t.Fatalf("unexpected vision image size: got %v want %v", got, want)
	}
	if got, want := kv["vision.projector.scale_factor"], uint32(2); got != want {
		t.Fatalf("unexpected projector scale factor: got %v want %v", got, want)
	}
	if got, want := kv["audio.block_count"], uint32(24); got != want {
		t.Fatalf("unexpected audio block count: got %v want %v", got, want)
	}
	if got, want := kv["audio.sound_token_id"], uint32(27); got != want {
		t.Fatalf("unexpected audio token id: got %v want %v", got, want)
	}
	if got, want := kv["audio.subsampling_factor"], uint32(8); got != want {
		t.Fatalf("unexpected audio subsampling factor: got %v want %v", got, want)
	}
}

func TestNemotronHNanoOmniReasoningV3LoadModelMetadata(t *testing.T) {
	tempDir := t.TempDir()

	config := `{
		"architectures": ["NemotronH_Nano_Omni_Reasoning_V3"],
		"model_type": "NemotronH_Nano_Omni_Reasoning_V3",
		"max_sequence_length": 131072,
		"force_image_size": 512,
		"downsample_ratio": 0.5,
		"patch_size": 16,
		"img_context_token_id": 18,
		"img_context_token": "<image>",
		"img_start_token": "<img>",
		"img_end_token": "</img>",
		"sound_context_token_id": 27,
		"sound_context_token": "<so_embedding>",
		"vit_hidden_size": 1280,
		"projector_hidden_size": 4096,
		"vision_config": {
			"version": "radio_v2.5-h",
			"patch_size": 16,
			"min_num_patches": 1024,
			"max_num_patches": 13312,
			"args": {
				"min_num_patches": 1024,
				"max_num_patches": 13312
			}
		},
		"sound_config": {
			"model_type": "parakeet",
			"hidden_size": 1024,
			"num_attention_heads": 8,
			"num_hidden_layers": 24,
			"intermediate_size": 4096,
			"conv_kernel_size": 9,
			"subsampling_conv_channels": 256,
			"subsampling_conv_kernel_size": 3,
			"subsampling_conv_stride": 2,
			"subsampling_factor": 8,
			"num_mel_bins": 128,
			"projection_hidden_size": 4096,
			"sampling_rate": 16000
		},
		"llm_config": {
			"architectures": ["NemotronHForCausalLM"],
			"model_type": "nemotron_h",
			"num_hidden_layers": 4,
			"hidden_size": 512,
			"max_position_embeddings": 262144,
			"num_attention_heads": 8,
			"num_key_value_heads": 2,
			"head_dim": 64,
			"layer_norm_epsilon": 1e-5,
			"conv_kernel": 4,
			"ssm_state_size": 128,
			"mamba_num_heads": 16,
			"mamba_head_dim": 32,
			"n_groups": 8,
			"hybrid_override_pattern": "ME*M",
			"n_routed_experts": 16,
			"num_experts_per_tok": 4,
			"moe_intermediate_size": 256
		}
	}`

	if err := os.WriteFile(filepath.Join(tempDir, "config.json"), []byte(config), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(tempDir, "tokenizer.json"), []byte(`{}`), 0o644); err != nil {
		t.Fatal(err)
	}

	conv, tokenizer, err := LoadModelMetadata(os.DirFS(tempDir))
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := conv.(*nemotronHNanoVLModel); !ok {
		t.Fatalf("unexpected converter type: %T", conv)
	}

	kv := conv.KV(tokenizer)
	if got, want := kv["general.architecture"], "nemotron_h_omni"; got != want {
		t.Fatalf("unexpected architecture: got %v want %v", got, want)
	}
	if got, want := kv["vision.block_count"], uint32(32); got != want {
		t.Fatalf("unexpected vision block count: got %v want %v", got, want)
	}
	if got, want := kv["vision.min_num_patches"], uint32(1024); got != want {
		t.Fatalf("unexpected vision min patches: got %v want %v", got, want)
	}
	if got, want := kv["vision.max_num_patches"], uint32(13312); got != want {
		t.Fatalf("unexpected vision max patches: got %v want %v", got, want)
	}
	if got, want := kv["audio.block_count"], uint32(24); got != want {
		t.Fatalf("unexpected audio block count: got %v want %v", got, want)
	}
	if got, want := kv["audio.sound_token_id"], uint32(27); got != want {
		t.Fatalf("unexpected audio token id: got %v want %v", got, want)
	}
}

func TestNemotronHNanoVLTensorsRetainVisionAndAudio(t *testing.T) {
	m := &nemotronHNanoVLModel{
		LLMConfig: nemotronHModel{NGroups: 8},
	}

	in := []Tensor{
		&fakeTensor{
			name:  "blk.0.ssm_a",
			shape: []uint64{4},
			data:  []float32{0, 1, 2, 3},
		},
		&fakeTensor{name: "v.blk.0.attn_qkv.weight", shape: []uint64{3840, 1280}},
		&fakeTensor{name: "v.position_embd", shape: []uint64{1, 16384, 1280}},
		&fakeTensor{name: "v.cls_embd", shape: []uint64{10, 1280}},
		&fakeTensor{name: "mm.norm.weight", shape: []uint64{5120}},
		&fakeTensor{name: "a.feature_extractor.fb", shape: []uint64{1, 128, 257}},
		&fakeTensor{name: "a.subsampling.dw1.weight", shape: []uint64{256, 1, 3, 3}},
		&fakeTensor{name: "a.blk.0.conv_dw.weight", shape: []uint64{1024, 1, 9}},
		&fakeTensor{name: "a.blk.0.conv_pw1.weight", shape: []uint64{2048, 1024, 1}},
		&fakeTensor{name: "a.blk.0.conv_bn.num_batches_tracked", shape: []uint64{1}},
		&fakeTensor{name: "mm.a.1.weight", shape: []uint64{4096, 1024}},
	}

	out := m.Tensors(in)
	got := map[string][]uint64{}
	for _, tns := range out {
		got[tns.Name] = tns.Shape
	}

	for _, name := range []string{
		"blk.0.ssm_a",
		"v.blk.0.attn_q.weight",
		"v.blk.0.attn_k.weight",
		"v.blk.0.attn_v.weight",
		"v.position_embd",
		"v.cls_embd",
		"mm.norm.weight",
		"a.feature_extractor.fb",
		"a.subsampling.dw1.weight",
		"a.blk.0.conv_dw.weight",
		"a.blk.0.conv_pw1.weight",
		"mm.a.1.weight",
	} {
		if _, ok := got[name]; !ok {
			t.Fatalf("expected tensor %q in output", name)
		}
	}

	if gotShape, want := got["blk.0.ssm_a"], []uint64{4, 1}; !slices.Equal(gotShape, want) {
		t.Fatalf("unexpected ssm_a shape: got %v want %v", gotShape, want)
	}
	if gotShape, want := got["v.position_embd"], []uint64{16384, 1280}; !slices.Equal(gotShape, want) {
		t.Fatalf("unexpected position embedding shape: got %v want %v", gotShape, want)
	}
	if gotShape, want := got["a.blk.0.conv_dw.weight"], []uint64{1024, 9}; !slices.Equal(gotShape, want) {
		t.Fatalf("unexpected audio conv_dw shape: got %v want %v", gotShape, want)
	}
	if gotShape, want := got["a.blk.0.conv_pw1.weight"], []uint64{2048, 1024}; !slices.Equal(gotShape, want) {
		t.Fatalf("unexpected audio conv_pw1 shape: got %v want %v", gotShape, want)
	}
	if _, ok := got["a.blk.0.conv_bn.num_batches_tracked"]; ok {
		t.Fatal("audio batchnorm num_batches_tracked should be omitted")
	}
}

func TestNemotronHNanoVLReplacements(t *testing.T) {
	m := &nemotronHNanoVLModel{}
	r := strings.NewReplacer(m.Replacements()...)

	if got, want := r.Replace("language_model.backbone.layers.1.mixer.fc1_latent_proj.weight"), "blk.1.ffn_latent_in.weight"; got != want {
		t.Fatalf("unexpected fc1 replacement: got %q want %q", got, want)
	}
	if got, want := r.Replace("language_model.lm_head.weight"), "output.weight"; got != want {
		t.Fatalf("unexpected lm_head replacement: got %q want %q", got, want)
	}
	if got, want := r.Replace("vision_model.radio_model.model.blocks.0.attn.qkv.weight"), "v.blk.0.attn_qkv.weight"; got != want {
		t.Fatalf("unexpected vision replacement: got %q want %q", got, want)
	}
	if got, want := r.Replace("mlp1.1.weight"), "mm.1.weight"; got != want {
		t.Fatalf("unexpected projector replacement: got %q want %q", got, want)
	}
	if got, want := r.Replace("sound_encoder.encoder.layers.0.self_attn.q_proj.weight"), "a.blk.0.attn_q.weight"; got != want {
		t.Fatalf("unexpected audio q_proj replacement: got %q want %q", got, want)
	}
	if got, want := r.Replace("sound_encoder.encoder.layers.0.conv.pointwise_conv1.weight"), "a.blk.0.conv_pw1.weight"; got != want {
		t.Fatalf("unexpected audio conv replacement: got %q want %q", got, want)
	}
	if got, want := r.Replace("sound_projection.linear2.weight"), "mm.a.2.weight"; got != want {
		t.Fatalf("unexpected audio projector replacement: got %q want %q", got, want)
	}
}

func TestNemotronHReplacementsLatentProjections(t *testing.T) {
	m := &nemotronHModel{}
	r := strings.NewReplacer(m.Replacements()...)

	if got, want := r.Replace("backbone.layers.1.mixer.fc1_latent_proj.weight"), "blk.1.ffn_latent_in.weight"; got != want {
		t.Fatalf("unexpected fc1 replacement: got %q want %q", got, want)
	}
	if got, want := r.Replace("backbone.layers.1.mixer.fc2_latent_proj.weight"), "blk.1.ffn_latent_out.weight"; got != want {
		t.Fatalf("unexpected fc2 replacement: got %q want %q", got, want)
	}
}
