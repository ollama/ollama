package lfm2

import (
	"strings"
	"testing"

	"github.com/ollama/ollama/fs/ggml"
)

func newTestKV(arch string) ggml.KV {
	return ggml.KV{
		"general.architecture": arch,
		"tokenizer.ggml.model": "gpt2",

		arch + ".block_count":                      uint32(4),
		arch + ".embedding_length":                 uint32(256),
		arch + ".attention.head_count":             uint32(8),
		arch + ".attention.head_count_kv":          uint32(8),
		arch + ".attention.key_length":             uint32(32),
		arch + ".attention.layer_norm_rms_epsilon": float32(1e-5),
		arch + ".rope.scaling.factor":              float32(1),
	}
}

func TestNewSupportsMoE(t *testing.T) {
	kv := newTestKV("lfm2moe")
	kv["lfm2moe.expert_count"] = uint32(16)
	kv["lfm2moe.expert_used_count"] = uint32(4)
	kv["lfm2moe.leading_dense_block_count"] = uint32(2)
	kv["lfm2moe.expert_gating_func"] = uint32(expertGatingFuncSigmoid)

	got, err := New(kv)
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	m, ok := got.(*Model)
	if !ok {
		t.Fatalf("New returned %T, want *Model", got)
	}

	if m.numExperts != 16 || m.numExpertsUsed != 4 {
		t.Fatalf("unexpected MoE options: experts=%d used=%d", m.numExperts, m.numExpertsUsed)
	}

	for i, layer := range m.Layers {
		if i < 2 {
			if _, ok := layer.MLP.(*denseMLP); !ok {
				t.Fatalf("layer %d MLP = %T, want *denseMLP", i, layer.MLP)
			}
		} else {
			if _, ok := layer.MLP.(*sparseMLP); !ok {
				t.Fatalf("layer %d MLP = %T, want *sparseMLP", i, layer.MLP)
			}
		}
	}
}

func TestNewRejectsInvalidMoEConfig(t *testing.T) {
	kv := newTestKV("lfm2moe")
	kv["lfm2moe.expert_count"] = uint32(16)
	kv["lfm2moe.expert_used_count"] = uint32(0)

	_, err := New(kv)
	if err == nil {
		t.Fatal("expected error for invalid MoE configuration")
	}
	if !strings.Contains(err.Error(), "invalid expert_used_count") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestNewDenseStillUsesDenseMLP(t *testing.T) {
	kv := newTestKV("lfm2")

	got, err := New(kv)
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	m, ok := got.(*Model)
	if !ok {
		t.Fatalf("New returned %T, want *Model", got)
	}

	for i, layer := range m.Layers {
		if _, ok := layer.MLP.(*denseMLP); !ok {
			t.Fatalf("layer %d MLP = %T, want *denseMLP", i, layer.MLP)
		}
	}
}
