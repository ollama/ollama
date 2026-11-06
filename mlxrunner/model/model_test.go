package model

import "testing"

func TestConfigArchitecture(t *testing.T) {
	for _, tt := range []struct {
		config, want string
	}{
		{`{"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5"}`, "Qwen3_5ForConditionalGeneration"},
		// MLX-native packs may declare only a model_type.
		{`{"model_type": "prism_hadamard_qwen35"}`, "prism_hadamard_qwen35"},
		{`{"architectures": [""], "model_type": "qwen3_5_mtp"}`, "qwen3_5_mtp"},
		{`{"text_config": {"model_type": "qwen3_5_text"}}`, ""},
	} {
		got, err := configArchitecture([]byte(tt.config))
		if err != nil {
			t.Fatalf("configArchitecture(%s): %v", tt.config, err)
		}
		if got != tt.want {
			t.Errorf("configArchitecture(%s) = %q, want %q", tt.config, got, tt.want)
		}
	}
	if _, err := configArchitecture([]byte(`{"architectures": "Qwen3"}`)); err == nil {
		t.Error("configArchitecture accepted a malformed architectures list")
	}
}
