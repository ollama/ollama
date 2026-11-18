package create

import (
	"errors"
	"strings"
	"testing"
)

func TestValidateMLXModelRejectsUnsupportedArchitecture(t *testing.T) {
	err := validateMLXSource(sourceModelConfig{Architectures: []string{"UnsupportedForCausalLM"}}, false, MLXValidationOptions{})
	if !errors.Is(err, ErrUnsupportedMLXArchitecture) {
		t.Fatalf("validateMLXSource() error = %v, want ErrUnsupportedMLXArchitecture", err)
	}
}

func TestValidateMLXModelRequiresArchitecturesField(t *testing.T) {
	err := validateMLXSource(sourceModelConfig{ModelType: "qwen3"}, false, MLXValidationOptions{})
	if !errors.Is(err, ErrUnsupportedMLXArchitecture) {
		t.Fatalf("validateMLXSource() error = %v, want ErrUnsupportedMLXArchitecture", err)
	}
}

func TestValidateMLXDraftIgnoresTextConfigFallback(t *testing.T) {
	cfg := sourceModelConfig{}
	cfg.TextConfig.ModelType = "DFlashDraftModel"
	err := validateMLXSource(cfg, true, MLXValidationOptions{})
	if !errors.Is(err, ErrUnsupportedMLXArchitecture) {
		t.Fatalf("validateMLXSource() error = %v, want ErrUnsupportedMLXArchitecture", err)
	}
}

func TestValidateMLXModelFallsBackToModelType(t *testing.T) {
	// Checkpoints that write only model_type to config.json resolve through a
	// registered model_type alias (gemma4_unified) instead of being rejected.
	err := validateMLXSource(sourceModelConfig{ModelType: "gemma4_unified"}, false, MLXValidationOptions{Force: true})
	if err != nil {
		t.Fatalf("validateMLXSource() error = %v", err)
	}
}

func TestValidateMLXDraftFallsBackToModelType(t *testing.T) {
	err := validateMLXSource(sourceModelConfig{ModelType: "gemma4_assistant"}, true, MLXValidationOptions{Force: true})
	if err != nil {
		t.Fatalf("validateMLXSource() error = %v", err)
	}
}

func TestValidateMLXModelForceWarnsAndContinues(t *testing.T) {
	var warnings []string
	err := validateMLXSource(sourceModelConfig{Architectures: []string{"UnsupportedForCausalLM"}}, false, MLXValidationOptions{
		Force: true,
		Warning: func(message string) {
			warnings = append(warnings, message)
		},
	})
	if err != nil {
		t.Fatalf("validateMLXSource() error = %v", err)
	}
	if len(warnings) == 0 || !strings.Contains(warnings[0], "UnsupportedForCausalLM") {
		t.Fatalf("warnings = %q, want unsupported architecture warning", warnings)
	}
}
