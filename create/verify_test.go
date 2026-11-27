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

func TestValidateMLXModelRequiresAnArchitecture(t *testing.T) {
	err := validateMLXSource(sourceModelConfig{}, false, MLXValidationOptions{})
	if !errors.Is(err, ErrUnsupportedMLXArchitecture) || !strings.Contains(err.Error(), "does not name an architecture") {
		t.Fatalf("validateMLXSource() error = %v, want an unnamed architecture error", err)
	}
}

// A config without architectures, as MLX-native packs may ship, resolves its
// architecture from model_type like the runner does.
func TestValidateMLXModelFallsBackToModelType(t *testing.T) {
	err := validateMLXSource(sourceModelConfig{ModelType: "qwen3"}, false, MLXValidationOptions{})
	if !errors.Is(err, ErrUnsupportedMLXArchitecture) || !strings.Contains(err.Error(), `"qwen3"`) {
		t.Fatalf("validateMLXSource() error = %v, want unsupported model \"qwen3\"", err)
	}

	err = validateMLXSource(sourceModelConfig{ModelType: "gemma4_unified"}, false, MLXValidationOptions{})
	if errors.Is(err, ErrUnsupportedMLXArchitecture) {
		t.Fatalf("validateMLXSource() error = %v, want gemma4_unified supported", err)
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
