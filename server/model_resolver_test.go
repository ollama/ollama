package server

import (
	"errors"
	"strings"
	"testing"
)

func TestParseModelSelector(t *testing.T) {
	t.Run("cloud suffix", func(t *testing.T) {
		got, err := parseAndValidateModelRef("gpt-oss:20b:cloud")
		if err != nil {
			t.Fatalf("parseModelSelector returned error: %v", err)
		}

		if got.Source != modelSourceCloud {
			t.Fatalf("expected source cloud, got %v", got.Source)
		}

		if got.Base != "gpt-oss:20b" {
			t.Fatalf("expected base gpt-oss:20b, got %q", got.Base)
		}

		if got.Name.String() != "registry.ollama.ai/library/gpt-oss:20b" {
			t.Fatalf("unexpected resolved name: %q", got.Name.String())
		}
	})

	t.Run("legacy cloud suffix", func(t *testing.T) {
		got, err := parseAndValidateModelRef("gpt-oss:20b-cloud")
		if err != nil {
			t.Fatalf("parseModelSelector returned error: %v", err)
		}

		if got.Source != modelSourceCloud {
			t.Fatalf("expected source cloud, got %v", got.Source)
		}

		if got.Base != "gpt-oss:20b" {
			t.Fatalf("expected base gpt-oss:20b, got %q", got.Base)
		}
	})

	t.Run("bare dash cloud name is not explicit cloud", func(t *testing.T) {
		got, err := parseAndValidateModelRef("my-cloud-model")
		if err != nil {
			t.Fatalf("parseModelSelector returned error: %v", err)
		}

		if got.Source != modelSourceUnspecified {
			t.Fatalf("expected source unspecified, got %v", got.Source)
		}

		if got.Base != "my-cloud-model" {
			t.Fatalf("expected base my-cloud-model, got %q", got.Base)
		}
	})

	t.Run("local suffix", func(t *testing.T) {
		got, err := parseAndValidateModelRef("qwen3:8b:local")
		if err != nil {
			t.Fatalf("parseModelSelector returned error: %v", err)
		}

		if got.Source != modelSourceLocal {
			t.Fatalf("expected source local, got %v", got.Source)
		}

		if got.Base != "qwen3:8b" {
			t.Fatalf("expected base qwen3:8b, got %q", got.Base)
		}
	})

	t.Run("conflicting source suffixes fail", func(t *testing.T) {
		_, err := parseAndValidateModelRef("foo:cloud:local")
		if !errors.Is(err, errConflictingModelSource) {
			t.Fatalf("expected errConflictingModelSource, got %v", err)
		}
	})

	t.Run("unspecified source", func(t *testing.T) {
		got, err := parseAndValidateModelRef("llama3")
		if err != nil {
			t.Fatalf("parseModelSelector returned error: %v", err)
		}

		if got.Source != modelSourceUnspecified {
			t.Fatalf("expected source unspecified, got %v", got.Source)
		}

		if got.Name.Tag != "latest" {
			t.Fatalf("expected default latest tag, got %q", got.Name.Tag)
		}
	})

	t.Run("unknown suffix is treated as tag", func(t *testing.T) {
		got, err := parseAndValidateModelRef("gpt-oss:clod")
		if err != nil {
			t.Fatalf("parseModelSelector returned error: %v", err)
		}

		if got.Source != modelSourceUnspecified {
			t.Fatalf("expected source unspecified, got %v", got.Source)
		}

		if got.Name.Tag != "clod" {
			t.Fatalf("expected tag clod, got %q", got.Name.Tag)
		}
	})

	t.Run("empty model fails", func(t *testing.T) {
		_, err := parseAndValidateModelRef("")
		if !errors.Is(err, errModelRequired) {
			t.Fatalf("expected errModelRequired, got %v", err)
		}
	})

	t.Run("invalid model fails", func(t *testing.T) {
		_, err := parseAndValidateModelRef("::cloud")
		if err == nil {
			t.Fatal("expected error for invalid model")
		}
		if !strings.Contains(err.Error(), "unqualified") {
			t.Fatalf("expected unqualified model error, got %v", err)
		}
	})
}

func TestParsePullModelRef(t *testing.T) {
	t.Run("explicit local is normalized", func(t *testing.T) {
		got, err := parseNormalizePullModelRef("gpt-oss:20b:local")
		if err != nil {
			t.Fatalf("parseNormalizePullModelRef returned error: %v", err)
		}

		if got.Source != modelSourceLocal {
			t.Fatalf("expected source local, got %v", got.Source)
		}

		if got.Base != "gpt-oss:20b" {
			t.Fatalf("expected base gpt-oss:20b, got %q", got.Base)
		}
	})

	t.Run("explicit cloud with size maps to legacy cloud suffix", func(t *testing.T) {
		got, err := parseNormalizePullModelRef("gpt-oss:20b:cloud")
		if err != nil {
			t.Fatalf("parseNormalizePullModelRef returned error: %v", err)
		}
		if got.Base != "gpt-oss:20b-cloud" {
			t.Fatalf("expected base gpt-oss:20b-cloud, got %q", got.Base)
		}
		if got.Name.String() != "registry.ollama.ai/library/gpt-oss:20b-cloud" {
			t.Fatalf("unexpected resolved name: %q", got.Name.String())
		}
	})

	t.Run("explicit cloud without size maps to cloud tag", func(t *testing.T) {
		got, err := parseNormalizePullModelRef("qwen3:cloud")
		if err != nil {
			t.Fatalf("parseNormalizePullModelRef returned error: %v", err)
		}
		if got.Base != "qwen3:cloud" {
			t.Fatalf("expected base qwen3:cloud, got %q", got.Base)
		}
		if got.Name.String() != "registry.ollama.ai/library/qwen3:cloud" {
			t.Fatalf("unexpected resolved name: %q", got.Name.String())
		}
	})
}
