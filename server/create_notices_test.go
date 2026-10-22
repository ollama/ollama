package server

import (
	"bytes"
	"strings"
	"testing"

	"github.com/ollama/ollama/manifest"
)

func TestGGUFCreateNoticesRequiresFloor(t *testing.T) {
	t.Run("auto floor is named", func(t *testing.T) {
		notices := ggufCreateNotices(ggufCreateFacts{requires: ggufDefaultMinOllamaVersion, requiresAuto: true})
		if len(notices) != 1 || !strings.Contains(notices[0], "using default "+ggufDefaultMinOllamaVersion) {
			t.Fatalf("notices = %v, want one notice naming the default floor %s", notices, ggufDefaultMinOllamaVersion)
		}
	})

	t.Run("explicit floor is silent", func(t *testing.T) {
		notices := ggufCreateNotices(ggufCreateFacts{requires: "0.40.0"})
		if len(notices) != 0 {
			t.Fatalf("notices = %v, want none", notices)
		}
	})

	t.Run("no facts, no notice", func(t *testing.T) {
		if notices := ggufCreateNotices(ggufCreateFacts{}); len(notices) != 0 {
			t.Fatalf("notices = %v, want none", notices)
		}
	})
}

func TestCreateParamsMergesInheritedLayers(t *testing.T) {
	inherited, err := manifest.NewLayer(bytes.NewReader([]byte(`{"temperature": 1, "num_ctx": 4096}`)), "application/vnd.ollama.image.params")
	if err != nil {
		t.Fatal(err)
	}
	other, err := manifest.NewLayer(bytes.NewReader([]byte(`{"num_ctx": 8192}`)), "application/vnd.ollama.image.params")
	if err != nil {
		t.Fatal(err)
	}
	model, err := manifest.NewLayer(bytes.NewReader([]byte("x")), "application/vnd.ollama.image.model")
	if err != nil {
		t.Fatal(err)
	}

	baseLayers := []*modelLayer{
		{Layer: inherited},
		{Layer: model},
		{Layer: other},
	}
	params := createParams(baseLayers, map[string]any{"temperature": 0.5})

	// Request parameters win; earlier inherited layers win over later ones,
	// matching ApplyModelfileLayers merge order.
	if params["temperature"] != 0.5 {
		t.Errorf("temperature = %v, want request value 0.5", params["temperature"])
	}
	if params["num_ctx"] != float64(4096) {
		t.Errorf("num_ctx = %v, want first inherited value 4096", params["num_ctx"])
	}
	if len(params) != 2 {
		t.Errorf("params = %v, want exactly the merged keys", params)
	}
}
