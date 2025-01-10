package bert_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model"
)

func blob(t *testing.T, tag string) string {
	t.Helper()
	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatal(err)
	}

	p := filepath.Join(home, ".ollama", "models")
	manifestBytes, err := os.ReadFile(filepath.Join(p, "manifests", "registry.ollama.ai", "library", "all-minilm", tag))
	if err != nil {
		t.Fatal(err)
	}

	var manifest struct {
		Layers []struct {
			MediaType string `json:"mediaType"`
			Digest    string `json:"digest"`
		}
	}

	if err := json.Unmarshal(manifestBytes, &manifest); err != nil {
		t.Fatal(err)
	}

	var digest string
	for _, layer := range manifest.Layers {
		if layer.MediaType == "application/vnd.ollama.image.model" {
			digest = layer.Digest
			break
		}
	}

	if digest == "" {
		t.Fatal("no model layer found")
	}

	return filepath.Join(p, "blobs", strings.ReplaceAll(digest, ":", "-"))
}

func TestEmbedding(t *testing.T) {
	m, err := model.New(blob(t, "latest"))
	if err != nil {
		t.Fatal(err)
	}

	text, err := os.ReadFile(filepath.Join("..", "testdata", "war-and-peace.txt"))
	if err != nil {
		t.Fatal(err)
	}

	inputIDs, err := m.(model.TextProcessor).Encode(string(text))
	if err != nil {
		t.Fatal(err)
	}

	logit, err := model.Forward(m, model.WithInputIDs(inputIDs))
	if err != nil {
		t.Fatal(err)
	}

	t.Log(ml.Dump(logit))
}
