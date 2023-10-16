package server

import (
	"testing"

	"github.com/jmorganca/ollama/api"
)

func TestModelPrompt(t *testing.T) {
	var m Model
	req := api.GenerateRequest{
		Template: "a{{ .Prompt }}b",
		Prompt:   "<h1>",
	}
	s, err := m.Prompt(req)
	if err != nil {
		t.Fatal(err)
	}
	want := "a<h1>b"
	if s != want {
		t.Errorf("got %q, want %q", s, want)
	}
}

func TestRunnerDigest_Success(t *testing.T) {
	model := &Model{
		Name:         "TestModel",
		ShortName:    "TM",
		ModelPath:    "/path/to/model",
		BaseModel:    "Original",
		AdapterPaths: []string{"/path/1", "/path/2"},
		License:      []string{"MIT"},
		Options:      map[string]interface{}{"key": "value"},
	}

	_, err := runnerDigest(model)
	if err != nil {
		t.Errorf("Failed to create digest: %v", err)
	}
}

func TestRunnerDigest_DifferentModels(t *testing.T) {
	model1 := &Model{
		Name:         "TestModel",
		ShortName:    "TM",
		ModelPath:    "/path/to/model",
		BaseModel:    "Original",
		AdapterPaths: []string{"/path/1", "/path/2"},
		License:      []string{"MIT"},
		Options:      map[string]interface{}{"key": "value"},
	}

	model2 := &Model{
		Name:         "AnotherModel",
		ShortName:    "AM",
		ModelPath:    "/another/path",
		BaseModel:    "DifferentOriginal",
		AdapterPaths: []string{"/path/3"},
		License:      []string{"Apache"},
		Options:      map[string]interface{}{"newKey": "newValue"},
	}

	digest1, _ := runnerDigest(model1)
	digest2, _ := runnerDigest(model2)

	if digest1 == digest2 {
		t.Error("Different models should have different digests")
	}
}

func TestRunnerDigest_SameDigestDifferentTemplate(t *testing.T) {
	model := &Model{
		Name:     "TestModel",
		Template: "Template1",
	}
	digest1, _ := runnerDigest(model)

	model.Template = "Template2"
	digest2, _ := runnerDigest(model)

	if digest1 != digest2 {
		t.Error("Changing only the Template should not change the digest")
	}
}

func TestRunnerDigest_SameDigestDifferentSystem(t *testing.T) {
	model := &Model{
		Name:   "TestModel",
		System: "System1",
	}
	digest1, _ := runnerDigest(model)

	model.System = "System2"
	digest2, _ := runnerDigest(model)

	if digest1 != digest2 {
		t.Error("Changing only the System should not change the digest")
	}
}
