package server

import (
	"bytes"
	"context"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

// A RIFF/WAVE header is all it takes for llm.DetectMediaKind to see a clip;
// none of this reaches a model.
var clipBytes = []byte("RIFF\x24\x00\x00\x00WAVEfmt ")

func mediaTestModel(t *testing.T, audio bool) *Model {
	t.Helper()
	textModelPath, _ := createBinFile(t, map[string]any{
		"general.architecture": "llama",
	}, nil)
	projectorKV := map[string]any{
		"general.architecture":    "clip",
		"clip.vision.block_count": uint32(1),
	}
	if audio {
		projectorKV["clip.has_audio_encoder"] = true
	}
	projectorPath, _ := createBinFile(t, projectorKV, nil)

	m := &Model{
		Name:           "test",
		ModelPath:      textModelPath,
		ProjectorPaths: []string{projectorPath},
	}
	loadTestMetadata(t, m)
	return m
}

func TestMessageMediaCapabilitiesRejectsAudioOnTextModel(t *testing.T) {
	m := mediaTestModel(t, false)
	_, _, _, err := (&Server{}).scheduleRunnerForModel(context.Background(), m, messageMediaCapabilities([]api.Message{{
		Role:    "user",
		Content: "Transcribe this.",
		Images:  []api.ImageData{clipBytes},
	}}), nil, nil, nil)
	if err == nil || !bytes.Contains([]byte(err.Error()), []byte("audio")) {
		t.Fatalf("audio sent to a model with %v: err = %v, want an audio capability error", m.Capabilities(), err)
	}
}

func TestMessageMediaCapabilitiesKeepsAudioForAudioModel(t *testing.T) {
	m := mediaTestModel(t, true)
	if err := m.CheckCapabilities(model.CapabilityAudio); err != nil {
		t.Fatalf("fixture has no audio capability: %v", err)
	}
	if err := m.CheckCapabilities(messageMediaCapabilities([]api.Message{{
		Role:    "user",
		Content: "Transcribe this.",
		Images:  []api.ImageData{clipBytes},
	}})...); err != nil {
		t.Fatalf("audio sent to an audio model: %v", err)
	}
}

func TestMessageMediaCapabilitiesKeepsImages(t *testing.T) {
	m := mediaTestModel(t, true)
	if err := m.CheckCapabilities(model.CapabilityVision); err != nil {
		t.Fatalf("fixture has no vision capability: %v", err)
	}
	// One message holding both kinds is fine as long as the model has the
	// encoders for them.
	if err := m.CheckCapabilities(messageMediaCapabilities([]api.Message{{
		Role:    "user",
		Content: "Look and listen.",
		Images:  []api.ImageData{[]byte("\x89PNG\r\n\x1a\n"), clipBytes},
	}})...); err != nil {
		t.Fatalf("image and audio on a model that takes both: %v", err)
	}
}
