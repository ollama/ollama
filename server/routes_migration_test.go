package server

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"net/http"
	"os"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/compatmigrate"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

func BenchmarkWarmModelForLoad(b *testing.B) {
	for _, format := range []string{manifest.FormatGGUF, manifest.FormatSafetensors} {
		b.Run(format, func(b *testing.B) {
			b.Setenv("OLLAMA_MODELS", b.TempDir())
			name := model.ParseName("warm-load:latest")
			data := []byte(`{"schemaVersion":2,"layers":[]}`)
			if err := manifest.WriteManifestData(name, data); err != nil {
				b.Fatal(err)
			}
			digest := fmt.Sprintf("%x", sha256.Sum256(data))
			if !manifest.NamedManifestPointsToDigest(name, digest) {
				b.Skip("requires symlink-based manifest lookup")
			}
			cached := &Model{Name: name.String(), Digest: digest, ManifestDigest: digest, Config: model.ConfigV2{ModelFormat: format}}
			s := &Server{sched: &Scheduler{loaded: map[string]*runnerRef{"warm": {model: cached}}}}
			b.ReportAllocs()
			for b.Loop() {
				m, err := s.getExistingModelForRunner(name.String(), "")
				if err != nil {
					b.Fatal(err)
				}
				m, err = s.modelForLoad(b.Context(), m)
				if err != nil || m != cached {
					b.Fatalf("warm lookup did not reuse its loaded model: %v", err)
				}
			}
		})
	}
}

// Exercise load coordination with tiny headers, not architecture conversions.
type loadTestMigrator struct {
	checks atomic.Int32
	err    error
}

func (m *loadTestMigrator) NeedsMigration(src *compatmigrate.SourceModel) bool {
	m.checks.Add(1)
	return src.GGUF.KeyValue("test.legacy").Bool()
}

func (m *loadTestMigrator) Migrate(*compatmigrate.SourceModel) (*compatmigrate.Result, error) {
	if m.err != nil {
		return nil, m.err
	}
	return &compatmigrate.Result{ModelKV: map[string]any{"general.architecture": "loadtest"}}, nil
}

func TestModelForLoadReusesCompatibleRunner(t *testing.T) {
	for _, tc := range []struct {
		name   string
		legacy bool
	}{{"clean", false}, {"converted", true}} {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("OLLAMA_MODELS", t.TempDir())
			migrator := &loadTestMigrator{}
			t.Cleanup(compatmigrate.SetMigratorsForTesting(map[string][]compatmigrate.Migrator{"loadtest": {migrator}}))
			name := model.ParseName("loadtest:latest")
			writeShowManifestVariant(t, name.String(), manifest.RunnerGGML, manifest.FormatGGUF,
				model.ConfigV2{ModelFormat: "gguf", ModelFamily: "loadtest"},
				map[string]any{"general.architecture": "loadtest", "test.legacy": tc.legacy})
			s := &Server{sched: &Scheduler{loaded: make(map[string]*runnerRef)}}
			initial, err := s.getExistingModelForRunner(name.String(), "")
			if err != nil {
				t.Fatal(err)
			}
			ready, err := s.modelForLoad(t.Context(), initial)
			if err != nil {
				t.Fatal(err)
			}
			if tc.legacy && (ready.ManifestDigest == initial.ManifestDigest || ready.Digest == ready.ManifestDigest) {
				t.Fatal("converted model must retain its new parent and child identities")
			}
			current, err := GetModel(name.String())
			if err != nil || current.Digest != ready.Digest || current.ManifestDigest != ready.ManifestDigest {
				t.Fatalf("converted load does not match its tag: %+v, %v", current, err)
			}
			s.sched.loaded["ready"] = &runnerRef{model: ready}
			checks := migrator.checks.Load()
			if checks == 0 {
				t.Fatal("cold load skipped compatibility detection")
			}
			for range 3 {
				warm, err := s.getExistingModelForRunner(name.String(), "")
				if err != nil {
					t.Fatal(err)
				}
				if manifest.NamedManifestPointsToDigest(name, ready.Digest) && warm != ready {
					t.Fatal("warm lookup did not reuse the loaded model")
				}
				if _, err := s.modelForLoad(t.Context(), warm); err != nil {
					t.Fatal(err)
				}
			}
			if got := migrator.checks.Load(); got != checks {
				t.Fatalf("warm requests repeated detection: checks %d -> %d", checks, got)
			}

			writeShowManifestVariant(t, name.String(), manifest.RunnerGGML, manifest.FormatGGUF,
				model.ConfigV2{ModelFormat: "gguf", ModelFamily: "loadtest"},
				map[string]any{"general.architecture": "loadtest", "test.replacement": true})
			changed, err := s.getExistingModelForRunner(name.String(), "")
			if err != nil {
				t.Fatal(err)
			}
			if _, err := s.modelForLoad(t.Context(), changed); err != nil {
				t.Fatal(err)
			}
			if migrator.checks.Load() <= checks {
				t.Fatal("a replacement tag bypassed compatibility detection")
			}

			// Even a copied manifest (Windows) produces a fresh Model pointer.
			// Matching loaded content must not reopen the GGUF for detection.
			copy := *ready
			if err := os.Remove(ready.ModelPath); err != nil {
				t.Fatal(err)
			}
			if _, err := s.modelForLoad(t.Context(), &copy); err != nil {
				t.Fatalf("loaded model reopened its GGUF: %v", err)
			}
			delete(s.sched.loaded, "ready")
			checks = migrator.checks.Load()
			if _, err := s.modelForLoad(t.Context(), changed); err != nil {
				t.Fatalf("cold replacement load: %v", err)
			}
			if migrator.checks.Load() <= checks {
				t.Fatal("cold reload bypassed compatibility detection")
			}
		})
	}
}

func TestInferenceWaitsForMigration(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	name := "legacy-load"
	writeShowManifestVariant(t, name, manifest.RunnerGGML, manifest.FormatGGUF,
		model.ConfigV2{ModelFormat: "gguf", ModelFamily: "loadtest"},
		map[string]any{"general.architecture": "loadtest", "test.legacy": true})
	failure := errors.New("conversion unavailable")
	migrator := &loadTestMigrator{err: failure}
	t.Cleanup(compatmigrate.SetMigratorsForTesting(map[string][]compatmigrate.Migrator{"loadtest": {migrator}}))
	s := &Server{}
	for _, tc := range []struct {
		name    string
		handler gin.HandlerFunc
		body    any
	}{
		{"generate", s.GenerateHandler, api.GenerateRequest{Model: name, Prompt: "hi"}},
		{"chat", s.ChatHandler, api.ChatRequest{Model: name, Messages: []api.Message{{Role: "user", Content: "hi"}}}},
		{"embed", s.EmbedHandler, api.EmbedRequest{Model: name, Input: "hi"}},
		{"embeddings", s.EmbeddingsHandler, api.EmbeddingRequest{Model: name, Prompt: "hi"}},
		{"decision", s.SystemOneHandler, map[string]any{"model": name, "state": "hello", "questions": map[string]any{"q": map[string]any{"type": "noul", "instructions": "Is this a greeting?"}}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for range 2 {
				before := migrator.checks.Load()
				w := createRequest(t, tc.handler, tc.body)
				if w.Code != http.StatusInternalServerError || !strings.Contains(w.Body.String(), failure.Error()) {
					t.Fatalf("request did not stop on conversion failure: %d %s", w.Code, w.Body)
				}
				if migrator.checks.Load() <= before {
					t.Fatal("failed conversion was not retried")
				}
			}
		})
	}
}
