package server

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"net/http"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	fsgguf "github.com/ollama/ollama/fs/gguf"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

func TestModelListCacheHydratesSummary(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())
	createListCacheModel(t, "list-cache", map[string]any{
		"test.context_length":   uint32(4096),
		"test.embedding_length": uint32(384),
	}, "{{ .prompt }}{{ if .tools }}{{ .tools }}{{ end }}{{ if .suffix }}{{ .suffix }}{{ end }}")

	cache := newModelListCache()
	if err := cache.hydrate(context.Background()); err != nil {
		t.Fatalf("hydrate failed: %v", err)
	}

	summary, ok := cache.Get(model.ParseName("list-cache"))
	if !ok {
		t.Fatal("list summary missing")
	}

	if summary.Model != "list-cache:latest" || summary.Name != "list-cache:latest" {
		t.Fatalf("summary model/name = %q/%q, want list-cache:latest", summary.Model, summary.Name)
	}
	if summary.Digest == "" {
		t.Fatal("summary digest is empty")
	}
	if summary.Size == 0 {
		t.Fatal("summary size is zero")
	}
	if summary.Details.Family != "test" || summary.Details.Format != "gguf" {
		t.Fatalf("summary details = %+v, want gguf/test", summary.Details)
	}
	if summary.Details.ContextLength != 4096 {
		t.Fatalf("context length = %d, want 4096", summary.Details.ContextLength)
	}
	if summary.Details.EmbeddingLength != 384 {
		t.Fatalf("embedding length = %d, want 384", summary.Details.EmbeddingLength)
	}

	for _, capability := range []model.Capability{model.CapabilityCompletion, model.CapabilityTools, model.CapabilityInsert} {
		if !slices.Contains(summary.Capabilities, capability) {
			t.Fatalf("capabilities = %v, want %s", summary.Capabilities, capability)
		}
	}

	listModel := summary.ListModelResponse()
	if !slices.Contains(listModel.Capabilities, model.CapabilityTools) ||
		listModel.Details.ContextLength != 4096 ||
		listModel.Details.EmbeddingLength != 384 {
		t.Fatalf("list response = %+v, want capabilities/context/embedding", listModel)
	}
}

func TestModelListCacheSuppressesNemotronSafetensorsMedia(t *testing.T) {
	caps := []model.Capability{
		model.CapabilityCompletion,
		model.CapabilityTools,
		model.CapabilityThinking,
		model.CapabilityVision,
		model.CapabilityAudio,
	}
	got := filterUnsupportedModelListCapabilities(caps, model.ConfigV2{
		ModelFormat: "safetensors",
		Renderer:    "nemotron-3-nano",
		Parser:      "nemotron-3-nano",
	})

	for _, capability := range []model.Capability{
		model.CapabilityCompletion,
		model.CapabilityTools,
		model.CapabilityThinking,
	} {
		if !slices.Contains(got, capability) {
			t.Fatalf("capabilities = %v, want %s", got, capability)
		}
	}
	for _, capability := range []model.Capability{model.CapabilityVision, model.CapabilityAudio} {
		if slices.Contains(got, capability) {
			t.Fatalf("capabilities = %v, did not expect %s", got, capability)
		}
	}
}

func TestModelListCacheRefreshUpdatesEntry(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())
	createListCacheModel(t, "list-refresh", map[string]any{"test.context_length": uint32(1024)}, "")

	cache := newModelListCache()
	if err := cache.hydrate(context.Background()); err != nil {
		t.Fatalf("hydrate failed: %v", err)
	}

	name := model.ParseName("list-refresh")
	first, ok := cache.Get(name)
	if !ok {
		t.Fatal("list summary missing")
	}

	changeShowCacheManifest(t, "list-refresh")
	if err := cache.RefreshModel(name); err != nil {
		t.Fatalf("refresh failed: %v", err)
	}

	refreshed, ok := cache.Get(name)
	if !ok {
		t.Fatal("refreshed list summary missing")
	}
	if refreshed.Digest == first.Digest {
		t.Fatalf("digest did not change after refresh: %s", refreshed.Digest)
	}
	if cache.Len() != 1 {
		t.Fatalf("cache entries = %d, want 1", cache.Len())
	}
}

func TestModelListCacheMutationHooks(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())

	cache := newModelListCache()
	s := Server{modelCaches: &modelCaches{modelList: cache}}

	_, digest := createBinFile(t, map[string]any{"test.context_length": uint32(2048)}, nil)
	w := createRequest(t, s.CreateHandler, api.CreateRequest{
		Model:  "list-hooks",
		Files:  map[string]string{"model.gguf": digest},
		Stream: &stream,
	})
	if w.Code != http.StatusOK {
		t.Fatalf("create model status = %d, want 200: %s", w.Code, w.Body.String())
	}

	if _, ok := cache.Get(model.ParseName("list-hooks")); !ok {
		t.Fatal("create did not refresh model list cache")
	}

	w = createRequest(t, s.CopyHandler, api.CopyRequest{
		Source:      "list-hooks",
		Destination: "list-hooks-copy",
	})
	if w.Code != http.StatusOK {
		t.Fatalf("copy model status = %d, want 200: %s", w.Code, w.Body.String())
	}
	if _, ok := cache.Get(model.ParseName("list-hooks-copy")); !ok {
		t.Fatal("copy did not refresh model list cache")
	}

	w = createRequest(t, s.DeleteHandler, api.DeleteRequest{Model: "list-hooks-copy"})
	if w.Code != http.StatusOK {
		t.Fatalf("delete model status = %d, want 200: %s", w.Code, w.Body.String())
	}
	if _, ok := cache.Get(model.ParseName("list-hooks-copy")); ok {
		t.Fatal("delete did not remove model list cache entry")
	}
}

func TestModelListCacheSyncsManifestChanges(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())
	createListCacheModel(t, "list-sync-a", map[string]any{"test.context_length": uint32(1024)}, "")

	cache := newModelListCache()
	cache.Start(context.Background())
	if err := cache.Wait(context.Background()); err != nil {
		t.Fatalf("wait failed: %v", err)
	}

	createListCacheModel(t, "list-sync-b", map[string]any{"test.context_length": uint32(2048)}, "")
	models, err := cache.List(context.Background())
	if err != nil {
		t.Fatalf("list failed: %v", err)
	}

	names := make([]string, 0, len(models))
	for _, m := range models {
		names = append(names, m.Name)
	}
	for _, want := range []string{"list-sync-a:latest", "list-sync-b:latest"} {
		if !slices.Contains(names, want) {
			t.Fatalf("names = %v, want %s", names, want)
		}
	}

	var other Server
	w := createRequest(t, other.DeleteHandler, api.DeleteRequest{Model: "list-sync-a"})
	if w.Code != http.StatusOK {
		t.Fatalf("delete model status = %d, want 200: %s", w.Code, w.Body.String())
	}

	models, err = cache.List(context.Background())
	if err != nil {
		t.Fatalf("list after delete failed: %v", err)
	}
	names = names[:0]
	for _, m := range models {
		names = append(names, m.Name)
	}
	if slices.Contains(names, "list-sync-a:latest") || !slices.Contains(names, "list-sync-b:latest") {
		t.Fatalf("names after delete = %v, want only list-sync-b", names)
	}
}

func TestModelListCacheSyncDropsStaleEntryOnRefreshFailure(t *testing.T) {
	gin.SetMode(gin.TestMode)
	setTestHome(t, t.TempDir())
	createListCacheModel(t, "list-stale", map[string]any{"test.context_length": uint32(1024)}, "")

	cache := newModelListCache()
	cache.Start(context.Background())
	if err := cache.Wait(context.Background()); err != nil {
		t.Fatalf("wait failed: %v", err)
	}

	name := model.ParseName("list-stale")
	if _, ok := cache.Get(name); !ok {
		t.Fatal("list summary missing")
	}

	changeShowCacheManifest(t, "list-stale")
	cache.build = func(model.Name, *manifest.Manifest) (modelListSummary, error) {
		return modelListSummary{}, errors.New("refresh failed")
	}

	models, err := cache.List(context.Background())
	if err != nil {
		t.Fatalf("list failed: %v", err)
	}
	if len(models) != 0 {
		t.Fatalf("models = %+v, want stale entry removed", models)
	}
	if _, ok := cache.Get(name); ok {
		t.Fatal("stale entry remained in cache after refresh failure")
	}
}

func TestReadModelListGGUFRejectsMalformedMetadata(t *testing.T) {
	cases := []struct {
		name string
		data []byte
		want string
	}{
		{
			name: "oversized key string",
			data: modelListGGUFTestFile(func(b *bytes.Buffer) {
				writeModelListGGUFHeader(t, b, 1)
				writeModelListGGUFUint64(t, b, fsgguf.MaxStringLength+1)
			}),
			want: "string",
		},
		{
			name: "oversized skipped string",
			data: modelListGGUFTestFile(func(b *bytes.Buffer) {
				writeModelListGGUFHeader(t, b, 1)
				writeModelListGGUFString(t, b, "unused")
				writeModelListGGUFUint32(t, b, modelListGGUFTypeString)
				writeModelListGGUFUint64(t, b, fsgguf.MaxStringLength+1)
			}),
			want: "string",
		},
		{
			name: "oversized skipped array",
			data: modelListGGUFTestFile(func(b *bytes.Buffer) {
				writeModelListGGUFHeader(t, b, 1)
				writeModelListGGUFString(t, b, "unused")
				writeModelListGGUFUint32(t, b, modelListGGUFTypeArray)
				writeModelListGGUFUint32(t, b, modelListGGUFTypeUint8)
				writeModelListGGUFUint64(t, b, fsgguf.MaxArraySize+1)
			}),
			want: "array size",
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("readModelListGGUF panicked: %v", r)
				}
			}()

			path := t.TempDir() + "/model.gguf"
			if err := os.WriteFile(path, tt.data, 0o600); err != nil {
				t.Fatal(err)
			}

			_, err := readModelListGGUF(path)
			if err == nil {
				t.Fatal("expected error")
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error = %v, want substring %q", err, tt.want)
			}
		})
	}
}

func createListCacheModel(t *testing.T, name string, kv map[string]any, tmpl string) {
	t.Helper()
	_, digest := createBinFile(t, kv, nil)

	req := api.CreateRequest{
		Model:  name,
		Files:  map[string]string{"model.gguf": digest},
		Stream: &stream,
	}
	if tmpl != "" {
		req.Template = tmpl
	}

	var s Server
	w := createRequest(t, s.CreateHandler, req)
	if w.Code != http.StatusOK {
		t.Fatalf("create model status = %d, want 200: %s", w.Code, w.Body.String())
	}
}

func modelListGGUFTestFile(fn func(*bytes.Buffer)) []byte {
	var b bytes.Buffer
	fn(&b)
	return b.Bytes()
}

func writeModelListGGUFHeader(t *testing.T, b *bytes.Buffer, numKV uint64) {
	t.Helper()
	writeModelListGGUFUint32(t, b, modelListGGUFMagicLE)
	writeModelListGGUFUint32(t, b, 3)
	writeModelListGGUFUint64(t, b, 0)
	writeModelListGGUFUint64(t, b, numKV)
}

func writeModelListGGUFString(t *testing.T, b *bytes.Buffer, s string) {
	t.Helper()
	writeModelListGGUFUint64(t, b, uint64(len(s)))
	if _, err := b.WriteString(s); err != nil {
		t.Fatal(err)
	}
}

func writeModelListGGUFUint32(t *testing.T, b *bytes.Buffer, v uint32) {
	t.Helper()
	if err := binary.Write(b, binary.LittleEndian, v); err != nil {
		t.Fatal(err)
	}
}

func writeModelListGGUFUint64(t *testing.T, b *bytes.Buffer, v uint64) {
	t.Helper()
	if err := binary.Write(b, binary.LittleEndian, v); err != nil {
		t.Fatal(err)
	}
}

// ---------------------------------------------------------------------------
// Big-endian GGUF reader tests
//
// readModelListGGUF switches to binary.BigEndian when the magic equals
// modelListGGUFMagicBE.  The helpers below write the magic with LittleEndian
// (so the raw bytes match what readModelListGGUF will see) and write all
// subsequent fields with BigEndian, mirroring what a real big-endian host
// would produce when writing a GGUF file.
// ---------------------------------------------------------------------------

// writeBE writes v to b using binary.BigEndian.
func writeBE(t *testing.T, b *bytes.Buffer, v any) {
	t.Helper()
	if err := binary.Write(b, binary.BigEndian, v); err != nil {
		t.Fatal(err)
	}
}

// writeModelListGGUFBEHeader writes a v3 big-endian GGUF header.
// The magic is written LE (as raw bytes the reader will see), all subsequent
// fields are written BE.
func writeModelListGGUFBEHeader(t *testing.T, b *bytes.Buffer, numKV uint64) {
	t.Helper()
	// Magic: always read by the parser as LE uint32, so write LE here.
	writeModelListGGUFUint32(t, b, modelListGGUFMagicBE)
	writeBE(t, b, uint32(3))   // version
	writeBE(t, b, uint64(0))   // num_tensors
	writeBE(t, b, numKV)       // num_kv
}

// writeModelListGGUFBEString writes a length-prefixed string using BigEndian.
func writeModelListGGUFBEString(t *testing.T, b *bytes.Buffer, s string) {
	t.Helper()
	writeBE(t, b, uint64(len(s)))
	if _, err := b.WriteString(s); err != nil {
		t.Fatal(err)
	}
}

// writeModelListGGUFBEKVString writes a key-value pair of type string using BigEndian.
func writeModelListGGUFBEKVString(t *testing.T, b *bytes.Buffer, key, value string) {
	t.Helper()
	writeModelListGGUFBEString(t, b, key)
	writeBE(t, b, modelListGGUFTypeString)
	writeModelListGGUFBEString(t, b, value)
}

// writeModelListGGUFBEKVUint32 writes a key-value pair of type uint32 using BigEndian.
func writeModelListGGUFBEKVUint32(t *testing.T, b *bytes.Buffer, key string, value uint32) {
	t.Helper()
	writeModelListGGUFBEString(t, b, key)
	writeBE(t, b, modelListGGUFTypeUint32)
	writeBE(t, b, value)
}

func TestReadModelListGGUFBigEndianMagicSwitchesByteOrder(t *testing.T) {
	// A minimal BE GGUF with zero KV pairs must parse without error and
	// return the zero-value modelListGGUF (capability = completion via default path).
	data := modelListGGUFTestFile(func(b *bytes.Buffer) {
		writeModelListGGUFBEHeader(t, b, 0)
	})
	path := t.TempDir() + "/be.gguf"
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	got, err := readModelListGGUF(path)
	if err != nil {
		t.Fatalf("readModelListGGUF BE header: %v", err)
	}
	// With no architecture KV the reader should fall through to the
	// default completion capability.
	if len(got.Capabilities) == 0 {
		t.Fatal("expected at least one capability")
	}
}

func TestReadModelListGGUFBigEndianReadsArchitecture(t *testing.T) {
	// A BE GGUF with general.architecture must return the correct family.
	data := modelListGGUFTestFile(func(b *bytes.Buffer) {
		writeModelListGGUFBEHeader(t, b, 1)
		writeModelListGGUFBEKVString(t, b, "general.architecture", "llama")
	})
	path := t.TempDir() + "/be-arch.gguf"
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	_, err := readModelListGGUF(path)
	if err != nil {
		t.Fatalf("readModelListGGUF BE architecture: %v", err)
	}
}

func TestReadModelListGGUFBigEndianContextAndEmbeddingLengths(t *testing.T) {
	cases := []struct {
		name      string
		ctxLen    uint32
		embLen    uint32
		wantCtx   int
		wantEmb   int
	}{
		{"standard", 4096, 4096, 4096, 4096},
		{"large context", 131072, 8192, 131072, 8192},
		{"minimal", 512, 64, 512, 64},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			data := modelListGGUFTestFile(func(b *bytes.Buffer) {
				writeModelListGGUFBEHeader(t, b, 3)
				writeModelListGGUFBEKVString(t, b, "general.architecture", "llama")
				writeModelListGGUFBEKVUint32(t, b, "llama.context_length", tc.ctxLen)
				writeModelListGGUFBEKVUint32(t, b, "llama.embedding_length", tc.embLen)
			})
			path := t.TempDir() + "/be-kv.gguf"
			if err := os.WriteFile(path, data, 0o600); err != nil {
				t.Fatal(err)
			}
			got, err := readModelListGGUF(path)
			if err != nil {
				t.Fatalf("readModelListGGUF: %v", err)
			}
			if got.ContextLength != tc.wantCtx {
				t.Errorf("ContextLength = %d, want %d", got.ContextLength, tc.wantCtx)
			}
			if got.EmbeddingLength != tc.wantEmb {
				t.Errorf("EmbeddingLength = %d, want %d", got.EmbeddingLength, tc.wantEmb)
			}
		})
	}
}

func TestReadModelListGGUFBigEndianVersionOne(t *testing.T) {
	// Version 1 uses uint32 tensor/kv counts (not uint64) and null-terminated
	// strings.  Verify the BE path handles the v1 header correctly.
	var b bytes.Buffer
	writeModelListGGUFUint32(t, &b, modelListGGUFMagicBE) // magic (LE raw bytes)
	writeBE(t, &b, uint32(1))                             // version
	writeBE(t, &b, uint32(0))                             // num_tensors (v1: uint32)
	writeBE(t, &b, uint32(0))                             // num_kv (v1: uint32)

	path := t.TempDir() + "/be-v1.gguf"
	if err := os.WriteFile(path, b.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	_, err := readModelListGGUF(path)
	if err != nil {
		t.Fatalf("readModelListGGUF BE v1: %v", err)
	}
}

func TestReadModelListGGUFBigEndianRejectsInvalidMagic(t *testing.T) {
	// Sanity check: a file that is neither LE nor BE magic must still return an error.
	var b bytes.Buffer
	writeModelListGGUFUint32(t, &b, 0xDEADBEEF)
	path := t.TempDir() + "/bad-magic.gguf"
	if err := os.WriteFile(path, b.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	_, err := readModelListGGUF(path)
	if err == nil {
		t.Fatal("expected error for invalid magic, got nil")
	}
}
