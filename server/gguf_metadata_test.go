package server

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"slices"
	"testing"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

func TestGGUFMetadataExtraction(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	big := make([]string, 5000)
	for i := range big {
		big[i] = "tok"
	}
	perLayer := make([]int32, 64)
	for i := range perLayer {
		perLayer[i] = int32(i)
	}

	path, _ := createBinFile(t, ggml.KV{
		"general.architecture":         "bert",
		"bert.attention.softcap":       float32(math.Inf(-1)),
		"general.file_type":            uint32(2),
		"bert.pooling_type":            uint32(1),
		"bert.context_length":          uint32(512),
		"bert.attention.head_count_kv": perLayer,
		"tokenizer.chat_template":      "{{ messages }}",
		"tokenizer.ggml.tokens":        big,
		"tokenizer.ggml.merges":        big,
	}, nil)

	md, err := extractGGUFMetadata(path)
	if err != nil {
		t.Fatal(err)
	}

	if got := md.String("general.architecture"); got != "bert" {
		t.Errorf("architecture = %q", got)
	}
	// Architecture-prefixed lookup, and the key order in the file must not matter.
	if !md.Valid("pooling_type") {
		t.Error("pooling_type not found")
	}
	if got := md.Int("context_length"); got != 512 {
		t.Errorf("context_length = %d, want 512", got)
	}
	if got := md.String("tokenizer.chat_template"); got != "{{ messages }}" {
		t.Errorf("chat_template = %q", got)
	}
	if md.Valid("vision.block_count") {
		t.Error("absent key reported present")
	}

	// A 64-element per-layer array is real metadata and must survive.
	if v, ok := md.KV["bert.attention.head_count_kv"].([]any); !ok || len(v) != 64 {
		t.Errorf("per-layer array = %#v, want 64 elements", md.KV["bert.attention.head_count_kv"])
	}

	// The tokenizer is recorded as omitted, not silently missing.
	want := map[string]bool{"tokenizer.ggml.tokens": true, "tokenizer.ggml.merges": true}
	for _, key := range md.Omitted {
		delete(want, key)
	}
	if len(want) != 0 {
		t.Errorf("not reported omitted: %v (omitted=%v)", want, md.Omitted)
	}
	if _, ok := md.KV["tokenizer.ggml.tokens"]; ok {
		t.Error("oversized array was copied into the metadata file")
	}

	// JSON cannot represent an infinity, and gemma3n ships one. Dropping that
	// key must not cost the rest of the file.
	if !slices.Contains(md.Omitted, "bert.attention.softcap") {
		t.Errorf("non-finite float not omitted; omitted=%v", md.Omitted)
	}
}

func TestGGUFMetadataFileRoundTrip(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	_, digest := createBinFile(t, ggml.KV{
		"general.architecture": "llama",
		"llama.block_count":    uint32(32),
	}, nil)

	first, err := readGGUFMetadata(digest)
	if err != nil {
		t.Fatal(err)
	}
	if got := first.Int("block_count"); got != 32 {
		t.Fatalf("block_count = %d, want 32", got)
	}

	// With the blob gone, only the metadata file can answer.
	if err := removeBlob(t, digest); err != nil {
		t.Fatal(err)
	}
	second, err := readGGUFMetadata(digest)
	if err != nil {
		t.Fatalf("metadata file read failed: %v", err)
	}
	if got := second.Int("block_count"); got != 32 {
		t.Fatalf("block_count from metadata file = %d, want 32", got)
	}

	// And it goes away with the blob.
	removeGGUFMetadata(digest)
	path, err := ggufMetadataPath(digest)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("metadata file still present after removal: %v", err)
	}
}

// The metadata file is only ever an optimization: anything unusable must be re-extracted.
func TestGGUFMetadataUnusableFile(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	_, digest := createBinFile(t, ggml.KV{"general.architecture": "llama"}, nil)
	if _, err := readGGUFMetadata(digest); err != nil {
		t.Fatal(err)
	}
	path, err := ggufMetadataPath(digest)
	if err != nil {
		t.Fatal(err)
	}

	for _, tt := range []struct {
		name    string
		content []byte
	}{
		{"truncated", []byte(`{"kv":{"general.arch`)},
		{"not json", []byte("\x00\x01\x02")},
		{"empty", []byte{}},
		{"wrong shape", []byte(`[1,2,3]`)},
		{"no kv", []byte(`{"ollama_version":"1.2.3"}`)},
		{"kv not an object", []byte(`{"kv":[]}`)},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if err := os.WriteFile(path, tt.content, 0o644); err != nil {
				t.Fatal(err)
			}
			if _, ok := loadGGUFMetadata(path); ok {
				t.Fatal("unusable metadata file reported usable")
			}
			md, err := readGGUFMetadata(digest)
			if err != nil {
				t.Fatal(err)
			}
			if got := md.String("general.architecture"); got != "llama" {
				t.Fatalf("architecture = %q, want llama", got)
			}
		})
	}
}

func TestGGUFMetadataRejectsMalformed(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	dir := t.TempDir()

	for _, tt := range []struct {
		name string
		data []byte
	}{
		{"empty file", nil},
		{"bad magic", []byte("NOPE")},
		{"truncated header", []byte("GGUF\x03\x00\x00\x00")},
	} {
		t.Run(tt.name, func(t *testing.T) {
			path := filepath.Join(dir, tt.name)
			if err := os.WriteFile(path, tt.data, 0o600); err != nil {
				t.Fatal(err)
			}
			if _, err := extractGGUFMetadata(path); err == nil {
				t.Fatal("expected an error")
			}
		})
	}
}

// The omission rule decides what never reaches the metadata file, so every array type
// has to agree on it. A miss here silently drops real metadata.
func TestGGUFMetadataOmitRule(t *testing.T) {
	short, long := 64, ggufMetadataMaxArray+1

	for _, tt := range []struct {
		name string
		keep any
		omit any
	}{
		{"string", make([]string, short), make([]string, long)},
		{"int8", make([]int8, short), make([]int8, long)},
		{"int16", make([]int16, short), make([]int16, long)},
		{"int32", make([]int32, short), make([]int32, long)},
		{"int64", make([]int64, short), make([]int64, long)},
		{"uint8", make([]uint8, short), make([]uint8, long)},
		{"uint16", make([]uint16, short), make([]uint16, long)},
		{"uint32", make([]uint32, short), make([]uint32, long)},
		{"uint64", make([]uint64, short), make([]uint64, long)},
		{"float32", make([]float32, short), make([]float32, long)},
		{"float64", make([]float64, short), make([]float64, long)},
		{"bool", make([]bool, short), make([]bool, long)},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if omitValue(tt.keep) {
				t.Errorf("%d-element %s array omitted; real per-layer metadata would be lost", short, tt.name)
			}
			if !omitValue(tt.omit) {
				t.Errorf("%d-element %s array kept; the tokenizer would be copied", long, tt.name)
			}
		})
	}

	// Scalars are always kept, except the floats JSON cannot encode.
	for _, keep := range []any{"s", int32(0), uint32(0), float32(1.5), float64(1.5), true} {
		if omitValue(keep) {
			t.Errorf("scalar %#v omitted", keep)
		}
	}
	for _, omit := range []any{
		float32(math.Inf(1)), float32(math.Inf(-1)), float64(math.NaN()),
		[]float32{1, float32(math.Inf(1))},
		[]float64{math.NaN()},
	} {
		if !omitValue(omit) {
			t.Errorf("non-finite %#v kept; json.Marshal would fail the whole file", omit)
		}
	}
}

func TestGGUFMetadataKeysIncludesOmitted(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	big := make([]string, ggufMetadataMaxArray+1)
	path, _ := createBinFile(t, ggml.KV{
		"general.architecture":  "llama",
		"llama.block_count":     uint32(4),
		"tokenizer.ggml.tokens": big,
	}, nil)

	md, err := extractGGUFMetadata(path)
	if err != nil {
		t.Fatal(err)
	}
	keys := md.Keys()
	for _, want := range []string{"general.architecture", "llama.block_count", "tokenizer.ggml.tokens"} {
		if !slices.Contains(keys, want) {
			t.Errorf("Keys() = %v, missing %s", keys, want)
		}
	}
}

// Projector audio detection reads through Keys(), including prefixed forms.
func TestProjectorAudioFromMetadata(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	for _, tt := range []struct {
		name  string
		kv    ggml.KV
		audio bool
	}{
		{"bare key", ggml.KV{"general.architecture": "clip", "has_audio_encoder": true}, true},
		{"prefixed key", ggml.KV{"general.architecture": "clip", "clip.has_audio_encoder": true}, true},
		{"present but false", ggml.KV{"general.architecture": "clip", "has_audio_encoder": false}, false},
		{"absent", ggml.KV{"general.architecture": "clip"}, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			path, _ := createBinFile(t, tt.kv, nil)
			md, err := extractGGUFMetadata(path)
			if err != nil {
				t.Fatal(err)
			}
			if got := projectorHasAudio(md); got != tt.audio {
				t.Errorf("projectorHasAudio = %v, want %v", got, tt.audio)
			}
		})
	}

	path, _ := createBinFile(t, ggml.KV{
		"general.architecture":       "clip",
		"has_audio_encoder":          true,
		"clip.vision.projector_type": "gemma3nv",
	}, nil)
	md, err := extractGGUFMetadata(path)
	if err != nil {
		t.Fatal(err)
	}
	if !projectorSuppressesAudioCapability(md) {
		t.Error("gemma3nv projector should suppress audio")
	}
}

func TestGGUFMetadataPathRejectsBadDigest(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	hex := "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"

	for _, digest := range []string{"", "sha256-short", "md5:" + hex, "../escape", "sha256:" + hex + "x"} {
		if _, err := ggufMetadataPath(digest); err == nil {
			t.Errorf("ggufMetadataPath(%q) accepted", digest)
		}
	}
	for _, digest := range []string{"sha256:" + hex, "sha256-" + hex} {
		if _, err := ggufMetadataPath(digest); err != nil {
			t.Errorf("ggufMetadataPath(%q) rejected: %v", digest, err)
		}
	}
}

// A file can carry no values at all, and every value it does carry can be one
// the omission rule drops. Either way what is left still says which keys the
// file had, so it has to be storable: treating it as corrupt would rescan the
// blob on every request forever.
func TestGGUFMetadataWithNothingToCopy(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	path := filepath.Join(t.TempDir(), "empty.gguf")
	if err := os.WriteFile(path, ggufBytes(0, nil), 0o600); err != nil {
		t.Fatal(err)
	}

	md, err := extractGGUFMetadata(path)
	if err != nil {
		t.Fatalf("a file with no metadata is still a file: %v", err)
	}
	if len(md.KV) != 0 {
		t.Errorf("KV = %v, want empty", md.KV)
	}

	encoded, err := json.Marshal(md)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := decodeGGUFMetadata(encoded); err != nil {
		t.Errorf("written metadata cannot be read back, so it would be rescanned forever: %v", err)
	}
}

// allocBudget is what a scan of a file this small may allocate. The files below
// are a header and at most one key, so anything larger means a count in the
// file was believed.
const allocBudget = 1 << 20

func ggufBytes(kvCount uint64, extra []byte) []byte {
	b := make([]byte, 0, 24+len(extra))
	b = binary.LittleEndian.AppendUint32(b, 0x46554747) // GGUF
	b = binary.LittleEndian.AppendUint32(b, 3)          // version
	b = binary.LittleEndian.AppendUint64(b, 0)          // tensor count
	b = binary.LittleEndian.AppendUint64(b, kvCount)    // kv count
	return append(b, extra...)
}

func hostileGGUF(t *testing.T, kvCount uint64, extra []byte) string {
	t.Helper()
	b := ggufBytes(kvCount, extra)

	path := filepath.Join(t.TempDir(), "hostile.gguf")
	if err := os.WriteFile(path, b, 0o600); err != nil {
		t.Fatal(err)
	}
	t.Logf("file is %d bytes, declares %d KV entries", len(b), kvCount)
	return path
}

func allocDelta(t *testing.T, fn func()) uint64 {
	t.Helper()
	var before, after runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&before)
	fn()
	runtime.ReadMemStats(&after)
	return after.TotalAlloc - before.TotalAlloc
}

func TestGGUFMetadataBoundsDeclaredKVCount(t *testing.T) {
	path := hostileGGUF(t, 1<<20, nil)
	got := allocDelta(t, func() {
		if _, err := extractGGUFMetadata(path); err == nil {
			t.Error("expected an error from a 24-byte file")
		}
	})
	t.Logf("allocated %d bytes before failing", got)
	if got > allocBudget {
		t.Errorf("allocated %d bytes from a 24-byte file, want at most %d", got, allocBudget)
	}
}

// Counts and lengths in a GGUF are attacker controlled, and the scanner reaches
// every value in the file, so nothing may be sized from them before the bytes
// have actually been read. Every element type is covered: the dispatch splits
// strings from fixed-width values into separate readers, and fixing only one of
// them is how this hole stayed open.
func TestGGUFMetadataBoundsDeclaredArray(t *testing.T) {
	for _, tt := range []struct {
		name     string
		elemType uint32
	}{
		{"uint8", 0},
		{"int8", 1},
		{"uint16", 2},
		{"int16", 3},
		{"uint32", 4},
		{"int32", 5},
		{"float32", 6},
		{"bool", 7},
		{"string", 8},
		{"uint64", 10},
		{"int64", 11},
		{"float64", 12},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var extra []byte
			extra = binary.LittleEndian.AppendUint64(extra, 3) // key length
			extra = append(extra, "big"...)
			extra = binary.LittleEndian.AppendUint32(extra, 9) // type: array
			extra = binary.LittleEndian.AppendUint32(extra, tt.elemType)
			extra = binary.LittleEndian.AppendUint64(extra, 8_000_000) // element count

			path := hostileGGUF(t, 1, extra)
			got := allocDelta(t, func() {
				if _, err := extractGGUFMetadata(path); err == nil {
					t.Error("expected an error")
				}
			})
			if got > allocBudget {
				t.Errorf("allocated %d bytes for a declared array of 8000000, want at most %d", got, allocBudget)
			}
		})
	}
}

func removeBlob(t *testing.T, digest string) error {
	t.Helper()
	path, err := manifest.BlobsPath(digest)
	if err != nil {
		t.Fatal(err)
	}
	return os.Remove(path)
}

func metadataCount(t *testing.T) int {
	t.Helper()
	entries, err := os.ReadDir(filepath.Join(envconfig.Models(), "metadata"))
	if err != nil {
		return 0
	}
	return len(entries)
}

// createModelFromBlob and deleteModelNamed drive the handlers rather than
// writing manifests, so the metadata hooks under test are the ones the server
// actually runs.
func createModelFromBlob(t *testing.T, name, digest, tmpl string) {
	t.Helper()
	var s Server
	w := createRequest(t, s.CreateHandler, api.CreateRequest{
		Model: name, Files: map[string]string{"model.gguf": digest},
		Template: tmpl, Stream: &stream,
	})
	if w.Code != http.StatusOK {
		t.Fatalf("create %s: %d %s", name, w.Code, w.Body.String())
	}
}

func deleteModelNamed(t *testing.T, name string) {
	t.Helper()
	var s Server
	w := createRequest(t, s.DeleteHandler, api.DeleteRequest{Model: name})
	if w.Code != http.StatusOK {
		t.Fatalf("delete %s: %d %s", name, w.Code, w.Body.String())
	}
}

// Metadata is derived from a blob, so it lives exactly as long as the blob: it
// survives while any manifest still references it and goes with the last one.
func TestGGUFMetadataRemovedWithLastReference(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	_, digest := createBinFile(t, ggml.KV{
		"general.architecture": "llama",
		"llama.block_count":    uint32(1),
	}, nil)

	createModelFromBlob(t, "alias-one", digest, "")
	createModelFromBlob(t, "alias-two", digest, "")

	if _, err := readGGUFMetadata(digest); err != nil {
		t.Fatal(err)
	}
	if got := metadataCount(t); got != 1 {
		t.Fatalf("metadata files = %d, want 1 (both aliases share one blob)", got)
	}

	deleteModelNamed(t, "alias-one")
	if got := metadataCount(t); got != 1 {
		t.Fatalf("metadata files after removing one alias = %d, want 1", got)
	}

	deleteModelNamed(t, "alias-two")
	if got := metadataCount(t); got != 0 {
		t.Fatalf("metadata files after removing the last reference = %d, want 0", got)
	}
}

// Recreating a model over the same name replaces its blob; the old blob's
// metadata must go with it.
func TestGGUFMetadataRemovedOnReplacement(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	_, first := createBinFile(t, ggml.KV{
		"general.architecture": "llama",
		"llama.block_count":    uint32(1),
	}, nil)
	createModelFromBlob(t, "replace-me", first, "")
	if _, err := readGGUFMetadata(first); err != nil {
		t.Fatal(err)
	}

	_, second := createBinFile(t, ggml.KV{
		"general.architecture": "llama",
		"llama.block_count":    uint32(2),
	}, nil)
	createModelFromBlob(t, "replace-me", second, "")
	if _, err := readGGUFMetadata(second); err != nil {
		t.Fatal(err)
	}

	if got := metadataCount(t); got != 1 {
		t.Fatalf("metadata files after replacement = %d, want 1 (the old blob's should be gone)", got)
	}
	old, err := ggufMetadataPath(first)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(old); !os.IsNotExist(err) {
		t.Errorf("metadata for the replaced blob still present: %v", err)
	}
}

// A blob missing from disk is still a blob whose metadata has to go: prune
// reaches this path for a partially removed store, and skipping it there leaks
// the metadata permanently, since nothing later reports that digest again.
func TestGGUFMetadataRemovedForMissingBlob(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	_, digest := createBinFile(t, ggml.KV{"general.architecture": "llama"}, nil)
	if _, err := readGGUFMetadata(digest); err != nil {
		t.Fatal(err)
	}
	if err := removeBlob(t, digest); err != nil {
		t.Fatal(err)
	}
	if got := metadataCount(t); got != 1 {
		t.Fatalf("metadata files before the sweep = %d, want 1", got)
	}

	if err := deleteUnusedLayers(map[string]struct{}{digest: {}}); err != nil {
		t.Fatal(err)
	}
	if got := metadataCount(t); got != 0 {
		t.Errorf("metadata files = %d, want 0", got)
	}
}

// The point of the metadata file is that loading a model no longer reads its
// blob. With the blob deleted, GetModel must still describe the model exactly
// as before - which is also the guard against anything reintroducing a
// per-request read of the model file.
func TestGetModelReadsNoBlob(t *testing.T) {
	gin.SetMode(gin.TestMode)
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	_, digest := createBinFile(t, ggml.KV{
		"general.architecture":  "bert",
		"bert.pooling_type":     uint32(1),
		"bert.context_length":   uint32(512),
		"bert.embedding_length": uint32(384),
	}, nil)
	createModelFromBlob(t, "embedder", digest, "")

	want, err := GetModel("embedder")
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Contains(want.Capabilities(), model.CapabilityEmbedding) {
		t.Fatalf("capabilities = %v, want embedding", want.Capabilities())
	}

	if err := removeBlob(t, digest); err != nil {
		t.Fatal(err)
	}
	got, err := GetModel("embedder")
	if err != nil {
		t.Fatalf("GetModel needed the blob: %v", err)
	}
	if !slices.Equal(got.Capabilities(), want.Capabilities()) {
		t.Errorf("capabilities without the blob = %v, want %v", got.Capabilities(), want.Capabilities())
	}
	if got.metadata.Int("context_length") != 512 {
		t.Errorf("context_length = %d, want 512", got.metadata.Int("context_length"))
	}
}

// Extraction reaches every value in a file nobody reviewed, and both unbounded
// allocations found in review were malformations no table case had thought of.
// Invariants only: extraction either fails or produces metadata that survives
// the trip through a metadata file unchanged.
func FuzzGGUFMetadata(f *testing.F) {
	f.Add(ggufBytes(0, nil))
	f.Add(ggufBytes(1<<20, nil))

	var str []byte
	str = binary.LittleEndian.AppendUint64(str, 20)
	str = append(str, "general.architecture"...)
	str = binary.LittleEndian.AppendUint32(str, 8) // type: string
	str = binary.LittleEndian.AppendUint64(str, 5)
	str = append(str, "llama"...)
	f.Add(ggufBytes(1, str))

	var arr []byte
	arr = binary.LittleEndian.AppendUint64(arr, 3)
	arr = append(arr, "big"...)
	arr = binary.LittleEndian.AppendUint32(arr, 9) // type: array
	arr = binary.LittleEndian.AppendUint32(arr, 8) // element type: string
	arr = binary.LittleEndian.AppendUint64(arr, 8_000_000)
	f.Add(ggufBytes(1, arr))

	path := filepath.Join(f.TempDir(), "fuzz.gguf")
	f.Fuzz(func(t *testing.T, data []byte) {
		if err := os.WriteFile(path, data, 0o600); err != nil {
			t.Fatal(err)
		}

		md, err := extractGGUFMetadata(path)
		if err != nil {
			return
		}
		if md.KV == nil {
			t.Fatal("extraction reported success with no metadata")
		}

		encoded, err := json.Marshal(md)
		if err != nil {
			t.Fatalf("extracted metadata cannot be written: %v", err)
		}
		loaded, err := decodeGGUFMetadata(encoded)
		if err != nil {
			t.Fatalf("extracted metadata cannot be read back: %v", err)
		}
		if !reflect.DeepEqual(md, loaded) {
			t.Errorf("metadata changed on the trip through a file:\n%#v\n%#v", md, loaded)
		}
	})
}
