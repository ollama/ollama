package server

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"os"
	"path/filepath"
	"strings"

	"github.com/ollama/ollama/envconfig"
	fsgguf "github.com/ollama/ollama/fs/gguf"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/version"
)

// A blob's metadata block is extracted once into a file beside the model
// store, keyed by blob digest. Values are held verbatim rather than derived, so
// reading them differently later needs no invalidation, and values left out are
// recorded by key so an absent key stays distinct from an uncopied one.

// Comfortably above per-layer arrays, which scale with block count, and far
// below any tokenizer vocabulary.
const ggufMetadataMaxArray = 4096

type ggufMetadata struct {
	OllamaVersion string `json:"ollama_version,omitempty"`

	// Values keyed exactly as they appear in the file. Never omitempty: absent
	// on load means the file is not ours, and an all-omitted file is still
	// usable.
	KV map[string]any `json:"kv"`
	// Keys present in the file whose values were too large to copy.
	Omitted []string `json:"omitted,omitempty"`
}

// Valid reports whether the key is present. Keys are architecture qualified the
// same way fs/gguf does it.
func (m ggufMetadata) Valid(key string) bool {
	_, ok := m.lookup(key)
	return ok
}

func (m ggufMetadata) String(key string) string {
	v, _ := m.lookup(key)
	s, _ := v.(string)
	return s
}

func (m ggufMetadata) Int(key string) int64 {
	v, _ := m.lookup(key)
	n, ok := v.(json.Number)
	if !ok {
		return 0
	}
	i, err := n.Int64()
	if err != nil {
		return 0
	}
	return i
}

// Keys returns every key the file carried, including omitted ones.
func (m ggufMetadata) Keys() []string {
	keys := make([]string, 0, len(m.KV)+len(m.Omitted))
	for k := range m.KV {
		keys = append(keys, k)
	}
	return append(keys, m.Omitted...)
}

func (m ggufMetadata) lookup(key string) (any, bool) {
	if !strings.HasPrefix(key, "general.") && !strings.HasPrefix(key, "tokenizer.") {
		arch, _ := m.KV["general.architecture"].(string)
		key = arch + "." + key
	}
	v, ok := m.KV[key]
	return v, ok
}

func ggufMetadataPath(digest string) (string, error) {
	if err := manifest.ValidateDigest(digest); err != nil {
		return "", fmt.Errorf("%w: %q", err, digest)
	}
	return filepath.Join(envconfig.Models(), "metadata", strings.ReplaceAll(digest, ":", "-")+".json"), nil
}

// readGGUFMetadata extracts the blob when there is no usable metadata file. Best
// effort throughout: a bad file is re-extracted, a failed write is dropped.
func readGGUFMetadata(digest string) (ggufMetadata, error) {
	path, err := ggufMetadataPath(digest)
	if err != nil {
		return ggufMetadata{}, err
	}
	if md, ok := loadGGUFMetadata(path); ok {
		return md, nil
	}

	blob, err := manifest.BlobsPath(digest)
	if err != nil {
		return ggufMetadata{}, err
	}
	md, err := extractGGUFMetadata(blob)
	if err != nil {
		return ggufMetadata{}, err
	}

	writeGGUFMetadata(path, md)
	return md, nil
}

func loadGGUFMetadata(path string) (ggufMetadata, bool) {
	data, err := os.ReadFile(path)
	if err != nil {
		if !errors.Is(err, os.ErrNotExist) {
			slog.Debug("could not read gguf metadata file", "path", path, "error", err)
		}
		return ggufMetadata{}, false
	}

	md, err := decodeGGUFMetadata(data)
	if err != nil {
		slog.Debug("ignoring unusable gguf metadata file", "path", path, "error", err)
		return ggufMetadata{}, false
	}
	return md, true
}

func writeGGUFMetadata(path string, md ggufMetadata) {
	data, err := json.Marshal(md)
	if err != nil {
		slog.Debug("could not encode gguf metadata", "path", path, "error", err)
		return
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		slog.Debug("could not create gguf metadata dir", "path", path, "error", err)
		return
	}

	tmp, err := os.CreateTemp(filepath.Dir(path), ".gguf-metadata-*.tmp")
	if err != nil {
		slog.Debug("could not create gguf metadata temp file", "path", path, "error", err)
		return
	}
	defer os.Remove(tmp.Name())

	if _, err = tmp.Write(data); err == nil {
		err = tmp.Sync()
	}
	if cerr := tmp.Close(); err == nil {
		err = cerr
	}
	if err == nil {
		// Can fail transiently on Windows while another process holds the path.
		err = os.Rename(tmp.Name(), path)
	}
	if err != nil {
		slog.Debug("could not write gguf metadata file", "path", path, "error", err)
	}
}

// removeGGUFMetadata drops metadata for deleted blobs. Pass only digests whose
// last reference is gone.
func removeGGUFMetadata(digests ...string) {
	for _, digest := range digests {
		path, err := ggufMetadataPath(digest)
		if err != nil {
			continue
		}
		if err := os.Remove(path); err != nil && !errors.Is(err, os.ErrNotExist) {
			slog.Debug("could not remove gguf metadata file", "path", path, "error", err)
		}
	}
}

// extractGGUFMetadata reads the whole block; proving a key absent means reaching
// the end. Normalized through JSON so extracted and loaded metadata hold the
// same value types.
func extractGGUFMetadata(path string) (ggufMetadata, error) {
	md, err := scanGGUFMetadata(path)
	if err != nil {
		return ggufMetadata{}, err
	}
	md.OllamaVersion = version.Version

	data, err := json.Marshal(md)
	if err != nil {
		return ggufMetadata{}, err
	}
	return decodeGGUFMetadata(data)
}

func decodeGGUFMetadata(data []byte) (ggufMetadata, error) {
	// UseNumber keeps integers exact; the default would widen them to float64.
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	var md ggufMetadata
	if err := dec.Decode(&md); err != nil {
		return ggufMetadata{}, err
	}
	if md.KV == nil {
		return ggufMetadata{}, fmt.Errorf("no metadata")
	}
	return md, nil
}

func scanGGUFMetadata(path string) (ggufMetadata, error) {
	f, err := fsgguf.Open(path)
	if err != nil {
		return ggufMetadata{}, err
	}
	defer f.Close()

	// not sized from f.NumKeyValues(): that count comes from the file
	md := ggufMetadata{KV: make(map[string]any)}
	for _, kv := range f.KeyValues() {
		if omitValue(kv.Any()) {
			md.Omitted = append(md.Omitted, kv.Key)
			continue
		}
		md.KV[kv.Key] = kv.Any()
	}
	if err := f.Err(); err != nil {
		return ggufMetadata{}, err
	}
	return md, nil
}

// omitValue reports values left out of the metadata file: long arrays, which are the
// tokenizer, and non-finite floats, which JSON cannot represent.
func omitValue(v any) bool {
	switch v := v.(type) {
	case float32:
		return !finite(float64(v))
	case float64:
		return !finite(v)
	case []float32:
		for _, f := range v {
			if !finite(float64(f)) {
				return true
			}
		}
	case []float64:
		for _, f := range v {
			if !finite(f) {
				return true
			}
		}
	}
	return oversized(v)
}

func finite(f float64) bool {
	return !math.IsInf(f, 0) && !math.IsNaN(f)
}

func oversized(v any) bool {
	switch v := v.(type) {
	case []string:
		return len(v) > ggufMetadataMaxArray
	case []int8:
		return len(v) > ggufMetadataMaxArray
	case []int16:
		return len(v) > ggufMetadataMaxArray
	case []int32:
		return len(v) > ggufMetadataMaxArray
	case []int64:
		return len(v) > ggufMetadataMaxArray
	case []uint8:
		return len(v) > ggufMetadataMaxArray
	case []uint16:
		return len(v) > ggufMetadataMaxArray
	case []uint32:
		return len(v) > ggufMetadataMaxArray
	case []uint64:
		return len(v) > ggufMetadataMaxArray
	case []float32:
		return len(v) > ggufMetadataMaxArray
	case []float64:
		return len(v) > ggufMetadataMaxArray
	case []bool:
		return len(v) > ggufMetadataMaxArray
	}
	return false
}
