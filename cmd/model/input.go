package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

// maxMetadataSize caps allocation for config and tokenizer metadata at 256 MiB.
const maxMetadataSize = 256 << 20

// A future API mode should expose a complete read-only artifact inventory with
// tensor-to-blob ranges and trusted blob digests. It should not hash tensor
// payloads on the server.

type Artifact struct {
	Name      string
	MediaType string
	File      string
	Blob      string
	From      string
	Bytes     int64
}

func inspect(ctx context.Context, ref, store string) (*inventory, error) {
	if store == "" {
		store = envconfig.Models()
	}
	store, err := filepath.Abs(store)
	if err != nil {
		return nil, err
	}
	path, err := filepath.Abs(ref)
	if err != nil {
		return nil, err
	}
	fi, err := os.Stat(path)
	isManifest := false
	if errors.Is(err, os.ErrNotExist) && !explicitPath(ref) {
		n := model.ParseName(ref)
		if !n.IsValid() {
			return nil, fmt.Errorf("invalid model name %q", ref)
		}
		path = filepath.Join(store, "manifests", n.Filepath())
		fi, err = os.Stat(path)
		isManifest = true
	}
	if err != nil {
		return nil, err
	}
	if !fi.Mode().IsRegular() && !fi.IsDir() {
		return nil, fmt.Errorf("not a regular file or directory: %s", path)
	}
	inv := &inventory{
		source:   Source{Reference: ref, Path: path},
		metadata: make(map[string]any), tensors: make(map[string]*Tensor), files: make(map[string]os.FileInfo), mediaTypes: make(map[string]any),
	}
	if fi.IsDir() {
		err = inv.directory(ctx, path)
	} else {
		if !isManifest {
			f, e := os.Open(path)
			if e != nil {
				return nil, e
			}
			var prefix [64]byte
			n, e := f.Read(prefix[:])
			f.Close()
			if e != nil {
				return nil, e
			}
			trimmed := bytes.TrimSpace(prefix[:n])
			if len(trimmed) > 1 && trimmed[0] == '{' {
				rest := bytes.TrimSpace(trimmed[1:])
				isManifest = len(rest) > 0 && rest[0] == '"'
			}
			isManifest = isManifest || filepath.Ext(path) == ".json"
		}
		if isManifest {
			inv.source.Store = store
			err = inv.manifest(ctx, path, store)
		} else {
			err = inv.tensorFile(ctx, Artifact{File: path}, "model")
		}
	}
	if err != nil {
		return nil, err
	}
	if len(inv.tensors) == 0 {
		return nil, fmt.Errorf("no tensors found in %s", path)
	}
	formats := make(map[string]bool)
	for _, t := range inv.tensors {
		formats[t.Format] = true
	}
	inv.source.Format = strings.Join(unionKeys(formats, nil), "+")
	if err := inv.linkCompanions(); err != nil {
		return nil, err
	}
	return inv, nil
}

func explicitPath(s string) bool {
	return filepath.IsAbs(s) || strings.HasPrefix(s, ".") || strings.HasSuffix(s, ".gguf") || strings.HasSuffix(s, ".safetensors") || strings.HasSuffix(s, ".json")
}

func (inv *inventory) track(path string) (os.FileInfo, error) {
	fi, err := os.Stat(path)
	if err != nil {
		return nil, err
	}
	if !fi.Mode().IsRegular() && !fi.IsDir() {
		return nil, fmt.Errorf("not a regular file or directory: %s", path)
	}
	if old, ok := inv.files[path]; ok && !unchangedFile(old, fi) {
		return nil, fmt.Errorf("file changed during comparison: %s", path)
	}
	inv.files[path] = fi
	return fi, nil
}

func unchangedFile(a, b os.FileInfo) bool {
	return os.SameFile(a, b) && a.Size() == b.Size() && a.ModTime() == b.ModTime()
}

func (inv *inventory) checkFiles() error {
	for _, path := range unionKeys(inv.files, nil) {
		if _, err := inv.track(path); err != nil {
			return err
		}
	}
	return nil
}

func (inv *inventory) readMetadata(path string) ([]byte, error) {
	fi, err := inv.track(path)
	if err != nil {
		return nil, err
	}
	if fi.Size() > maxMetadataSize {
		return nil, fmt.Errorf("metadata %s exceeds %d bytes", path, maxMetadataSize)
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	data, err := io.ReadAll(io.LimitReader(f, maxMetadataSize+1))
	if err != nil {
		return nil, err
	}
	if len(data) > maxMetadataSize {
		return nil, fmt.Errorf("metadata grew beyond size limit: %s", path)
	}
	return data, nil
}

func (inv *inventory) manifest(ctx context.Context, path, store string) error {
	data, err := inv.readMetadata(path)
	if err != nil {
		return err
	}
	decoded, err := decodeJSON(data)
	if err != nil {
		return fmt.Errorf("manifest: %w", err)
	}
	var m manifest.Manifest
	if err := json.Unmarshal(data, &m); err != nil {
		return err
	}
	if m.SchemaVersion != 2 || m.Config.Digest == "" {
		return fmt.Errorf("unsupported or incomplete manifest: %s", path)
	}
	sum := sha256.Sum256(data)
	inv.source.Digest = hex.EncodeToString(sum[:])
	meta, ok := decoded.(map[string]any)
	if !ok {
		return fmt.Errorf("manifest must be an object")
	}
	delete(meta, "config")
	delete(meta, "layers")
	inv.metadata["manifest"] = meta
	seenTensorFiles := make(map[string]bool)
	for i, layer := range append([]manifest.Layer{m.Config}, m.Layers...) {
		if err := ctx.Err(); err != nil {
			return err
		}
		blob, err := blobPathAt(store, layer.Digest)
		if err != nil || layer.Digest == "" {
			return fmt.Errorf("invalid blob digest %q", layer.Digest)
		}
		fi, err := inv.track(blob)
		if err != nil {
			return err
		}
		if fi.Size() != layer.Size {
			return fmt.Errorf("blob %s: size %d, manifest expects %d", layer.Digest, fi.Size(), layer.Size)
		}
		a := Artifact{Name: layer.Name, MediaType: layer.MediaType, File: blob, Blob: strings.ToLower(strings.Replace(layer.Digest, "sha256-", "sha256:", 1)), From: layer.From, Bytes: fi.Size()}
		inv.source.Artifacts = append(inv.source.Artifacts, a)
		kind := strings.TrimPrefix(layer.MediaType, "application/vnd.ollama.image.")
		if i == 0 {
			inv.mediaTypes["manifest_config"] = layer.MediaType
			if err := inv.artifact(ctx, "manifest_config", a, "json"); err != nil {
				return err
			}
			if config, ok := inv.metadata["manifest_config"].(map[string]any); ok {
				normalizeStringSet(config, "capabilities")
				if root, ok := config["rootfs"].(map[string]any); ok {
					delete(root, "diff_ids")
				}
			}
			continue
		}
		switch kind {
		case "tensor", "model", "projector", "adapter", "draft":
			role := kind
			if role == "tensor" {
				role = "model"
			}
			key := role + "/" + blob
			if seenTensorFiles[key] {
				return fmt.Errorf("duplicate tensor blob %s in role %s", blob, role)
			}
			seenTensorFiles[key] = true
			if err := inv.tensorFile(ctx, a, role); err != nil {
				return err
			}
		case "license":
			inv.mediaTypes["license"] = layer.MediaType
			data, err := inv.readMetadata(blob)
			if err != nil {
				return err
			}
			licenses, _ := inv.metadata["license"].([]any)
			inv.metadata["license"] = append(licenses, normalizeProse(string(data)))
		default:
			key := kind
			if layer.Name != "" {
				key = layer.Name
			}
			inv.mediaTypes[key] = layer.MediaType
			if err := inv.artifact(ctx, key, a, kind); err != nil {
				return err
			}
		}
	}
	if licenses, ok := inv.metadata["license"].([]any); ok {
		slices.SortFunc(licenses, func(a, b any) int { return strings.Compare(a.(proseMetadata).text, b.(proseMetadata).text) })
	}
	if len(inv.mediaTypes) > 0 {
		inv.metadata["artifact_media_types"] = inv.mediaTypes
	}
	return nil
}

func normalizeStringSet(object map[string]any, key string) {
	values, ok := object[key].([]any)
	if !ok {
		return
	}
	for _, value := range values {
		if _, ok := value.(string); !ok {
			return
		}
	}
	slices.SortFunc(values, func(a, b any) int { return strings.Compare(a.(string), b.(string)) })
}

func blobPathAt(root, digest string) (string, error) {
	encoded := ""
	switch {
	case strings.HasPrefix(digest, "sha256:"):
		encoded = strings.TrimPrefix(digest, "sha256:")
	case strings.HasPrefix(digest, "sha256-"):
		encoded = strings.TrimPrefix(digest, "sha256-")
	}
	decoded, err := hex.DecodeString(encoded)
	if err != nil || len(decoded) != sha256.Size {
		return "", manifest.ErrInvalidDigestFormat
	}
	return filepath.Join(root, "blobs", "sha256-"+strings.ToLower(encoded)), nil
}

func (inv *inventory) artifact(ctx context.Context, key string, a Artifact, kind string) error {
	if _, exists := inv.metadata[key]; exists {
		return fmt.Errorf("duplicate metadata artifact %q", key)
	}
	var v any
	switch {
	case kind == "json" || kind == "params" || kind == "messages" || strings.HasSuffix(a.Name, ".json"):
		data, err := inv.readMetadata(a.File)
		if err != nil {
			return err
		}
		v, err = decodeJSON(data)
		if err != nil {
			return fmt.Errorf("%s: %w", key, err)
		}
	case kind == "template" || kind == "system" || kind == "license" || strings.HasSuffix(a.Name, ".txt") || strings.HasSuffix(a.Name, ".md") || strings.HasSuffix(a.Name, ".jinja") || strings.HasPrefix(strings.ToLower(filepath.Base(a.Name)), "license"):
		data, err := inv.readMetadata(a.File)
		if err != nil {
			return err
		}
		// Templates/system text and JSON string values affect tokenization.
		// Prose ignores whitespace, but meaningful prompt whitespace is exact.
		if kind == "license" || strings.HasPrefix(strings.ToLower(filepath.Base(a.Name)), "license") || strings.HasSuffix(a.Name, ".md") {
			v = normalizeProse(string(data))
		} else {
			v = string(data)
		}
	default:
		sum := strings.TrimPrefix(a.Blob, "sha256:")
		if sum == "" {
			h := payloadHasher{cache: make(map[payloadRange]string), buffer: make([]byte, 1<<20)}
			var err error
			sum, err = h.hash(ctx, a.File, 0, a.Bytes)
			if err != nil {
				return err
			}
		}
		v = map[string]any{"sha256": sum, "bytes": json.Number(fmt.Sprint(a.Bytes)), "media_type": a.MediaType}
	}
	inv.metadata[key] = v
	return nil
}
