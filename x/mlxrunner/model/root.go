//go:build mlx

package model

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/ollama/ollama/x/imagegen/manifest"
)

// Root wraps a ModelManifest with pre-scanned quantization metadata.
type Root struct {
	Manifest  *manifest.ModelManifest
	quantType string
	groupSize int
}

// Open loads a manifest for the given model name and pre-scans the first
// tensor blob for quantization metadata (quant_type, group_size).
func Open(modelName string) (*Root, error) {
	m, err := manifest.LoadManifest(modelName)
	if err != nil {
		return nil, err
	}

	root := &Root{Manifest: m}

	// Pre-scan first tensor blob for quantization metadata
	for _, layer := range m.GetTensorLayers("") {
		blobPath := m.BlobPath(layer.Digest)
		meta, err := readBlobMetadata(blobPath)
		if err != nil || meta == nil {
			continue
		}
		if qt := meta["quant_type"]; qt != "" {
			root.quantType = strings.ToUpper(qt)
		}
		if gs := meta["group_size"]; gs != "" {
			fmt.Sscanf(gs, "%d", &root.groupSize)
		}
		break // only check the first tensor blob
	}

	return root, nil
}

// Close is a no-op for now (future: release resources).
func (r *Root) Close() {}

// QuantType returns the quantization type detected from tensor metadata.
func (r *Root) QuantType() string { return r.quantType }

// GroupSize returns the quantization group size detected from tensor metadata.
func (r *Root) GroupSize() int { return r.groupSize }

// readBlobMetadata reads the __metadata__ from a safetensors blob header.
func readBlobMetadata(path string) (map[string]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var headerSize uint64
	if err := binary.Read(f, binary.LittleEndian, &headerSize); err != nil {
		return nil, err
	}
	if headerSize > 1024*1024 {
		return nil, fmt.Errorf("header too large: %d", headerSize)
	}

	data := make([]byte, headerSize)
	if _, err := io.ReadFull(f, data); err != nil {
		return nil, err
	}

	var header map[string]json.RawMessage
	if err := json.Unmarshal(data, &header); err != nil {
		return nil, err
	}

	metaRaw, ok := header["__metadata__"]
	if !ok {
		return nil, nil
	}

	var meta map[string]string
	if err := json.Unmarshal(metaRaw, &meta); err != nil {
		return nil, err
	}
	return meta, nil
}
