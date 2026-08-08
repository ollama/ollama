package server

import (
	"fmt"
	"io"
	"maps"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"

	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/manifest"
)

const (
	// Bound split bookkeeping for malformed inputs. Expected split GGUFs should
	// stay well below this, and tensor data is still streamed from shard blobs.
	maxSplitGGUFParts = 1024
	maxSplitGGUFInt64 = int64(^uint64(0) >> 1)
)

var splitGGUFNameRe = regexp.MustCompile(`^(.*)-(\d{5})-of-(\d{5})\.gguf$`)

type splitGGUFPart struct {
	Digest string
	Name   string
	GGML   *ggml.GGML
}

type splitGGUFSlot struct {
	layer    *layerGGML
	splitKey string
}

type splitGGUFCollector struct {
	slots  []splitGGUFSlot
	groups map[string][]*layerGGML
}

func newSplitGGUFCollector() *splitGGUFCollector {
	return &splitGGUFCollector{
		groups: make(map[string][]*layerGGML),
	}
}

func (c *splitGGUFCollector) Add(layer *layerGGML) error {
	key, ok, err := splitGGUFGroupKey(layer)
	if err != nil {
		return invalidSplitGGUF(err)
	}
	if !ok {
		c.slots = append(c.slots, splitGGUFSlot{layer: layer})
		return nil
	}

	if _, ok := c.groups[key]; !ok {
		c.slots = append(c.slots, splitGGUFSlot{splitKey: key})
	}
	c.groups[key] = append(c.groups[key], layer)
	return nil
}

func (c *splitGGUFCollector) Layers() ([]*layerGGML, error) {
	layers := make([]*layerGGML, 0, len(c.slots))
	for _, slot := range c.slots {
		if slot.layer != nil {
			layers = append(layers, slot.layer)
			continue
		}

		layer, err := groupSplitGGUFLayers(c.groups[slot.splitKey])
		if err != nil {
			return nil, invalidSplitGGUF(err)
		}
		layers = append(layers, layer)
	}

	return layers, nil
}

type splitGGUFTensorReader struct {
	path   string
	offset int64
	size   int64
}

func (r splitGGUFTensorReader) WriteTo(w io.Writer) (int64, error) {
	f, err := os.Open(r.path)
	if err != nil {
		return 0, err
	}
	defer f.Close()

	return io.Copy(w, io.NewSectionReader(f, r.offset, r.size))
}

func invalidSplitGGUF(err error) error {
	if err == nil {
		return nil
	}
	return fmt.Errorf("%w: %w", errInvalidSplitGGUF, err)
}

func splitGGUFName(name string) (prefix string, index, count uint16, ok bool) {
	matches := splitGGUFNameRe.FindStringSubmatch(path.Base(name))
	if len(matches) != 4 {
		return "", 0, 0, false
	}

	idx, err := strconv.ParseUint(matches[2], 10, 16)
	if err != nil || idx == 0 {
		return "", 0, 0, false
	}
	n, err := strconv.ParseUint(matches[3], 10, 16)
	if err != nil || n == 0 {
		return "", 0, 0, false
	}
	return matches[1], uint16(idx - 1), uint16(n), true
}

func splitGGUFGroupKey(layer *layerGGML) (string, bool, error) {
	if layer == nil || layer.GGML == nil {
		return "", false, nil
	}

	count, ok := splitGGUFUint64(layer.GGML.KV(), "split.count")
	if !ok || count <= 1 {
		return "", false, nil
	}
	if count > maxSplitGGUFParts {
		return "", false, fmt.Errorf("split GGUF %q has too many shards: %d", layer.From, count)
	}

	prefix, index, nameCount, ok := splitGGUFName(layer.From)
	if !ok {
		return "", false, fmt.Errorf("split GGUF %q must use llama.cpp split filename pattern", layer.From)
	}
	if uint64(nameCount) != count {
		return "", false, fmt.Errorf("split GGUF %q filename count %d does not match metadata count %d", layer.From, nameCount, count)
	}
	if uint64(index) >= count {
		return "", false, fmt.Errorf("split GGUF %q filename index %d exceeds metadata count %d", layer.From, index, count)
	}
	splitNo, ok := splitGGUFUint64(layer.GGML.KV(), "split.no")
	if !ok {
		return "", false, fmt.Errorf("split GGUF %q is missing split.no metadata", layer.From)
	}
	if splitNo != uint64(index) {
		return "", false, fmt.Errorf("split GGUF %q filename index %d does not match metadata index %d", layer.From, index, splitNo)
	}

	return fmt.Sprintf("%s:%s:%d", layer.MediaType, prefix, count), true, nil
}

func splitGGUFUint64(kv ggml.KV, key string) (uint64, bool) {
	keys := []string{key}
	if !strings.HasPrefix(key, "tokenizer.") && !strings.HasPrefix(key, "general.") {
		keys = append(keys, kv.Architecture()+"."+key)
	}
	for _, k := range keys {
		switch v := kv.Value(k).(type) {
		case uint8:
			return uint64(v), true
		case uint16:
			return uint64(v), true
		case uint32:
			return uint64(v), true
		case uint64:
			return v, true
		case int8:
			if v >= 0 {
				return uint64(v), true
			}
		case int16:
			if v >= 0 {
				return uint64(v), true
			}
		case int32:
			if v >= 0 {
				return uint64(v), true
			}
		case int64:
			if v >= 0 {
				return uint64(v), true
			}
		}
	}
	return 0, false
}

func splitGGUFTensorCount(kv ggml.KV) (uint64, bool) {
	return splitGGUFUint64(kv, "split.tensors.count")
}

func groupSplitGGUFLayers(layers []*layerGGML) (*layerGGML, error) {
	if len(layers) == 0 {
		return nil, fmt.Errorf("split GGUF has no shards")
	}
	if layers[0] == nil || layers[0].GGML == nil {
		return nil, fmt.Errorf("split GGUF has an invalid shard")
	}

	_, _, count, ok := splitGGUFName(layers[0].From)
	if !ok {
		return nil, fmt.Errorf("split GGUF %q must use llama.cpp split filename pattern", layers[0].From)
	}
	if int(count) > maxSplitGGUFParts {
		return nil, fmt.Errorf("split GGUF %q has too many shards: %d", layers[0].From, count)
	}
	if len(layers) != int(count) {
		return nil, fmt.Errorf("split GGUF %q has %d shards, expected %d", layers[0].From, len(layers), count)
	}

	parts := make([]splitGGUFPart, count)
	seen := make([]bool, count)
	architecture := layers[0].GGML.KV().Architecture()
	fileType := layers[0].GGML.KV().FileType()
	mediaType := layers[0].MediaType
	var tensorCount uint64
	var hasTensorCount bool
	var primary *layerGGML

	for _, layer := range layers {
		if layer == nil || layer.GGML == nil {
			return nil, fmt.Errorf("split GGUF has an invalid shard")
		}
		if layer.MediaType != mediaType {
			return nil, fmt.Errorf("split GGUF %q media type does not match %q", layer.From, layers[0].From)
		}
		layerArchitecture := layer.GGML.KV().Architecture()
		if architecture != "unknown" && layerArchitecture != "unknown" && layerArchitecture != architecture {
			return nil, fmt.Errorf("split GGUF %q architecture does not match %q", layer.From, layers[0].From)
		}
		layerFileType := layer.GGML.KV().FileType()
		if fileType != ggml.FileTypeUnknown && layerFileType != ggml.FileTypeUnknown && layerFileType != fileType {
			return nil, fmt.Errorf("split GGUF %q file type does not match %q", layer.From, layers[0].From)
		}

		_, index, nameCount, ok := splitGGUFName(layer.From)
		if !ok {
			return nil, fmt.Errorf("split GGUF %q must use llama.cpp split filename pattern", layer.From)
		}
		if nameCount != count {
			return nil, fmt.Errorf("split GGUF %q filename count %d does not match expected count %d", layer.From, nameCount, count)
		}
		if index >= count {
			return nil, fmt.Errorf("split GGUF %q filename index %d exceeds split count %d", layer.From, index, count)
		}
		if seen[index] {
			return nil, fmt.Errorf("split GGUF %q duplicate shard index %d", layer.From, index)
		}
		seen[index] = true
		if index == 0 {
			primary = layer
		}
		if c, ok := splitGGUFTensorCount(layer.GGML.KV()); ok {
			if !hasTensorCount {
				tensorCount = c
				hasTensorCount = true
			} else if tensorCount != c {
				return nil, fmt.Errorf("split GGUF %q tensor count does not match %q", layer.From, layers[0].From)
			}
		}
		parts[index] = splitGGUFPart{
			Digest: layer.Digest,
			Name:   layer.From,
			GGML:   layer.GGML,
		}
	}

	if primary == nil {
		return nil, fmt.Errorf("split GGUF %q is missing first shard", layers[0].From)
	}
	primary.splitParts = parts
	return primary, nil
}

// Split GGUF imports are merged into a single GGUF layer so manifests remain
// loadable by older Ollama versions. A future manifest format may support
// multiple GGUF blobs for one model and retain split shards without rewriting.
func mergeSplitGGUFToLayer(layer *layerGGML) (*layerGGML, error) {
	if len(layer.splitParts) == 0 {
		return layer, nil
	}

	tensors, err := splitGGUFTensors(layer)
	if err != nil {
		return nil, invalidSplitGGUF(err)
	}

	kv := maps.Clone(layer.GGML.KV())
	removeSplitGGUFMetadata(kv, layer.GGML.KV().Architecture())

	blobDir, err := manifest.BlobsPath("")
	if err != nil {
		return nil, err
	}
	temp, err := os.CreateTemp(blobDir, "split-gguf-")
	if err != nil {
		return nil, err
	}
	defer os.Remove(temp.Name())
	defer temp.Close()

	if err := ggml.WriteGGUF(temp, kv, tensors); err != nil {
		return nil, err
	}
	if _, err := temp.Seek(0, io.SeekStart); err != nil {
		return nil, err
	}

	merged, err := manifest.NewLayer(temp, layer.MediaType)
	if err != nil {
		return nil, err
	}
	merged.From = layer.From

	if _, err := temp.Seek(0, io.SeekStart); err != nil {
		return nil, err
	}
	mergedGGML, err := ggml.Decode(temp, 1024)
	if err != nil {
		return nil, err
	}

	return &layerGGML{
		Layer: merged,
		GGML:  mergedGGML,
	}, nil
}

func splitGGUFTensors(layer *layerGGML) ([]*ggml.Tensor, error) {
	if len(layer.splitParts) > maxSplitGGUFParts {
		return nil, fmt.Errorf("split GGUF %q has too many shards: %d", layer.From, len(layer.splitParts))
	}

	var tensors []*ggml.Tensor
	seen := make(map[string]struct{})
	var expectedCount uint64
	var hasExpectedCount bool
	for _, part := range layer.splitParts {
		if part.GGML == nil {
			return nil, fmt.Errorf("split GGUF %q has an invalid shard", part.Name)
		}

		blobPath, err := manifest.BlobsPath(part.Digest)
		if err != nil {
			return nil, err
		}
		blobPath = filepath.Clean(blobPath)
		info, err := os.Stat(blobPath)
		if err != nil {
			return nil, err
		}
		if !info.Mode().IsRegular() {
			return nil, fmt.Errorf("split GGUF %q shard is not a regular file", part.Name)
		}
		fileSize := uint64(info.Size())

		if c, ok := splitGGUFTensorCount(part.GGML.KV()); ok {
			if !hasExpectedCount {
				expectedCount = c
				hasExpectedCount = true
			} else if expectedCount != c {
				return nil, fmt.Errorf("split GGUF %q tensor count does not match %q", part.Name, layer.From)
			}
		}

		for _, src := range part.GGML.Tensors().Items() {
			if _, ok := seen[src.Name]; ok {
				return nil, fmt.Errorf("split GGUF %q contains duplicate tensor %q", part.Name, src.Name)
			}
			seen[src.Name] = struct{}{}

			tensor := *src
			tensor.Shape = append([]uint64(nil), src.Shape...)
			size := tensor.Size()
			offset, ok := splitGGUFTensorOffset(part.GGML, tensor)
			if !ok || offset > uint64(maxSplitGGUFInt64) || size > uint64(maxSplitGGUFInt64) {
				return nil, fmt.Errorf("split GGUF %q tensor %q is too large", part.Name, tensor.Name)
			}
			if offset > fileSize || size > fileSize-offset {
				return nil, fmt.Errorf("split GGUF %q tensor %q extends beyond shard data", part.Name, tensor.Name)
			}
			tensor.WriterTo = splitGGUFTensorReader{
				path:   blobPath,
				offset: int64(offset),
				size:   int64(size),
			}
			tensors = append(tensors, &tensor)
		}
	}

	if hasExpectedCount && expectedCount != uint64(len(tensors)) {
		return nil, fmt.Errorf("split GGUF %q has %d tensors, expected %d", layer.From, len(tensors), expectedCount)
	}

	return tensors, nil
}

func splitGGUFTensorOffset(model *ggml.GGML, tensor ggml.Tensor) (uint64, bool) {
	base := model.Tensors().Offset
	if tensor.Offset > ^uint64(0)-base {
		return 0, false
	}
	return base + tensor.Offset, true
}

func removeSplitGGUFMetadata(kv ggml.KV, architecture string) {
	for _, key := range []string{"split.no", "split.count", "split.tensors.count"} {
		delete(kv, key)
		if architecture != "" {
			delete(kv, architecture+"."+key)
		}
	}
}
