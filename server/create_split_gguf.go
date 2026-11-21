package server

import (
	"fmt"
	"math"
	"path"
	"strings"

	"github.com/ollama/ollama/fs/gguf"
	"github.com/ollama/ollama/manifest"
)

const (
	maxSplitGGUFParts         = maxCreateFiles
	splitGGUFMinOllamaVersion = "0.35.0"
)

type splitGGUFSlot struct {
	layer    *modelLayer
	splitKey splitGGUFKey
}

type splitGGUFKey struct {
	mediaType string
	dir       string
	prefix    string
	count     uint16
}

type splitGGUFCollector struct {
	slots  []splitGGUFSlot
	groups map[splitGGUFKey][]*modelLayer
}

func newSplitGGUFCollector() *splitGGUFCollector {
	return &splitGGUFCollector{
		groups: make(map[splitGGUFKey][]*modelLayer),
	}
}

func (c *splitGGUFCollector) Add(layer *modelLayer) error {
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

func (c *splitGGUFCollector) Layers() ([]*modelLayer, error) {
	layers := make([]*modelLayer, 0, len(c.slots))
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

func invalidSplitGGUF(err error) error {
	if err == nil {
		return nil
	}
	return fmt.Errorf("%w: %w", errInvalidSplitGGUF, err)
}

// splitGGUFName parses a llama.cpp split GGUF shard filename, returning the
// shard prefix, zero-based shard index, and total shard count. Shard numbers
// that do not fit in a uint16 are rejected, matching historical behavior.
func splitGGUFName(name string) (prefix string, index, count uint16, ok bool) {
	p, i, c, ok := gguf.SplitShardName(name)
	if !ok || i > math.MaxUint16 || c > math.MaxUint16 {
		return "", 0, 0, false
	}
	return p, uint16(i), uint16(c), true
}

func splitGGUFGroupKey(layer *modelLayer) (splitGGUFKey, bool, error) {
	if layer == nil || layer.GGUF == nil {
		return splitGGUFKey{}, false, nil
	}
	count, ok := layer.GGUF.UintOK("split.count")
	if !ok || count <= 1 {
		return splitGGUFKey{}, false, nil
	}
	if count > maxSplitGGUFParts {
		return splitGGUFKey{}, false, fmt.Errorf("split GGUF %q has too many shards: %d", layer.From, count)
	}

	filename := layer.splitFile
	if filename == "" {
		filename = layer.From
	}
	prefix, index, nameCount, ok := splitGGUFName(filename)
	if !ok {
		return splitGGUFKey{}, false, fmt.Errorf("split GGUF %q must use llama.cpp split filename pattern", filename)
	}
	if uint64(nameCount) != count {
		return splitGGUFKey{}, false, fmt.Errorf("split GGUF %q filename count %d does not match metadata count %d", filename, nameCount, count)
	}
	if uint64(index) >= count {
		return splitGGUFKey{}, false, fmt.Errorf("split GGUF %q filename index %d exceeds metadata count %d", filename, index, count)
	}
	splitNo, ok := layer.GGUF.UintOK("split.no")
	if !ok {
		return splitGGUFKey{}, false, fmt.Errorf("split GGUF %q is missing split.no metadata", filename)
	}
	if splitNo != uint64(index) {
		return splitGGUFKey{}, false, fmt.Errorf("split GGUF %q filename index %d does not match metadata index %d", filename, index, splitNo)
	}

	return splitGGUFKey{
		mediaType: layer.MediaType,
		dir:       path.Dir(filename),
		prefix:    prefix,
		count:     nameCount,
	}, true, nil
}

func groupManifestSplitGGUFLayers(layers []*modelLayer) ([]*modelLayer, error) {
	grouped := make([]*modelLayer, 0, len(layers))
	for i := 0; i < len(layers); {
		layer := layers[i]
		if layer == nil || layer.GGUF == nil {
			grouped = append(grouped, layer)
			i++
			continue
		}

		count, ok := layer.GGUF.UintOK("split.count")
		if !ok || count <= 1 {
			grouped = append(grouped, layer)
			i++
			continue
		}
		if count > uint64(len(layers)-i) {
			return nil, invalidSplitGGUF(fmt.Errorf("split GGUF has %d remaining shards, expected %d", len(layers)-i, count))
		}
		logicalLayer, err := groupSplitGGUFLayers(layers[i : i+int(count)])
		if err != nil {
			return nil, invalidSplitGGUF(err)
		}
		grouped = append(grouped, logicalLayer)
		i += int(count)
	}
	return grouped, nil
}

// missingSplitGGUFShards names the shards absent from a split GGUF group, e.g.
// " (missing model-00002-of-00002.gguf)". It returns an empty string when the
// missing shards cannot be derived from the shard filenames.
func missingSplitGGUFShards(layers []*modelLayer, count uint16) string {
	if len(layers) == 0 || layers[0] == nil {
		return ""
	}
	filename := layers[0].splitFile
	if filename == "" {
		filename = layers[0].From
	}
	prefix, _, nameCount, ok := splitGGUFName(filename)
	if !ok || nameCount != count {
		return ""
	}

	seen := make([]bool, count)
	for _, layer := range layers {
		if layer == nil {
			continue
		}
		name := layer.splitFile
		if name == "" {
			name = layer.From
		}
		if _, index, _, ok := splitGGUFName(name); ok && index < count {
			seen[index] = true
		}
	}

	var missing []string
	for i := range count {
		if !seen[i] {
			missing = append(missing, gguf.SplitShardFileName(prefix, int(i), int(count)))
		}
	}
	if len(missing) == 0 {
		return ""
	}
	return fmt.Sprintf(" (missing %s)", strings.Join(missing, ", "))
}

func groupSplitGGUFLayers(layers []*modelLayer) (*modelLayer, error) {
	if len(layers) == 0 {
		return nil, fmt.Errorf("split GGUF has no shards")
	}
	if layers[0] == nil || layers[0].GGUF == nil {
		return nil, fmt.Errorf("split GGUF has an invalid shard")
	}

	countValue, ok := layers[0].GGUF.UintOK("split.count")
	if !ok || countValue <= 1 {
		return nil, fmt.Errorf("split GGUF has invalid shard count")
	}
	if countValue > maxSplitGGUFParts {
		return nil, fmt.Errorf("split GGUF %q has too many shards: %d", layers[0].From, countValue)
	}
	count := uint16(countValue)
	if len(layers) != int(count) {
		return nil, fmt.Errorf("split GGUF %q has %d shards, expected %d%s", layers[0].From, len(layers), count, missingSplitGGUFShards(layers, count))
	}

	parts := make([]manifest.Layer, count)
	seen := make([]bool, count)
	architecture := layers[0].GGUF.Architecture()
	fileType := layers[0].GGUF.FileType()
	mediaType := layers[0].MediaType
	if mediaType != "application/vnd.ollama.image.model" && mediaType != manifest.MediaTypeImageDraft {
		return nil, fmt.Errorf("split GGUF %q has unsupported media type %q", layers[0].From, mediaType)
	}
	var expectedTensors uint64
	var hasExpectedTensors bool
	var actualTensors, parameterCount uint64
	var primary *modelLayer
	tensorNames := make(map[string]struct{})

	for _, layer := range layers {
		if layer == nil || layer.GGUF == nil {
			return nil, fmt.Errorf("split GGUF has an invalid shard")
		}
		if layer.MediaType != mediaType {
			return nil, fmt.Errorf("split GGUF %q media type does not match %q", layer.From, layers[0].From)
		}
		layerCount, ok := layer.GGUF.UintOK("split.count")
		if !ok || layerCount != uint64(count) {
			return nil, fmt.Errorf("split GGUF %q shard count does not match %q", layer.From, layers[0].From)
		}
		indexValue, ok := layer.GGUF.UintOK("split.no")
		if !ok || indexValue >= uint64(count) {
			return nil, fmt.Errorf("split GGUF %q has invalid shard index", layer.From)
		}
		layerArchitecture := layer.GGUF.Architecture()
		if architecture != "unknown" && layerArchitecture != "unknown" && layerArchitecture != architecture {
			return nil, fmt.Errorf("split GGUF %q architecture does not match %q", layer.From, layers[0].From)
		}
		layerFileType := layer.GGUF.FileType()
		if fileType != gguf.FileTypeUnknown && layerFileType != gguf.FileTypeUnknown && layerFileType != fileType {
			return nil, fmt.Errorf("split GGUF %q file type does not match %q", layer.From, layers[0].From)
		}

		index := uint16(indexValue)
		if layer.splitFile != "" {
			_, filenameIndex, filenameCount, ok := splitGGUFName(layer.splitFile)
			if !ok {
				return nil, fmt.Errorf("split GGUF %q must use llama.cpp split filename pattern", layer.splitFile)
			}
			if filenameCount != count {
				return nil, fmt.Errorf("split GGUF %q filename count %d does not match expected count %d", layer.splitFile, filenameCount, count)
			}
			if filenameIndex != index {
				return nil, fmt.Errorf("split GGUF %q filename index %d does not match metadata index %d", layer.splitFile, filenameIndex, index)
			}
		}
		if seen[index] {
			return nil, fmt.Errorf("split GGUF %q duplicate shard index %d", layer.From, index)
		}
		seen[index] = true
		if index == 0 {
			primary = layer
		}
		if layerTensorCount, ok := layer.GGUF.UintOK("split.tensors.count"); ok {
			if !hasExpectedTensors {
				expectedTensors = layerTensorCount
				hasExpectedTensors = true
			} else if expectedTensors != layerTensorCount {
				return nil, fmt.Errorf("split GGUF %q tensor count does not match %q", layer.From, layers[0].From)
			}
		}
		numTensors := uint64(layer.GGUF.NumTensors())
		if numTensors > ^uint64(0)-actualTensors {
			return nil, fmt.Errorf("split GGUF %q tensor count overflows", layer.From)
		}
		actualTensors += numTensors
		for _, tensor := range layer.GGUF.TensorInfos() {
			if _, ok := tensorNames[tensor.Name]; ok {
				return nil, fmt.Errorf("split GGUF %q contains duplicate tensor %q", layer.From, tensor.Name)
			}
			tensorNames[tensor.Name] = struct{}{}
		}
		if layer.parameterCount > ^uint64(0)-parameterCount {
			return nil, fmt.Errorf("split GGUF %q parameter count overflows", layer.From)
		}
		parameterCount += layer.parameterCount
		parts[index] = layer.Layer
	}

	if primary == nil {
		return nil, fmt.Errorf("split GGUF %q is missing first shard", layers[0].From)
	}
	if hasExpectedTensors && expectedTensors != actualTensors {
		return nil, fmt.Errorf("split GGUF %q has %d tensors, expected %d", primary.From, actualTensors, expectedTensors)
	}
	primary.parameterCount = parameterCount
	primary.splitLayers = parts
	return primary, nil
}
