//go:build mlx

package model

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/ollama/ollama/x/imagegen/manifest"
)

// TensorQuantInfo describes per-tensor quantization metadata.
type TensorQuantInfo struct {
	QuantType string
	GroupSize int
}

// Root wraps a ModelManifest with pre-scanned quantization metadata.
type Root struct {
	Manifest *manifest.ModelManifest

	// Backwards-compatible model-level quant metadata (first tensor blob).
	quantType string
	groupSize int

	// Per-tensor quantization metadata.
	tensorQuant map[string]*TensorQuantInfo
}

// Open loads a manifest for the given model name and scans tensor blobs for
// quantization metadata.
func Open(modelName string) (*Root, error) {
	m, err := manifest.LoadManifest(modelName)
	if err != nil {
		return nil, err
	}

	root := &Root{
		Manifest:    m,
		tensorQuant: make(map[string]*TensorQuantInfo),
	}

	for _, layer := range m.GetTensorLayers("") {
		blobPath := m.BlobPath(layer.Digest)

		infos, blobQuantType, blobGroupSize, err := readBlobTensorQuantInfo(blobPath)
		if err != nil {
			continue
		}

		for name, info := range infos {
			root.tensorQuant[name] = info
		}

		if root.quantType == "" && blobQuantType != "" {
			root.quantType = strings.ToUpper(blobQuantType)
			root.groupSize = blobGroupSize
			if root.groupSize == 0 {
				root.groupSize = defaultGroupSize(root.quantType)
			}
		}
	}

	return root, nil
}

// Close is a no-op for now (future: release resources).
func (r *Root) Close() {}

// QuantType returns the quantization type detected from the first tensor blob metadata.
func (r *Root) QuantType() string { return r.quantType }

// GroupSize returns the quantization group size detected from the first tensor blob metadata.
func (r *Root) GroupSize() int { return r.groupSize }

// TensorQuant returns per-tensor quantization metadata if available.
func (r *Root) TensorQuant(name string) *TensorQuantInfo {
	if r == nil {
		return nil
	}
	return r.tensorQuant[name]
}

// AllTensorQuant returns a copy of the per-tensor quantization metadata.
func (r *Root) AllTensorQuant() map[string]*TensorQuantInfo {
	out := make(map[string]*TensorQuantInfo, len(r.tensorQuant))
	for k, v := range r.tensorQuant {
		if v == nil {
			continue
		}
		copy := *v
		out[k] = &copy
	}
	return out
}

func defaultGroupSize(quantType string) int {
	groupSize, _, _ := QuantizationParams(quantType)
	return groupSize
}

func readBlobTensorQuantInfo(path string) (map[string]*TensorQuantInfo, string, int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, "", 0, err
	}
	defer f.Close()

	var headerSize uint64
	if err := binary.Read(f, binary.LittleEndian, &headerSize); err != nil {
		return nil, "", 0, err
	}
	if headerSize > 100*1024*1024 {
		return nil, "", 0, fmt.Errorf("header too large: %d", headerSize)
	}

	data := make([]byte, headerSize)
	if _, err := io.ReadFull(f, data); err != nil {
		return nil, "", 0, err
	}

	var header map[string]json.RawMessage
	if err := json.Unmarshal(data, &header); err != nil {
		return nil, "", 0, err
	}

	globalQuantType, globalGroupSize := parseGlobalQuantMetadata(header)
	globalQuantType = strings.ToUpper(globalQuantType)

	mainNames := mainTensorNames(header)
	infos := make(map[string]*TensorQuantInfo)
	for _, name := range mainNames {
		if _, ok := header[name+".scale"]; !ok {
			continue
		}

		quantType := globalQuantType
		groupSize := globalGroupSize

		inferredType, inferredGroup := inferQuantTypeFromShapes(header, name, quantType)
		if quantType == "" {
			quantType = inferredType
		}
		if groupSize == 0 {
			groupSize = inferredGroup
		}
		if quantType == "" {
			continue
		}
		if groupSize == 0 {
			groupSize = defaultGroupSize(quantType)
		}

		infos[name] = &TensorQuantInfo{QuantType: quantType, GroupSize: groupSize}
	}

	return infos, globalQuantType, globalGroupSize, nil
}

func parseGlobalQuantMetadata(header map[string]json.RawMessage) (quantType string, groupSize int) {
	metaRaw, ok := header["__metadata__"]
	if !ok {
		return "", 0
	}

	var meta map[string]string
	if err := json.Unmarshal(metaRaw, &meta); err != nil {
		return "", 0
	}

	quantType = meta["quant_type"]
	if gs := meta["group_size"]; gs != "" {
		groupSize, _ = strconv.Atoi(gs)
	}
	return quantType, groupSize
}

func mainTensorNames(header map[string]json.RawMessage) []string {
	names := make([]string, 0, len(header))
	for name := range header {
		if name == "__metadata__" || strings.HasSuffix(name, ".scale") || strings.HasSuffix(name, ".bias") {
			continue
		}
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

func inferQuantTypeFromShapes(header map[string]json.RawMessage, tensorName string, hintQuantType string) (string, int) {
	type tensorShape struct {
		Shape []int64 `json:"shape"`
	}

	mainRaw, ok := header[tensorName]
	if !ok {
		return "", 0
	}
	scaleRaw, ok := header[tensorName+".scale"]
	if !ok {
		return "", 0
	}

	var mainInfo tensorShape
	if err := json.Unmarshal(mainRaw, &mainInfo); err != nil || len(mainInfo.Shape) == 0 {
		return "", 0
	}

	var scaleInfo tensorShape
	if err := json.Unmarshal(scaleRaw, &scaleInfo); err != nil || len(scaleInfo.Shape) == 0 {
		return "", 0
	}

	weightCols := int(mainInfo.Shape[len(mainInfo.Shape)-1])
	scalesCols := int(scaleInfo.Shape[len(scaleInfo.Shape)-1])
	if weightCols <= 0 || scalesCols <= 0 {
		return "", 0
	}

	groupSize4 := weightCols * 8 / scalesCols
	groupSize8 := weightCols * 4 / scalesCols

	switch {
	case groupSize4 == 32:
		return "INT4", 32
	case groupSize8 == 64:
		return "INT8", 64
	case groupSize4 == 64 && groupSize8 == 32:
		h := strings.ToUpper(hintQuantType)
		if strings.Contains(h, "8") {
			return "INT8", 32
		}
		if strings.Contains(h, "4") {
			return "INT4", 64
		}
	}

	if isCommonGroupSize(groupSize4) && !isCommonGroupSize(groupSize8) {
		return "INT4", groupSize4
	}
	if isCommonGroupSize(groupSize8) && !isCommonGroupSize(groupSize4) {
		return "INT8", groupSize8
	}

	return "", 0
}
