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

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/quant"
)

// TensorQuantInfo describes per-tensor quantization metadata.
type TensorQuantInfo struct {
	QuantType string
	GroupSize int
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

	// Parse full metadata for per-tensor quant info
	var metaMap map[string]string
	if metaRaw, ok := header["__metadata__"]; ok {
		json.Unmarshal(metaRaw, &metaMap)
	}

	mainNames := mainTensorNames(header)
	infos := make(map[string]*TensorQuantInfo)
	for _, name := range mainNames {
		if _, ok := header[name+".scale"]; !ok {
			continue
		}

		quantType := globalQuantType
		groupSize := globalGroupSize

		// Check per-tensor metadata (e.g. from packed expert blobs with mixed precision)
		if metaMap != nil {
			if qt, ok := metaMap[name+".quant_type"]; ok && qt != "" {
				quantType = strings.ToUpper(qt)
			}
			if gs, ok := metaMap[name+".group_size"]; ok && gs != "" {
				if v, err := strconv.Atoi(gs); err == nil {
					groupSize = v
				}
			}
		}

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

func defaultGroupSize(quantType string) int {
	groupSize, _, _ := QuantizationParams(quantType)
	return groupSize
}

// QuantizationParams returns default groupSize, bits, and mode for a
// quantization type. The values live in the shared mlx/quant package so the
// importer, the runtime loader, and `ollama show` agree on them.
func QuantizationParams(quantization string) (groupSize, bits int, mode string) {
	return quant.Params(quantization)
}

// TensorQuantParams resolves quant params for a tensor using per-tensor metadata
// when available, otherwise falling back to the provided model defaults.
func TensorQuantParams(
	defaultGroupSize, defaultBits int,
	defaultMode string,
	tensorQuant map[string]*TensorQuantInfo,
	tensorName string,
) (groupSize, bits int, mode string, fromTensor bool) {
	if tensorQuant != nil {
		if tq := tensorQuant[tensorName]; tq != nil {
			groupSize, bits, mode = QuantizationParams(tq.QuantType)
			if tq.GroupSize > 0 {
				groupSize = tq.GroupSize
			}
			return groupSize, bits, mode, true
		}
	}
	return defaultGroupSize, defaultBits, defaultMode, false
}

// ResolveLinearQuantParams resolves quantization params for a quantized linear
// tensor, preferring per-tensor metadata and falling back to shape-based
// inference for affine packed tensors.
func ResolveLinearQuantParams(
	defaultGroupSize, defaultBits int,
	defaultMode string,
	tensorQuant map[string]*TensorQuantInfo,
	tensorName string,
	weight, scales *mlx.Array,
) (groupSize, bits int, mode string) {
	groupSize, bits, mode, fromTensor := TensorQuantParams(
		defaultGroupSize,
		defaultBits,
		defaultMode,
		tensorQuant,
		tensorName,
	)

	if mode == "affine" {
		if inferredGroupSize, inferredBits, ok := InferAffineQuantParamsFromShapes(weight, scales, bits); ok {
			if !fromTensor || groupSize == 0 || bits == 0 {
				groupSize = inferredGroupSize
				bits = inferredBits
			}
		}
	}

	return groupSize, bits, mode
}

// InferAffineQuantParamsFromShapes infers (groupSize,bits) for affine quantized
// tensors from packed weight and scale shapes.
func InferAffineQuantParamsFromShapes(weight, scales *mlx.Array, hintBits int) (groupSize, bits int, ok bool) {
	if weight == nil || scales == nil {
		return 0, 0, false
	}

	weightShape := weight.Dims()
	scaleShape := scales.Dims()
	if len(weightShape) == 0 || len(scaleShape) == 0 {
		return 0, 0, false
	}

	weightCols := weightShape[len(weightShape)-1]
	scalesCols := scaleShape[len(scaleShape)-1]
	if weightCols <= 0 || scalesCols <= 0 {
		return 0, 0, false
	}

	groupSize4 := weightCols * 8 / scalesCols
	groupSize8 := weightCols * 4 / scalesCols

	switch {
	case groupSize4 == 32:
		return 32, 4, true
	case groupSize8 == 64:
		return 64, 8, true
	case groupSize4 == 64 && groupSize8 == 32:
		if hintBits == 8 {
			return 32, 8, true
		}
		if hintBits == 4 {
			return 64, 4, true
		}
	}

	if isCommonGroupSize(groupSize4) && !isCommonGroupSize(groupSize8) {
		return groupSize4, 4, true
	}
	if isCommonGroupSize(groupSize8) && !isCommonGroupSize(groupSize4) {
		return groupSize8, 8, true
	}

	return 0, 0, false
}

func isCommonGroupSize(v int) bool {
	switch v {
	case 16, 32, 64, 128:
		return true
	default:
		return false
	}
}
