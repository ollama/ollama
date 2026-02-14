package server

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

// modelConfig represents the HuggingFace config.json structure
type modelConfig struct {
	Architectures         []string `json:"architectures"`
	ModelType             string   `json:"model_type"`
	HiddenSize            int      `json:"hidden_size"`
	NumHiddenLayers       int      `json:"num_hidden_layers"`
	MaxPositionEmbeddings int      `json:"max_position_embeddings"`
	IntermediateSize      int      `json:"intermediate_size"`
	NumAttentionHeads     int      `json:"num_attention_heads"`
	NumKeyValueHeads      int      `json:"num_key_value_heads"`
	VocabSize             int      `json:"vocab_size"`
	RMSNormEps            float64  `json:"rms_norm_eps"`
	RopeTheta             float64  `json:"rope_theta"`
	TorchDtype            string   `json:"torch_dtype"`
	TextConfig            *struct {
		HiddenSize            int `json:"hidden_size"`
		MaxPositionEmbeddings int `json:"max_position_embeddings"`
		NumHiddenLayers       int `json:"num_hidden_layers"`
	} `json:"text_config"`
}

// GetSafetensorsLLMInfo extracts model information from safetensors LLM models.
// It reads the config.json layer and returns a map compatible with GGML's KV format.
func GetSafetensorsLLMInfo(name model.Name) (map[string]any, error) {
	mf, err := manifest.ParseNamedManifest(name)
	if err != nil {
		return nil, fmt.Errorf("failed to load manifest: %w", err)
	}

	var config modelConfig
	if err := mf.ReadConfigJSON("config.json", &config); err != nil {
		return nil, fmt.Errorf("failed to read config.json: %w", err)
	}

	// Calculate total tensor bytes from manifest layers
	var totalBytes int64
	var tensorCount int64
	for _, layer := range mf.Layers {
		if layer.MediaType == manifest.MediaTypeImageTensor {
			totalBytes += layer.Size
			tensorCount++
		}
	}

	return buildModelInfo(config, totalBytes, tensorCount), nil
}

// buildModelInfo constructs the model info map from config and tensor stats.
// This is separated for testability.
func buildModelInfo(config modelConfig, totalTensorBytes, tensorCount int64) map[string]any {
	// Determine architecture
	arch := config.ModelType
	if arch == "" && len(config.Architectures) > 0 {
		// Convert HuggingFace architecture name to Ollama format
		// e.g., "Gemma3ForCausalLM" -> "gemma3"
		hfArch := config.Architectures[0]
		arch = strings.ToLower(hfArch)
		arch = strings.TrimSuffix(arch, "forcausallm")
		arch = strings.TrimSuffix(arch, "forconditionalgeneration")
	}

	// Use text_config values if they exist (for multimodal models)
	hiddenSize := config.HiddenSize
	maxPosEmbed := config.MaxPositionEmbeddings
	numLayers := config.NumHiddenLayers

	if config.TextConfig != nil {
		if config.TextConfig.HiddenSize > 0 {
			hiddenSize = config.TextConfig.HiddenSize
		}
		if config.TextConfig.MaxPositionEmbeddings > 0 {
			maxPosEmbed = config.TextConfig.MaxPositionEmbeddings
		}
		if config.TextConfig.NumHiddenLayers > 0 {
			numLayers = config.TextConfig.NumHiddenLayers
		}
	}

	// Get dtype to determine bytes per parameter for count calculation
	dtype := config.TorchDtype

	// Determine bytes per parameter based on dtype
	var bytesPerParam int64 = 2 // default to float16/bfloat16
	switch strings.ToLower(dtype) {
	case "float32":
		bytesPerParam = 4
	case "float16", "bfloat16":
		bytesPerParam = 2
	case "int8", "uint8":
		bytesPerParam = 1
	}

	// Subtract safetensors header overhead per tensor blob.
	// Headers include __metadata__ with the tensor name, so overhead is ~150 bytes on average.
	totalBytes := totalTensorBytes - tensorCount*150

	paramCount := totalBytes / bytesPerParam

	info := map[string]any{
		"general.architecture": arch,
	}

	if maxPosEmbed > 0 {
		info[fmt.Sprintf("%s.context_length", arch)] = maxPosEmbed
	}

	if hiddenSize > 0 {
		info[fmt.Sprintf("%s.embedding_length", arch)] = hiddenSize
	}

	if numLayers > 0 {
		info[fmt.Sprintf("%s.block_count", arch)] = numLayers
	}

	if config.NumAttentionHeads > 0 {
		info[fmt.Sprintf("%s.attention.head_count", arch)] = config.NumAttentionHeads
	}

	if config.NumKeyValueHeads > 0 {
		info[fmt.Sprintf("%s.attention.head_count_kv", arch)] = config.NumKeyValueHeads
	}

	if config.IntermediateSize > 0 {
		info[fmt.Sprintf("%s.feed_forward_length", arch)] = config.IntermediateSize
	}

	if config.VocabSize > 0 {
		info[fmt.Sprintf("%s.vocab_size", arch)] = config.VocabSize
	}

	if paramCount > 0 {
		info["general.parameter_count"] = paramCount
	}

	return info
}

// GetSafetensorsTensorInfo extracts tensor information from safetensors model layers.
// Each tensor is stored as a minimal safetensors file with an 88-byte header containing metadata.
func GetSafetensorsTensorInfo(name model.Name) ([]api.Tensor, error) {
	mf, err := manifest.ParseNamedManifest(name)
	if err != nil {
		return nil, fmt.Errorf("failed to load manifest: %w", err)
	}

	return getTensorInfoFromManifest(mf)
}

// getTensorInfoFromManifest extracts tensor info from a manifest.
// This is separated for testability.
// For quantized tensors, reads quant_type from blob __metadata__.
// For packed blobs (multiple tensors per blob), enumerates all tensors in the blob.
func getTensorInfoFromManifest(mf *manifest.Manifest) ([]api.Tensor, error) {
	var tensors []api.Tensor

	for _, layer := range mf.Layers {
		if layer.MediaType != manifest.MediaTypeImageTensor {
			continue
		}

		// Read all tensor entries from the safetensors header
		blobPath, err := manifest.BlobsPath(layer.Digest)
		if err != nil {
			continue
		}

		f, err := os.Open(blobPath)
		if err != nil {
			continue
		}

		allInfos, err := parseSafetensorsAllHeaders(f)
		f.Close()
		if err != nil {
			continue
		}

		// Determine if this is a packed blob (multiple main tensors)
		isPacked := len(allInfos) > 1

		for _, info := range allInfos {
			tensorName := layer.Name
			if isPacked {
				// For packed blobs, use the tensor name from the header
				tensorName = info.Name
			}

			if info.QuantType != "" {
				quantType := strings.ToUpper(info.QuantType)

				shape := make([]uint64, len(info.Shape))
				for i, s := range info.Shape {
					shape[i] = uint64(s)
				}

				var packFactor int64
				switch strings.ToLower(info.QuantType) {
				case "int4", "nvfp4":
					packFactor = 8
				case "int8", "mxfp8":
					packFactor = 4
				}
				if packFactor > 0 && len(shape) >= 2 {
					shape[len(shape)-1] = uint64(info.Shape[len(info.Shape)-1] * packFactor)
				}

				tensors = append(tensors, api.Tensor{
					Name:  tensorName,
					Type:  quantType,
					Shape: shape,
				})
			} else {
				shape := make([]uint64, len(info.Shape))
				for i, s := range info.Shape {
					shape[i] = uint64(s)
				}

				tensors = append(tensors, api.Tensor{
					Name:  tensorName,
					Type:  info.Dtype,
					Shape: shape,
				})
			}
		}
	}

	sort.Slice(tensors, func(i, j int) bool {
		return tensors[i].Name < tensors[j].Name
	})

	return tensors, nil
}

// GetSafetensorsDtype returns the quantization type for a safetensors model.
// Reads quant_type from the first tensor blob's __metadata__.
// Falls back to torch_dtype from config.json if no quant metadata.
func GetSafetensorsDtype(name model.Name) (string, error) {
	mf, err := manifest.ParseNamedManifest(name)
	if err != nil {
		return "", fmt.Errorf("failed to load manifest: %w", err)
	}

	// Check first tensor blob for quant_type metadata
	for _, layer := range mf.Layers {
		if layer.MediaType != manifest.MediaTypeImageTensor {
			continue
		}
		blobPath, err := manifest.BlobsPath(layer.Digest)
		if err != nil {
			continue
		}
		info, err := readSafetensorsHeader(blobPath)
		if err != nil {
			continue
		}
		if info.QuantType != "" {
			return strings.ToUpper(info.QuantType), nil
		}
		// Only check the first tensor blob
		break
	}

	// Not quantized - return torch_dtype from config.json
	var cfg struct {
		TorchDtype string `json:"torch_dtype"`
	}
	if err := mf.ReadConfigJSON("config.json", &cfg); err != nil {
		return "", fmt.Errorf("failed to read config.json: %w", err)
	}

	return cfg.TorchDtype, nil
}

// safetensorsTensorInfo holds metadata about a tensor from a safetensors header
type safetensorsTensorInfo struct {
	Name      string  // tensor name from the header key
	Dtype     string  `json:"dtype"`
	Shape     []int64 `json:"shape"`
	QuantType string  // from __metadata__.quant_type (e.g., "int4", "int8", "nvfp4", "mxfp8")
	GroupSize string  // from __metadata__.group_size (e.g., "32", "64")
}

// readSafetensorsHeader reads the JSON header from a safetensors file to get tensor metadata.
// Safetensors format: 8-byte header size (little endian) + JSON header + tensor data
func readSafetensorsHeader(path string) (*safetensorsTensorInfo, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	return parseSafetensorsHeader(f)
}

// parseSafetensorsHeader parses a safetensors header from a reader.
// This is separated for testability.
// Parses __metadata__ for quant_type and group_size if present.
func parseSafetensorsHeader(r io.Reader) (*safetensorsTensorInfo, error) {
	// Read header size (8 bytes, little endian)
	var headerSize uint64
	if err := binary.Read(r, binary.LittleEndian, &headerSize); err != nil {
		return nil, fmt.Errorf("failed to read header size: %w", err)
	}

	// Sanity check - header shouldn't be too large
	if headerSize > 1024*1024 {
		return nil, fmt.Errorf("header size too large: %d", headerSize)
	}

	// Read header JSON
	headerBytes := make([]byte, headerSize)
	if _, err := io.ReadFull(r, headerBytes); err != nil {
		return nil, fmt.Errorf("failed to read header: %w", err)
	}

	// Parse as map of tensor name -> info
	var header map[string]json.RawMessage
	if err := json.Unmarshal(headerBytes, &header); err != nil {
		return nil, fmt.Errorf("failed to parse header: %w", err)
	}

	// Parse metadata if present
	var quantType, groupSize string
	if metaRaw, ok := header["__metadata__"]; ok {
		var meta map[string]string
		if json.Unmarshal(metaRaw, &meta) == nil {
			quantType = meta["quant_type"]
			groupSize = meta["group_size"]
		}
	}

	// Find the main tensor entry (not __metadata__, .scale, or .bias)
	for name, raw := range header {
		if name == "__metadata__" || strings.HasSuffix(name, ".scale") || strings.HasSuffix(name, ".bias") {
			continue
		}
		var info safetensorsTensorInfo
		if err := json.Unmarshal(raw, &info); err != nil {
			return nil, fmt.Errorf("failed to parse tensor info: %w", err)
		}
		info.QuantType = quantType
		info.GroupSize = groupSize
		return &info, nil
	}

	// Fall back to first non-metadata tensor entry
	for name, raw := range header {
		if name == "__metadata__" {
			continue
		}
		var info safetensorsTensorInfo
		if err := json.Unmarshal(raw, &info); err != nil {
			return nil, fmt.Errorf("failed to parse tensor info: %w", err)
		}
		info.QuantType = quantType
		info.GroupSize = groupSize
		return &info, nil
	}

	return nil, fmt.Errorf("no tensor found in header")
}

// parseSafetensorsAllHeaders parses all tensor entries from a safetensors header.
// Returns one safetensorsTensorInfo per main tensor (skipping __metadata__, .scale, .bias).
// For packed blobs this returns multiple entries; for single-tensor blobs, one entry.
// Each tensor's quant type is inferred from its shape and the presence of .scale/.bias entries
// when no global __metadata__ quant_type is present.
func parseSafetensorsAllHeaders(r io.Reader) ([]safetensorsTensorInfo, error) {
	var headerSize uint64
	if err := binary.Read(r, binary.LittleEndian, &headerSize); err != nil {
		return nil, fmt.Errorf("failed to read header size: %w", err)
	}

	if headerSize > 100*1024*1024 { // 100MB limit for packed blob headers
		return nil, fmt.Errorf("header size too large: %d", headerSize)
	}

	headerBytes := make([]byte, headerSize)
	if _, err := io.ReadFull(r, headerBytes); err != nil {
		return nil, fmt.Errorf("failed to read header: %w", err)
	}

	var header map[string]json.RawMessage
	if err := json.Unmarshal(headerBytes, &header); err != nil {
		return nil, fmt.Errorf("failed to parse header: %w", err)
	}

	// Parse global metadata if present
	var globalQuantType, globalGroupSize string
	if metaRaw, ok := header["__metadata__"]; ok {
		var meta map[string]string
		if json.Unmarshal(metaRaw, &meta) == nil {
			globalQuantType = meta["quant_type"]
			globalGroupSize = meta["group_size"]
		}
	}

	// Build a set of all keys for checking .scale/.bias presence
	headerKeys := make(map[string]bool, len(header))
	for k := range header {
		headerKeys[k] = true
	}

	// Collect all main tensor entries (sorted for deterministic output)
	var mainNames []string
	for name := range header {
		if name == "__metadata__" || strings.HasSuffix(name, ".scale") || strings.HasSuffix(name, ".bias") {
			continue
		}
		mainNames = append(mainNames, name)
	}
	sort.Strings(mainNames)

	var results []safetensorsTensorInfo
	for _, name := range mainNames {
		var info safetensorsTensorInfo
		if err := json.Unmarshal(header[name], &info); err != nil {
			return nil, fmt.Errorf("failed to parse tensor info for %s: %w", name, err)
		}
		info.Name = name

		if globalQuantType != "" {
			// Use global metadata
			info.QuantType = globalQuantType
			info.GroupSize = globalGroupSize
		} else if headerKeys[name+".scale"] {
			// No global metadata, but has .scale - infer quant type from shape
			info.QuantType = inferQuantType(header, name)
		}

		results = append(results, info)
	}

	if len(results) == 0 {
		return nil, fmt.Errorf("no tensor found in header")
	}

	return results, nil
}

// inferQuantType infers the quantization type for a tensor from its shape and scale shape.
// Returns "int4", "int8", etc. or "" if not quantized.
func inferQuantType(header map[string]json.RawMessage, name string) string {
	// Parse the main tensor shape
	var mainInfo struct {
		Shape []int64 `json:"shape"`
	}
	if json.Unmarshal(header[name], &mainInfo) != nil || len(mainInfo.Shape) < 2 {
		return ""
	}

	// Parse scale shape to determine group size
	scaleRaw, ok := header[name+".scale"]
	if !ok {
		return ""
	}
	var scaleInfo struct {
		Shape []int64 `json:"shape"`
	}
	if json.Unmarshal(scaleRaw, &scaleInfo) != nil || len(scaleInfo.Shape) < 2 {
		return ""
	}

	// Calculate group size: main_cols * pack_factor / scale_cols
	// Main dtype is U32, so we need to figure out the pack factor
	// For int4: pack=8, group=32. scale_cols = original_cols / 32 = main_cols * 8 / 32 = main_cols / 4
	// For int8: pack=4, group=64. scale_cols = original_cols / 64 = main_cols * 4 / 64 = main_cols / 16
	mainCols := mainInfo.Shape[len(mainInfo.Shape)-1]
	scaleCols := scaleInfo.Shape[len(scaleInfo.Shape)-1]
	if scaleCols == 0 {
		return ""
	}

	ratio := mainCols / scaleCols // main_packed_cols / scale_cols
	// int4: ratio = (orig/8) / (orig/32) = 32/8 = 4
	// int8: ratio = (orig/4) / (orig/64) = 64/4 = 16
	switch ratio {
	case 4:
		return "int4"
	case 16:
		return "int8"
	default:
		return ""
	}
}
