package server

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
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

	// Subtract safetensors header overhead (88 bytes per tensor file)
	// Each tensor is stored as a minimal safetensors file
	totalBytes := totalTensorBytes - tensorCount*88

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
// For quantized models, groups weight/scale/qbias into single entries with detected quantization type.
func getTensorInfoFromManifest(mf *manifest.Manifest) ([]api.Tensor, error) {
	var tensors []api.Tensor

	// First pass: collect all tensor info and identify scale tensors
	type tensorData struct {
		info   *safetensorsTensorInfo
		digest string
	}
	tensorMap := make(map[string]*tensorData)
	scaleMap := make(map[string]*tensorData) // base name -> scale tensor info

	for _, layer := range mf.Layers {
		if layer.MediaType != manifest.MediaTypeImageTensor {
			continue
		}

		// Read the safetensors header from the blob
		blobPath, err := manifest.BlobsPath(layer.Digest)
		if err != nil {
			continue
		}
		info, err := readSafetensorsHeader(blobPath)
		if err != nil {
			continue
		}

		td := &tensorData{info: info, digest: layer.Digest}

		if strings.HasSuffix(layer.Name, "_scale") {
			baseName := strings.TrimSuffix(layer.Name, "_scale")
			scaleMap[baseName] = td
		} else if strings.HasSuffix(layer.Name, "_qbias") {
			// Skip qbias tensors - they're included with the quantized weight
			continue
		} else {
			tensorMap[layer.Name] = td
		}
	}

	// Second pass: build tensor list with quantization info
	for _, layer := range mf.Layers {
		if layer.MediaType != manifest.MediaTypeImageTensor {
			continue
		}

		// Skip scale and qbias tensors
		if strings.HasSuffix(layer.Name, "_scale") || strings.HasSuffix(layer.Name, "_qbias") {
			continue
		}

		td := tensorMap[layer.Name]
		if td == nil {
			continue
		}

		// Check if this tensor has a corresponding scale tensor (quantized)
		scaleTd := scaleMap[layer.Name]
		if scaleTd != nil && len(td.info.Shape) >= 2 && len(scaleTd.info.Shape) >= 2 {
			// Quantized tensor - detect bits from shapes
			weightCols := td.info.Shape[len(td.info.Shape)-1]
			scaleCols := scaleTd.info.Shape[len(scaleTd.info.Shape)-1]

			// Detect quantization: Q4 has pack_factor=8, Q8 has pack_factor=4
			// Q4 uses group_size=32: weightCols * 8 / scaleCols = 32
			// Q8 uses group_size=64: weightCols * 4 / scaleCols = 64
			var bits int
			var quantType string
			if weightCols*8/scaleCols == 32 {
				bits = 4
				quantType = "Q4"
			} else if weightCols*4/scaleCols == 64 {
				bits = 8
				quantType = "Q8"
			} else {
				// Unknown quantization, show raw
				quantType = td.info.Dtype
			}

			// Calculate unpacked shape
			shape := make([]uint64, len(td.info.Shape))
			for i, s := range td.info.Shape {
				shape[i] = uint64(s)
			}
			if bits > 0 {
				packFactor := int64(32 / bits)
				shape[len(shape)-1] = uint64(td.info.Shape[len(td.info.Shape)-1] * packFactor)
			}

			tensors = append(tensors, api.Tensor{
				Name:  layer.Name,
				Type:  quantType,
				Shape: shape,
			})
		} else {
			// Non-quantized tensor
			shape := make([]uint64, len(td.info.Shape))
			for i, s := range td.info.Shape {
				shape[i] = uint64(s)
			}

			tensors = append(tensors, api.Tensor{
				Name:  layer.Name,
				Type:  td.info.Dtype,
				Shape: shape,
			})
		}
	}

	return tensors, nil
}

// GetSafetensorsDtype returns the quantization type for a safetensors model.
// Reads from model_index.json first, falls back to detection from tensor names.
// Otherwise returns the torch_dtype from config.json.
func GetSafetensorsDtype(name model.Name) (string, error) {
	mf, err := manifest.ParseNamedManifest(name)
	if err != nil {
		return "", fmt.Errorf("failed to load manifest: %w", err)
	}

	// First try to read quantization from model_index.json
	var modelIndex struct {
		Quantization string `json:"quantization"`
	}
	if err := mf.ReadConfigJSON("model_index.json", &modelIndex); err == nil && modelIndex.Quantization != "" {
		return modelIndex.Quantization, nil
	}

	// Fallback: detect from tensor names
	hasScales := false
	hasQBias := false
	for _, layer := range mf.Layers {
		if layer.MediaType == manifest.MediaTypeImageTensor {
			if strings.HasSuffix(layer.Name, "_scale") {
				hasScales = true
			}
			if strings.HasSuffix(layer.Name, "_qbias") {
				hasQBias = true
			}
		}
	}

	if hasScales {
		if hasQBias {
			// Affine mode (has scale + qbias) - could be Q4 or Q8
			// Default to Q4 as it's more common
			return "Q4", nil
		}
		// No qbias = NVFP4
		return "NVFP4", nil
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
	Dtype string  `json:"dtype"`
	Shape []int64 `json:"shape"`
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

	// Find the first (and should be only) tensor entry
	for name, raw := range header {
		if name == "__metadata__" {
			continue
		}
		var info safetensorsTensorInfo
		if err := json.Unmarshal(raw, &info); err != nil {
			return nil, fmt.Errorf("failed to parse tensor info: %w", err)
		}
		return &info, nil
	}

	return nil, fmt.Errorf("no tensor found in header")
}
