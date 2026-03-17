package base

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"sync"

	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/tokenizer"
)

// Model is the interface that model implementations must satisfy.
type Model interface {
	Forward(inputs *mlx.Array, cache []cache.Cache) *mlx.Array
	Unembed(x *mlx.Array) *mlx.Array
	NumLayers() int
	Tokenizer() *tokenizer.Tokenizer
	MaxContextLength() int

	// LoadWeights receives all tensors loaded from the manifest and assigns
	// them to model fields. Model-specific logic (MLA absorption, expert
	// stacking, quantized layer creation) happens here.
	LoadWeights(tensors map[string]*mlx.Array) error
}

// MultimodalModel extends Model with multimodal capabilities (vision, audio, etc.).
//
// Each model owns its full multimodal pipeline: decoding, preprocessing,
// encoding, and embedding construction.
//
// The interface is modality-agnostic: EncodeMultimodal accepts raw bytes
// (JPEG/PNG for images, WAV/PCM for audio, etc.) and the model determines
// what to do based on its own configuration. For models that support
// multiple modalities (e.g., models with both vision and audio), the
// model can inspect the data format or use separate placeholder token IDs
// to distinguish modalities.
type MultimodalModel interface {
	Model

	// EncodeMultimodal decodes raw bytes (image, audio, etc.) and preprocesses
	// them into a tensor ready for the model's encoder. The model inspects
	// the data to determine its modality (e.g., JPEG magic bytes → image,
	// WAV header → audio) and returns the appropriate placeholder token ID.
	//
	// Returns:
	//   - data: preprocessed tensor ready for the model's encoder
	//   - placeholderID: token ID to use as placeholder (e.g., image_token_id
	//     or audio_token_id — each modality may use a different ID)
	//   - numTokens: number of soft tokens this input will produce
	//   - err: any error during decoding or preprocessing
	EncodeMultimodal(data []byte) (EncodedMultimodal, error)

	// Prefill runs multimodal encoding and the embedding-based forward pass
	// for the prefill phase. The model handles embedding construction,
	// soft token replacement, and any model-specific logic (e.g.,
	// spatial position embeddings for vision, etc.).
	//
	// tokens: full remaining token sequence (after cache prefix removal).
	// segments: describes where placeholder tokens are and their encoded data.
	// caches: KV caches for each layer.
	// chunkSize: maximum number of tokens to process per forward call.
	// materializeCaches: callback to evaluate cache state between chunks.
	//
	// Returns the number of tokens processed (all but the last token).
	Prefill(tokens []int32, segments []MultimodalSegment, caches []cache.Cache, chunkSize int, materializeCaches func()) (int, error)
}

// EncodedMultimodal is the result of preprocessing a single multimodal input.
type EncodedMultimodal struct {
	Data          *mlx.Array // preprocessed tensor ready for the encoder
	PlaceholderID int32      // token ID to insert as placeholder (model-determined)
	NumTokens     int        // number of soft tokens this input will produce
	PrefixTokens  []int32    // tokens to insert before the placeholder run (e.g., begin-of-audio)
	SuffixTokens  []int32    // tokens to insert after the placeholder run (e.g., end-of-audio)
}

// MultimodalSegment describes where a multimodal input's placeholder tokens
// are in the token sequence and holds the preprocessed data for encoding.
type MultimodalSegment struct {
	Data      *mlx.Array // preprocessed tensor from EncodeMultimodal
	ID        int        // input ID for logging
	StartPos  int        // position in token sequence where placeholders begin
	NumTokens int        // number of placeholder tokens
}

var (
	mu       sync.Mutex
	registry = make(map[string]func(root *model.Root) (Model, error))
)

// Register registers a model constructor by architecture name.
// Called from init() in model packages. Panics on duplicate registration.
func Register(arch string, fn func(root *model.Root) (Model, error)) {
	mu.Lock()
	defer mu.Unlock()

	if _, exists := registry[arch]; exists {
		panic(fmt.Sprintf("model architecture %q already registered", arch))
	}
	registry[arch] = fn
}

// New reads config.json from the manifest, detects the architecture, looks up
// the registered constructor, and calls it to create the model (with config
// parsed and struct created, but weights not yet loaded).
func New(root *model.Root) (Model, error) {
	configData, err := root.Manifest.ReadConfig("config.json")
	if err != nil {
		return nil, fmt.Errorf("failed to read config.json: %w", err)
	}

	var archConfig struct {
		Architectures []string `json:"architectures"`
	}
	if err := json.Unmarshal(configData, &archConfig); err != nil {
		return nil, fmt.Errorf("failed to parse config.json: %w", err)
	}

	if len(archConfig.Architectures) == 0 {
		return nil, fmt.Errorf("no architectures found in config.json")
	}

	arch := archConfig.Architectures[0]
	slog.Info("Model architecture", "arch", arch)

	mu.Lock()
	fn, ok := registry[arch]
	mu.Unlock()

	if !ok {
		return nil, fmt.Errorf("unsupported architecture: %s", arch)
	}

	return fn(root)
}

// Weights returns a function that loads model weights, then pins all
// arrays reachable from the model struct and sweeps everything else.
func Weights(m Model) func(map[string]*mlx.Array) error {
	return func(tensors map[string]*mlx.Array) error {
		if err := m.LoadWeights(tensors); err != nil {
			return err
		}

		collected := mlx.Collect(m)
		for _, arr := range collected {
			mlx.Pin(arr)
		}
		mlx.Sweep()
		mlx.Eval(collected...)

		return nil
	}
}
