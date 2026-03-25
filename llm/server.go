package llm

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"os"
	"slices"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/ml"
)

var ErrLoadRequiredFull = errors.New("unable to load full model on GPU")

type filteredEnv []string

func (e filteredEnv) LogValue() slog.Value {
	var attrs []slog.Attr
	for _, env := range e {
		if key, value, ok := strings.Cut(env, "="); ok {
			switch {
			case strings.HasPrefix(key, "OLLAMA_"),
				strings.HasPrefix(key, "CUDA_"),
				strings.HasPrefix(key, "ROCR_"),
				strings.HasPrefix(key, "ROCM_"),
				strings.HasPrefix(key, "HIP_"),
				strings.HasPrefix(key, "GPU_"),
				strings.HasPrefix(key, "HSA_"),
				strings.HasPrefix(key, "GGML_"),
				slices.Contains([]string{
					"PATH",
					"LD_LIBRARY_PATH",
					"DYLD_LIBRARY_PATH",
				}, key):
				attrs = append(attrs, slog.String(key, value))
			}
		}
	}
	return slog.GroupValue(attrs...)
}

type LlamaServer interface {
	ModelPath() string
	Load(ctx context.Context, systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, requireFull bool) ([]ml.DeviceID, error)
	Ping(ctx context.Context) error
	WaitUntilRunning(ctx context.Context) error
	Completion(ctx context.Context, req CompletionRequest, fn func(CompletionResponse)) error
	Embedding(ctx context.Context, input string) ([]float32, int, error)
	Tokenize(ctx context.Context, content string) ([]int, error)
	Detokenize(ctx context.Context, tokens []int) (string, error)
	Close() error
	MemorySize() (total, vram uint64)
	VRAMByGPU(id ml.DeviceID) uint64
	Pid() int
	GetPort() int
	GetDeviceInfos(ctx context.Context) []ml.DeviceInfo
	HasExited() bool
	ContextLength() int
}

// LoadModel will load a model from disk. The model must be in the GGML format.
//
// It collects array values for arrays with a size less than or equal to
// maxArraySize. If maxArraySize is 0, the default value of 1024 is used. If
// the maxArraySize is negative, all arrays are collected.
func LoadModel(model string, maxArraySize int) (*ggml.GGML, error) {
	if _, err := os.Stat(model); err != nil {
		return nil, err
	}

	f, err := os.Open(model)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	return ggml.Decode(f, maxArraySize)
}

// NewLlamaServer creates a new llama-server runner for the given model.
// All GGML models are served via the upstream llama-server subprocess.
func NewLlamaServer(systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, modelPath string, f *ggml.GGML, adapters, projectors []string, opts api.Options, numParallel int) (LlamaServer, error) {
	slog.Info("using llama-server for model", "model", modelPath)

	// Verify the requested context size is <= the model training size
	trainCtx := f.KV().ContextLength()
	if opts.NumCtx > int(trainCtx) && trainCtx > 0 {
		slog.Warn("requested context size too large for model", "num_ctx", opts.NumCtx, "n_ctx_train", trainCtx)
		opts.NumCtx = int(trainCtx)
	}

	kvct := strings.ToLower(envconfig.KvCacheType())
	return NewLlamaServerRunner(gpus, modelPath, f, adapters, projectors, opts, numParallel, kvct)
}

// Server status types

type ServerStatus int

const (
	ServerStatusReady ServerStatus = iota
	ServerStatusNoSlotsAvailable
	ServerStatusLaunched
	ServerStatusLoadingModel
	ServerStatusNotResponding
	ServerStatusError
)

func (s ServerStatus) String() string {
	switch s {
	case ServerStatusReady:
		return "llm server ready"
	case ServerStatusNoSlotsAvailable:
		return "llm busy - no slots available"
	case ServerStatusLaunched:
		return "llm server launched"
	case ServerStatusLoadingModel:
		return "llm server loading model"
	case ServerStatusNotResponding:
		return "llm server not responding"
	default:
		return "llm server error"
	}
}

type ServerStatusResponse struct {
	Status   ServerStatus `json:"status"`
	Progress float32      `json:"progress"`
}

// Request/Response types

const maxBufferSize = 512 * format.KiloByte

type ImageData struct {
	Data []byte `json:"data"`
	ID   int    `json:"id"`
}

type CompletionRequest struct {
	Prompt  string
	Format  json.RawMessage
	Images  []ImageData
	Options *api.Options

	Grammar         string // set before sending the request to the subprocess
	Shift           bool
	Truncate        bool
	PreservedTokens []string // special tokens to render as text (not strip) during detokenization

	// Logprobs specifies whether to include log probabilities in the response
	Logprobs bool

	// TopLogprobs specifies the number of most likely alternative tokens to return (0-20)
	TopLogprobs int

	// Image generation fields
	Width  int32 `json:"width,omitempty"`
	Height int32 `json:"height,omitempty"`
	Steps  int32 `json:"steps,omitempty"`
	Seed   int64 `json:"seed,omitempty"`
}

// DoneReason represents the reason why a completion response is done
type DoneReason int

const (
	DoneReasonStop DoneReason = iota
	DoneReasonLength
	DoneReasonConnectionClosed
)

func (d DoneReason) String() string {
	switch d {
	case DoneReasonLength:
		return "length"
	case DoneReasonStop:
		return "stop"
	default:
		return ""
	}
}

// TokenLogprob represents log probability information for a single token alternative.
type TokenLogprob struct {
	Token   string  `json:"token"`
	Logprob float64 `json:"logprob"`
}

// Logprob contains log probability information for a generated token.
type Logprob struct {
	TokenLogprob
	TopLogprobs []TokenLogprob `json:"top_logprobs,omitempty"`
}

type CompletionResponse struct {
	Content            string        `json:"content"`
	DoneReason         DoneReason    `json:"done_reason"`
	Done               bool          `json:"done"`
	PromptEvalCount    int           `json:"prompt_eval_count"`
	PromptEvalDuration time.Duration `json:"prompt_eval_duration"`
	EvalCount          int           `json:"eval_count"`
	EvalDuration       time.Duration `json:"eval_duration"`

	// Logprobs contains log probability information if requested
	Logprobs []Logprob `json:"logprobs,omitempty"`

	// Image contains base64-encoded image data for image generation
	Image string `json:"image,omitempty"`

	// Step is the current step in image generation
	Step int `json:"step,omitempty"`

	// TotalSteps is the total number of steps for image generation
	TotalSteps int `json:"total_steps,omitempty"`
}
