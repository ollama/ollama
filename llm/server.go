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
			if filteredEnvLogKey(key) {
				attrs = append(attrs, slog.String(key, filteredEnvLogValue(key, value)))
			}
		}
	}
	return slog.GroupValue(attrs...)
}

func filteredEnvLogKey(key string) bool {
	return strings.HasPrefix(key, "CUDA_") ||
		strings.HasPrefix(key, "ROCR_") ||
		strings.HasPrefix(key, "ROCM_") ||
		strings.HasPrefix(key, "HIP_") ||
		strings.HasPrefix(key, "HSA_") ||
		strings.HasPrefix(key, "GGML_") ||
		slices.Contains([]string{
			"PATH",
			"LD_LIBRARY_PATH",
			"DYLD_LIBRARY_PATH",
		}, key)
}

func filteredEnvLogValue(key, value string) string {
	for _, token := range []string{"API", "KEY", "TOKEN", "SECRET", "PASSWORD", "PASS", "CREDENTIAL", "AUTH"} {
		if strings.Contains(strings.ToUpper(key), token) {
			return "[redacted]"
		}
	}
	return value
}

type LlamaServer interface {
	ModelPath() string
	Load(ctx context.Context, systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, requireFull bool) ([]ml.DeviceID, error)
	Ping(ctx context.Context) error
	WaitUntilRunning(ctx context.Context) error
	Completion(ctx context.Context, req CompletionRequest, fn func(CompletionResponse)) error
	Chat(ctx context.Context, req ChatRequest, fn func(ChatResponse)) error
	ApplyChatTemplate(ctx context.Context, req ChatRequest) (string, error)
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

type LlamaServerConfig struct {
	DisableJinja   bool
	ContextShift   bool
	EnableMTP      bool
	DraftModelPath string
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
func NewLlamaServer(systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, modelPath string, f *ggml.GGML, adapters, projectors []string, opts api.Options, numParallel int, config LlamaServerConfig) (LlamaServer, error) {
	slog.Info("using llama-server for model", "model", modelPath)

	// Verify the requested context size is <= the model training size
	trainCtx := f.KV().ContextLength()
	if opts.NumCtx > int(trainCtx) && trainCtx > 0 {
		slog.Warn("requested context size too large for model", "num_ctx", opts.NumCtx, "n_ctx_train", trainCtx)
		opts.NumCtx = int(trainCtx)
	}

	kvct := strings.ToLower(envconfig.KvCacheType())
	return NewLlamaServerRunner(gpus, modelPath, f, adapters, projectors, opts, numParallel, kvct, config)
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

const (
	llamaServerStreamInitialBufferSize = 64 * 1024
	// llamaServerStreamMaxBufferSize bounds a single runner response stream line.
	llamaServerStreamMaxBufferSize = 8 * format.MegaByte
)

type MediaKind string

const (
	MediaKindUnknown MediaKind = ""
	MediaKindImage   MediaKind = "image"
	MediaKindAudio   MediaKind = "audio"
)

type MediaData struct {
	Data []byte `json:"data"`
	ID   int    `json:"id"`
	Kind MediaKind
}

type Message struct {
	Role       string
	Content    string
	Thinking   string
	Media      []MediaData
	ToolCalls  []api.ToolCall
	ToolName   string
	ToolCallID string
}

func MessageFromAPI(msg api.Message) Message {
	media := make([]MediaData, len(msg.Images))
	for i, data := range msg.Images {
		media[i] = NewMediaData(i, data)
	}

	return Message{
		Role:       msg.Role,
		Content:    msg.Content,
		Thinking:   msg.Thinking,
		Media:      media,
		ToolCalls:  msg.ToolCalls,
		ToolName:   msg.ToolName,
		ToolCallID: msg.ToolCallID,
	}
}

type CompletionRequest struct {
	Prompt  string
	Format  json.RawMessage
	Media   []MediaData
	Options *api.Options

	Grammar         string // set before sending the request to the subprocess
	Shift           bool
	Truncate        bool
	PreservedTokens []string // parser tokens to render as text; ignored by non-llama-server runners
	ToolCallTag     string   // raw generic tool parser tag, if any
	LeadingBOS      string   // textual BOS emitted by Go rendering, if any

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

type ChatRequest struct {
	Messages []api.Message
	Tools    api.Tools
	Format   json.RawMessage
	Options  *api.Options
	Think    *api.ThinkValue
	Shift    bool

	Logprobs    bool
	TopLogprobs int
}

type ChatResponse struct {
	Message            api.Message   `json:"message"`
	DoneReason         DoneReason    `json:"done_reason"`
	Done               bool          `json:"done"`
	PromptEvalCount    int           `json:"prompt_eval_count"`
	PromptEvalDuration time.Duration `json:"prompt_eval_duration"`
	EvalCount          int           `json:"eval_count"`
	EvalDuration       time.Duration `json:"eval_duration"`
	Logprobs           []Logprob     `json:"logprobs,omitempty"`
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
