package ortrunner

import (
	"context"
	"log/slog"
	"os"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/x/ortrunner/oga"
)

// Runner manages ORT GenAI model loading and inference.
type Runner struct {
	model     *oga.Model
	tokenizer *oga.Tokenizer
	config    *oga.Config
	modelDir  string

	Requests chan Request
}

// Request represents a completion request.
type Request struct {
	TextCompletionsRequest
	Responses chan CompletionResponse
	Ctx       context.Context
}

// TextCompletionsRequest is the JSON body for /v1/completions.
type TextCompletionsRequest struct {
	Prompt  string `json:"prompt"`
	Options struct {
		Temperature float32 `json:"temperature"`
		TopP        float32 `json:"top_p"`
		TopK        int     `json:"top_k"`
		MaxTokens   int     `json:"max_tokens"`
		NumPredict  int     `json:"num_predict"`
	} `json:"options"`
}

// CompletionResponse is a single JSONL line streamed back to the client.
type CompletionResponse struct {
	Content    string           `json:"content,omitempty"`
	Done       bool             `json:"done"`
	DoneReason int              `json:"done_reason,omitempty"`
	Error      *api.StatusError `json:"error,omitempty"`

	PromptEvalCount    int           `json:"prompt_eval_count,omitempty"`
	PromptEvalDuration time.Duration `json:"prompt_eval_duration,omitempty"`
	EvalCount          int           `json:"eval_count,omitempty"`
	EvalDuration       time.Duration `json:"eval_duration,omitempty"`
}

// Load loads the ORT GenAI model from a directory.
func (r *Runner) Load(modelDir string) error {
	r.modelDir = modelDir

	cfg, err := oga.NewConfig(modelDir)
	if err != nil {
		return err
	}
	r.config = cfg

	// Configure execution provider based on environment
	provider := os.Getenv("OLLAMA_ONNX_PROVIDER")
	if provider == "" {
		provider = "dml" // default to DirectML for GPU
	}

	if os.Getenv("OLLAMA_ONNX_NPU") == "1" {
		provider = "qnn"
	}

	// Device targeting: OLLAMA_ORT_DEVICE_TYPE=npu|gpu, OLLAMA_ORT_DEVICE_ID=<index>
	deviceType := os.Getenv("OLLAMA_ORT_DEVICE_TYPE")
	deviceID := os.Getenv("OLLAMA_ORT_DEVICE_ID")

	slog.Info("configuring ORT GenAI execution provider", "provider", provider, "device_type", deviceType, "device_id", deviceID)

	if err := cfg.ClearProviders(); err != nil {
		slog.Warn("failed to clear providers, using defaults", "error", err)
	} else {
		switch provider {
		case "qnn":
			if err := cfg.AppendProvider("QNN"); err != nil {
				return err
			}
			if err := cfg.SetProviderOption("QNN", "backend_type", "htp"); err != nil {
				slog.Warn("failed to set QNN backend_type", "error", err)
			}
		case "dml":
			if err := cfg.AppendProvider("dml"); err != nil {
				return err
			}
			// Apply device targeting options for DML
			if deviceID != "" {
				// Explicit device index takes priority (DXGI enumeration order)
				if err := cfg.SetProviderOption("dml", "device_id", deviceID); err != nil {
					slog.Warn("failed to set DML device_id", "error", err)
				}
			} else if deviceType != "" {
				// Use DXCore device_filter for type-based targeting
				switch deviceType {
				case "npu", "NPU":
					if err := cfg.SetProviderOption("dml", "device_filter", "npu"); err != nil {
						slog.Warn("failed to set DML device_filter=npu, trying performance_preference", "error", err)
						// Fallback: prefer minimum power (NPU > GPU in sort order)
						if err := cfg.SetProviderOption("dml", "performance_preference", "minimum_power"); err != nil {
							slog.Warn("failed to set DML performance_preference", "error", err)
						}
					}
				case "gpu", "GPU":
					if err := cfg.SetProviderOption("dml", "device_filter", "gpu"); err != nil {
						slog.Warn("failed to set DML device_filter=gpu", "error", err)
					}
				default:
					slog.Warn("unknown OLLAMA_ORT_DEVICE_TYPE, ignoring", "device_type", deviceType)
				}
			}
		case "cpu":
			// No provider needed, CPU is the default fallback
		default:
			if err := cfg.AppendProvider(provider); err != nil {
				return err
			}
		}
	}

	slog.Info("loading ORT GenAI model", "dir", modelDir)
	model, err := oga.NewModel(cfg)
	if err != nil {
		return err
	}
	r.model = model

	tok, err := oga.NewTokenizer(model)
	if err != nil {
		return err
	}
	r.tokenizer = tok

	slog.Info("ORT GenAI model loaded successfully")
	return nil
}

// Close frees all ORT GenAI resources.
func (r *Runner) Close() {
	if r.tokenizer != nil {
		r.tokenizer.Close()
	}
	if r.model != nil {
		r.model.Close()
	}
	if r.config != nil {
		r.config.Close()
	}
}
