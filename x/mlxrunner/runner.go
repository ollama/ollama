//go:build mlx

package mlxrunner

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"time"

	"golang.org/x/sync/errgroup"

	"github.com/ollama/ollama/x/imagegen/manifest"
	"github.com/ollama/ollama/x/imagegen/tokenizer"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/sample"
	"github.com/ollama/ollama/x/models/glm4_moe_lite"
)

// TextModel is the interface that model implementations must satisfy.
type TextModel interface {
	Forward(inputs *mlx.Array, cache []cache.Cache) *mlx.Array
	Unembed(x *mlx.Array) *mlx.Array
	NumLayers() int
}

type Request struct {
	TextCompletionsRequest
	Responses chan Response
	Pipeline  func(Request) error

	sample.Sampler
	caches []cache.Cache
}

type TextCompletionsRequest struct {
	Prompt  string `json:"prompt"`
	Options struct {
		Temperature float32 `json:"temperature"`
		TopP        float32 `json:"top_p"`
		MinP        float32 `json:"min_p"`
		TopK        int     `json:"top_k"`
		MaxTokens   int     `json:"max_tokens"`

		// Deprecated: use MaxTokens instead
		NumPredict int `json:"num_predict"`
	} `json:"options"`
}

type Response struct {
	Text       string    `json:"content,omitempty"`
	Token      int       `json:"token,omitempty"`
	Logprobs   []float32 `json:"logprobs,omitempty"`
	Done       bool      `json:"done,omitempty"`
	DoneReason int       `json:"done_reason,omitempty"`

	PromptTokens             int           `json:"prompt_eval_count,omitempty"`
	PromptTokensDuration     time.Duration `json:"prompt_eval_duration,omitempty"`
	CompletionTokens         int           `json:"eval_count,omitempty"`
	CompletionTokensDuration time.Duration `json:"eval_duration,omitempty"`
	TotalTokens              int           `json:"total_tokens,omitempty"`
}

type Runner struct {
	Model        TextModel
	Tokenizer    *tokenizer.Tokenizer
	Requests     chan Request
	CacheEntries map[int32]*CacheEntry
}

func (r *Runner) Load(modelName string) error {
	modelManifest, err := manifest.LoadManifest(modelName)
	if err != nil {
		return err
	}

	// Read config to detect architecture
	configData, err := modelManifest.ReadConfig("config.json")
	if err != nil {
		return fmt.Errorf("failed to read config.json: %w", err)
	}

	var archConfig struct {
		Architectures []string `json:"architectures"`
	}
	if err := json.Unmarshal(configData, &archConfig); err != nil {
		return fmt.Errorf("failed to parse config.json: %w", err)
	}

	if len(archConfig.Architectures) == 0 {
		return fmt.Errorf("no architectures found in config.json")
	}

	slog.Info("Model architecture", "arch", archConfig.Architectures[0])

	switch archConfig.Architectures[0] {
	case "Glm4MoeLiteForCausalLM", "GLM4MoeLite":
		model, err := glm4_moe_lite.LoadFromManifest(modelManifest)
		if err != nil {
			return fmt.Errorf("failed to load GLM4-MoE-Lite model: %w", err)
		}
		r.Model = model
		r.Tokenizer = model.Tokenizer()
	default:
		return fmt.Errorf("unsupported architecture: %s", archConfig.Architectures[0])
	}

	return nil
}

func (r *Runner) Run(host, port string, mux http.Handler) error {
	g, ctx := errgroup.WithContext(context.Background())

	g.Go(func() error {
		for {
			select {
			case <-ctx.Done():
				return nil
			case request := <-r.Requests:
				if err := request.Pipeline(request); err != nil {
					break
				}

				close(request.Responses)
			}
		}
	})

	g.Go(func() error {
		slog.Info("Starting HTTP server", "host", host, "port", port)
		return http.ListenAndServe(net.JoinHostPort(host, port), mux)
	})

	return g.Wait()
}
