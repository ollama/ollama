package mlxrunner

import (
	"context"
	"log/slog"
	"net"
	"net/http"
	"time"

	"golang.org/x/sync/errgroup"

	"github.com/ollama/ollama/tokenizer"
	"github.com/ollama/ollama/types/model"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	_ "github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
	"github.com/ollama/ollama/x/mlxrunner/sample"
)

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
	Model        base.Model
	Tokenizer    tokenizer.Tokenizer
	Requests     chan Request
	CacheEntries map[int32]*CacheEntry
}

func (r *Runner) Load(name model.Name) (err error) {
	root, err := model.Open(name)
	if err != nil {
		return err
	}
	defer root.Close()

	r.Model, err = base.New(root)
	if err != nil {
		return err
	}

	r.Tokenizer, err = tokenizer.New(root)
	if err != nil {
		return err
	}

	weights, quantizations, afterLoadFuncs := base.Walk(r.Model)
	return mlx.LoadAll(root, weights, quantizations, afterLoadFuncs)
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
