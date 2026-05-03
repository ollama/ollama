package mlxrunner

import (
	"context"
	"errors"
	"log/slog"
	"net"
	"net/http"
	"strings"

	"golang.org/x/sync/errgroup"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/x/internal/mlxthread"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
	"github.com/ollama/ollama/x/mlxrunner/sample"
	"github.com/ollama/ollama/x/tokenizer"
)

// Request is a short-lived struct that carries a completion request through
// a channel from the HTTP handler to the runner goroutine. The ctx field
// must travel with the request so that cancellation propagates across the
// channel boundary.
type Request struct {
	CompletionRequest
	Responses chan CompletionResponse
	Pipeline  func(context.Context, Request) error

	Ctx         context.Context //nolint:containedctx
	Tokens      []int32
	SamplerOpts sample.Options
}

type Runner struct {
	Model         base.Model
	Tokenizer     *tokenizer.Tokenizer
	Requests      chan Request
	Sampler       *sample.Sampler
	cache         kvCache
	contextLength int
	mlxThread     *mlxthread.Thread
}

func (r *Runner) Load(modelName string) error {
	root, err := model.Open(modelName)
	if err != nil {
		return err
	}
	defer root.Close()

	m, err := base.New(root)
	if err != nil {
		return err
	}

	// Load all tensor blobs from manifest
	tensors, err := loadTensorsFromManifest(root)
	if err != nil {
		return err
	}

	// Assign weights to model (model-specific logic)
	loadWeights := base.Weights(m)
	if err := loadWeights(tensors); err != nil {
		return err
	}

	r.Model = m
	r.Tokenizer = m.Tokenizer()
	r.contextLength = m.MaxContextLength()
	r.Sampler = sample.New(r.contextLength)

	mlx.EnableCompile()
	return nil
}

// loadTensorsFromManifest loads all tensor blobs from the manifest into a
// flat map, deduplicating by digest and remapping safetensors key suffixes.
//
// Two aux-naming conventions may appear in the source blobs (either because
// of how Ollama packages tensors on import, or because "ollama create
// --experimental" occasionally emits an orphan blob that retains the
// original mlx-lm naming):
//
//   - Dot-child singular: "<foo>.weight.scale" / "<foo>.weight.bias".
//     Ollama-native form.
//   - Sibling plural: "<foo>.scales" / "<foo>.biases".
//     mlx-lm / mx.nn.quantize-native form.
//
// Both are normalised to the canonical "<foo>.weight_scale" / "<foo>.weight_qbias"
// form so downstream consumers (MakeLinearLayer, loadStackedProjection, etc.)
// can use a single lookup key without caring which convention was in the blob.
//
// Uses a three-phase approach so that .bias / .biases → _qbias remapping can
// consult complete scale knowledge regardless of Go map iteration order.
func loadTensorsFromManifest(root *model.Root) (map[string]*mlx.Array, error) {
	// Phase 1: Load all tensors raw from all blobs
	rawTensors := make(map[string]*mlx.Array)
	seen := make(map[string]bool)
	for _, layer := range root.Manifest.GetTensorLayers("") {
		if seen[layer.Digest] {
			continue
		}
		seen[layer.Digest] = true
		blobPath := root.Manifest.BlobPath(layer.Digest)
		for name, arr := range mlx.Load(blobPath) {
			rawTensors[name] = arr
		}
	}

	allTensors := normaliseAuxNames(rawTensors)
	slog.Info("Loaded tensors from manifest", "count", len(allTensors))
	return allTensors, nil
}

// normaliseAuxNames rewrites quantisation aux tensor keys to the canonical
// "<base>.weight_scale" / "<base>.weight_qbias" form, recognising both the
// Ollama-native dot-child singular naming (".scale" / ".bias") and the
// mlx-lm sibling plural naming (".scales" / ".biases").
//
// Singular wins over plural when both forms target the same canonical key —
// Ollama-native singular is the canonical convention; the plural fallback
// only fills targets the singular pass did not populate. Two-pass also
// ensures .bias / .biases → _qbias remapping has complete scale knowledge.
func normaliseAuxNames[V any](raw map[string]V) map[string]V {
	scaleBaseNames := make(map[string]bool)
	out := make(map[string]V, len(raw))

	// Pass 1a: singular .scale (Ollama-native, canonical).
	for name, arr := range raw {
		if strings.HasSuffix(name, ".scale") {
			baseName := strings.TrimSuffix(name, ".scale")
			out[baseName+"_scale"] = arr
			scaleBaseNames[baseName] = true
		}
	}
	// Pass 1b: plural .scales (mlx-lm sibling) — only fill targets the
	// singular pass did not populate, so the precedence is deterministic.
	for name, arr := range raw {
		if strings.HasSuffix(name, ".scales") {
			stem := strings.TrimSuffix(name, ".scales")
			target := stem + ".weight_scale"
			if _, exists := out[target]; !exists {
				out[target] = arr
				scaleBaseNames[stem+".weight"] = true
			}
		}
	}

	// Pass 2: bias-like auxiliaries and pass-through
	for name, arr := range raw {
		if strings.HasSuffix(name, ".scale") || strings.HasSuffix(name, ".scales") {
			continue // already handled in Pass 1
		}
		switch {
		case strings.HasSuffix(name, ".bias") && !strings.HasSuffix(name, ".weight_qbias"):
			baseName := strings.TrimSuffix(name, ".bias")
			if scaleBaseNames[baseName] {
				out[baseName+"_qbias"] = arr
			} else {
				out[name] = arr
			}
		case strings.HasSuffix(name, ".biases"):
			// mlx-lm sibling plural bias → "<foo>.weight_qbias" if a matching
			// scale was remapped, else keep the original name (some layers
			// use .biases for dense bias).
			stem := strings.TrimSuffix(name, ".biases")
			if scaleBaseNames[stem+".weight"] {
				out[stem+".weight_qbias"] = arr
			} else {
				out[name] = arr
			}
		default:
			out[name] = arr
		}
	}
	return out
}

func (r *Runner) Run(host, port string, mux http.Handler) error {
	g, ctx := errgroup.WithContext(context.Background())

	g.Go(func() error {
		for {
			select {
			case <-ctx.Done():
				return nil
			case request := <-r.Requests:
				err := r.runRequest(request)
				if err != nil {
					slog.Info("Request terminated", "error", err)
					var statusErr api.StatusError
					if !errors.As(err, &statusErr) {
						statusErr = api.StatusError{
							StatusCode:   http.StatusInternalServerError,
							ErrorMessage: err.Error(),
						}
					}
					select {
					case request.Responses <- CompletionResponse{Error: &statusErr}:
					case <-request.Ctx.Done():
					}
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

func (r *Runner) runRequest(request Request) error {
	if r.mlxThread == nil {
		return request.Pipeline(request.Ctx, request)
	}

	return r.mlxThread.Do(request.Ctx, func() error {
		return request.Pipeline(request.Ctx, request)
	})
}
