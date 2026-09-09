package mlxrunner

import (
	"context"
	"errors"
	"log/slog"
	"maps"
	"net"
	"net/http"
	"slices"
	"strings"

	"golang.org/x/sync/errgroup"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/x/internal/mlxthread"
	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/cache"
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

	Ctx         context.Context //nolint:containedctx // Queued requests carry caller cancellation to the runner.
	Tokens      []int32
	MediaItems  []mediaItem
	Layout      any // opaque PrepareMedia layout state, stamped on every batch
	SamplerOpts sample.Options
	Grammar     *grammarCompilation
}

type Runner struct {
	Model         base.Model
	Tokenizer     *tokenizer.Tokenizer
	Requests      chan Request
	Sampler       *sample.Sampler
	cache         *prefixCache
	contextLength int
	mlxThread     *mlxthread.Thread
	// grammarEngine is the structured-output subsystem; nil when the grammar
	// library or vocabulary failed to load.
	grammarEngine *grammarEngine
	// spec is the speculative-decoding subsystem. Nil when the model ships no
	// draft head.
	spec *speculation
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

	// On Metal, materialize the loaded tensors with CPU reads before any
	// weight graph exists, so the weight eval never commits a command buffer
	// that waits on file data. CUDA loads read at dispatch and need no pre-pass.
	if mlx.MetalIsAvailable() {
		mlx.Eval(slices.Collect(maps.Values(tensors))...)
	}

	// Assign weights to model (model-specific logic). Target and draft weights
	// must be loaded before sweeping so tensors from a combined manifest are
	// not discarded before the draft model can retain them.
	if err := m.LoadWeights(tensors); err != nil {
		return err
	}

	var draftModel base.DraftModel
	draft, err := base.NewDraft(root, m)
	if err != nil {
		return err
	}
	if draft != nil {
		if err := draft.LoadWeights(tensors); err != nil {
			return err
		}
		draftModel = draft
	} else if sd, ok := m.(base.SelfDraft); ok {
		// Inline draft head: already loaded with the target; nil if none shipped.
		draftModel = sd.SelfDraft()
	}

	collected := mlx.Collect(m)
	if draft != nil {
		draftArrays := mlx.Collect(draft)
		collected = append(collected, draftArrays...)
		if root.Draft != nil {
			slog.Info("Loaded draft model", "tensor_prefix", root.Draft.TensorPrefix, "config", root.Draft.Config, "arrays", len(draftArrays))
		} else {
			slog.Info("Loaded draft model", "arrays", len(draftArrays))
		}
	}
	for _, arr := range collected {
		mlx.Pin(arr)
	}
	mlx.Sweep()
	mlx.Eval(collected...)
	configureWiredMemory()

	r.Model = m
	r.Tokenizer = m.Tokenizer()
	r.contextLength = m.MaxContextLength()
	caches := m.NewCaches()
	draftCaches := newDraftCaches(draftModel)
	r.cache = newPrefixCache(slices.Concat(caches, draftCaches))
	r.Sampler = sample.New(r.contextLength)
	r.spec = newSpeculation(r, draftModel, caches, draftCaches)
	r.grammarEngine = newGrammarEngine(logitsWidth(m), r.Tokenizer)

	mlx.EnableCompile()

	return nil
}

func (r *Runner) Close() {
	if r.grammarEngine != nil {
		r.grammarEngine.close()
		r.grammarEngine = nil
	}
}

// newDraftCaches returns nil when the model ships no draft.
func newDraftCaches(draft base.DraftModel) []cache.Cache {
	if draft == nil {
		return nil
	}
	return draft.NewCaches()
}

// logitsWidth reads a model's logits width off a one-token forward's static
// shape — the same Forward and Unembed path decode logits take. Nothing is
// evaluated, and the probe's caches and graph are released before returning,
// which sweeps every unpinned array: call this only at load, after the
// model's weights are pinned.
func logitsWidth(m base.Model) int {
	caches := m.NewCaches()
	hidden, _ := m.Forward(&batch.Batch{
		InputIDs:     mlx.FromValues([]int32{0}, 1, 1),
		SeqOffsets:   []int32{0},
		SeqQueryLens: []int32{1},
	}, caches)
	logits := m.Unembed(hidden)
	width := logits.Dim(logits.NumDims() - 1)
	for _, c := range caches {
		if c != nil {
			c.Free()
		}
	}
	mlx.Sweep()
	return width
}

func configureWiredMemory() {
	if !mlx.GPUIsAvailable() {
		return
	}

	active := mlx.ActiveMemory()
	maxRecommended, err := mlx.MaxRecommendedWorkingSetSize()
	if err != nil {
		slog.Warn("Unable to query MLX recommended working set; using pageable memory", "error", err)
		return
	}

	limit := min(active, maxRecommended)
	previous, err := mlx.SetWiredLimit(limit)
	if err != nil {
		slog.Warn("Unable to configure MLX wired memory; using pageable memory",
			"active", mlx.PrettyBytes(active),
			"limit", mlx.PrettyBytes(limit),
			"error", err)
		return
	}

	if active > maxRecommended {
		slog.Warn("MLX model exceeds the recommended working set; performance may be degraded",
			"active", mlx.PrettyBytes(active),
			"recommended", mlx.PrettyBytes(maxRecommended))
	}
	// Limiting residency to the loaded model's active allocations avoids
	// reserving the remaining capacity for growing KV caches.
	slog.Debug("Configured MLX wired memory",
		"active", mlx.PrettyBytes(active),
		"limit", mlx.PrettyBytes(limit),
		"previous", mlx.PrettyBytes(previous))
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
	defer request.Grammar.close()
	if r.mlxThread == nil {
		return request.Pipeline(request.Ctx, request)
	}

	return r.mlxThread.Do(request.Ctx, func() error {
		return request.Pipeline(request.Ctx, request)
	})
}
