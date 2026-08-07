package mlxrunner

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"sync"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
)

// mediaHeadroom reports how many bytes are left of the device's recommended
// working set once the model is resident, for models to size media work
// against. It is sampled once rather than per request because both inputs are
// fixed for the life of the runner: the recommended working set is a device
// property, and residency does not change after the model loads. Sampling it
// per request would also make the answer drift with unrelated allocator
// activity, which matters because a model may fold the budget into a cache key.
//
// This lives here rather than beside the runner's other memory setup because
// the multimodal path is temporary; it should move once media inputs join the
// normal request lifecycle.
//
// Returns 0 when the device cannot report a working set, which models read as
// a request for their cheapest setting rather than as an error.
var mediaHeadroom = sync.OnceValue(func() int {
	if !mlx.GPUIsAvailable() {
		slog.Warn("MLX GPU unavailable; media inputs will use their most conservative memory budget")
		return 0
	}

	recommended, err := mlx.MaxRecommendedWorkingSetSize()
	if err != nil {
		slog.Warn("Unable to query the MLX recommended working set; media inputs will use their most conservative memory budget", "error", err)
		return 0
	}

	active := mlx.ActiveMemory()
	headroom := max(0, recommended-active)
	slog.Info("media memory budget",
		"headroom", mlx.PrettyBytes(headroom),
		"active", mlx.PrettyBytes(active),
		"recommended", mlx.PrettyBytes(recommended))
	return headroom
})

// multimodalRequestModel uses the prepared media inputs while processing the
// prompt, then delegates ordinary generation to the loaded model.
type multimodalRequestModel struct {
	base.Model
	prompt     base.PreparedMultimodalPrompt
	prefillEnd int
}

func (m *multimodalRequestModel) Forward(b *batch.Batch, caches []cache.Cache) *mlx.Array {
	if len(b.SeqOffsets) == 1 && int(b.SeqOffsets[0]) < m.prefillEnd {
		return m.prompt.Forward(b, caches)
	}
	return m.Model.Forward(b, caches)
}

// MultimodalPipeline prepares request-local media state and delegates the
// generation lifecycle to TextGenerationPipeline.
func (r *Runner) MultimodalPipeline(ctx context.Context, request Request) error {
	prompt, err := r.prepareMultimodal(&request)
	if err != nil {
		return err
	}
	if prompt == nil {
		return errors.New("multimodal request contains no media")
	}
	defer prompt.Close()

	requestRunner := *r
	requestRunner.Model = &multimodalRequestModel{
		Model:      r.Model,
		prompt:     prompt,
		prefillEnd: len(request.Tokens) - 1,
	}
	requestRunner.cache = newPrefixCache(r.Model)
	requestRunner.spec = nil
	requestRunner.disablePrefillSnapshots = true
	defer requestRunner.cache.freeAll()

	return requestRunner.TextGenerationPipeline(ctx, request)
}

func (r *Runner) prepareMultimodal(request *Request) (base.PreparedMultimodalPrompt, error) {
	if len(request.Media) == 0 {
		return nil, nil
	}

	model, ok := r.Model.(base.MultimodalModel)
	if !ok {
		return nil, errors.New("model does not support multimodal input")
	}

	media := make([]base.MediaInput, len(request.Media))
	for i := range request.Media {
		media[i] = base.MediaInput{
			ID:   request.Media[i].ID,
			Kind: string(request.Media[i].Kind),
			Data: request.Media[i].Data,
		}
	}

	prompt, err := model.PrepareMultimodalPrompt(request.Prompt, media, mediaHeadroom())
	if err != nil {
		return nil, api.StatusError{StatusCode: http.StatusBadRequest, ErrorMessage: err.Error()}
	}
	if prompt == nil {
		return nil, errors.New("model returned an empty multimodal prompt")
	}

	inputs := prompt.Tokens()
	if len(inputs) == 0 {
		prompt.Close()
		return nil, api.StatusError{StatusCode: http.StatusBadRequest, ErrorMessage: "empty prompt"}
	}
	if len(inputs) >= r.contextLength {
		prompt.Close()
		return nil, api.StatusError{
			StatusCode: http.StatusBadRequest,
			ErrorMessage: fmt.Sprintf(
				"input length (%d tokens) exceeds the model's maximum context length (%d tokens)",
				len(inputs),
				r.contextLength,
			),
		}
	}

	maxGenerate := r.contextLength - len(inputs)
	if request.Options.NumPredict <= 0 {
		request.Options.NumPredict = maxGenerate
	} else {
		request.Options.NumPredict = min(request.Options.NumPredict, maxGenerate)
	}
	request.Tokens = inputs

	return prompt, nil
}
