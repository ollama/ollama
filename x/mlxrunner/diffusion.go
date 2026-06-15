package mlxrunner

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/ollama/ollama/x/mlxrunner/model/base"
)

// fitDiffusionCanvases bounds the number of canvases so the prompt plus every
// processed canvas fits the model's context window. Block diffusion processes
// each canvas at positions [nPast, nPast+canvas) and commits whole canvases, so
// the context budget is promptLen + n*canvas - not just the emitted tokens that
// Prepare accounts for. Returns the reduced canvas count, or an error when not
// even a single canvas fits.
func fitDiffusionCanvases(maxCanvases, canvas, promptLen, contextLength int) (int, error) {
	if canvas <= 0 {
		return maxCanvases, nil
	}
	fit := (contextLength - promptLen) / canvas
	if fit < 1 {
		return 0, fmt.Errorf("prompt (%d tokens) leaves no room for a diffusion canvas (%d tokens) within the context window (%d tokens)", promptLen, canvas, contextLength)
	}
	return min(maxCanvases, fit), nil
}

// DiffusionGenerationPipeline drives a block-diffusion model. The denoising loop
// itself lives on the model (Diffuse); this pipeline resolves the runtime config,
// streams the emitted tokens (buffering partial UTF-8), and sends the final
// summary response. It is selected over TextGenerationPipeline when the loaded
// model implements base.DiffusionModel.
func (r *Runner) DiffusionGenerationPipeline(ctx context.Context, request Request) error {
	dm, ok := r.Model.(base.DiffusionModel)
	if !ok {
		return errors.New("model does not support diffusion generation")
	}

	// Seed 0 lets the model apply the reference default (1234).
	var seed int64
	if request.SamplerOpts.UseSeed {
		seed = int64(request.SamplerOpts.Seed)
	}
	// The denoising step count uses the model's configured default. A per-request
	// override is intentionally not exposed as a global API option yet (pending
	// maintainer agreement on how diffusion-specific knobs should be surfaced).
	cfg := dm.ResolveDiffuse(request.Options.NumPredict, 0, seed)

	// Prepare caps NumPredict as if tokens were emitted autoregressively, but the
	// denoiser commits whole canvases to the KV cache. Reduce the canvas count so
	// the prompt plus every canvas stays within the context window.
	fitted, err := fitDiffusionCanvases(cfg.MaxCanvases, cfg.Canvas, len(request.Tokens), r.contextLength)
	if err != nil {
		return err
	}
	cfg.MaxCanvases = fitted

	now := time.Now()
	var buf bytes.Buffer
	var emitted int
	emit := func(tok int32) error {
		emitted++
		buf.WriteString(r.Tokenizer.Decode([]int32{tok}))
		content := flushValidUTF8Prefix(&buf)
		if content == "" {
			return nil
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case request.Responses <- CompletionResponse{Content: content}:
			return nil
		}
	}

	if err := dm.Diffuse(ctx, request.Tokens, cfg, emit); err != nil {
		return err
	}

	// Flush any bytes still buffered (e.g. a trailing incomplete rune).
	if tail := buf.String(); tail != "" {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case request.Responses <- CompletionResponse{Content: tail}:
		}
	}

	final := CompletionResponse{
		Done:            true,
		DoneReason:      1,
		PromptEvalCount: len(request.Tokens),
		EvalCount:       emitted,
		EvalDuration:    time.Since(now),
	}
	select {
	case <-ctx.Done():
		return ctx.Err()
	case request.Responses <- final:
		return nil
	}
}
