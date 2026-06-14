package mlxrunner

import (
	"bytes"
	"context"
	"errors"
	"time"

	"github.com/ollama/ollama/x/mlxrunner/model/base"
)

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
	cfg := dm.ResolveDiffuse(request.Options.NumPredict, request.Options.DiffusionSteps, seed)

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
