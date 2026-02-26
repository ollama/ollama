//go:build mlx

package mlxrunner

import (
	"bytes"
	"context"
	"errors"
	"log/slog"
	"time"

	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

func (r *Runner) TextGenerationPipeline(request Request) error {
	if r.Model == nil {
		return errors.New("model not loaded")
	}

	var (
		sample, logprobs         *mlx.Array
		nextSample, nextLogprobs *mlx.Array
	)

	defer func() {
		mlx.Unpin(sample, logprobs)
		mlx.Unpin(nextSample, nextLogprobs)
		mlx.Sweep()
		mlx.ClearCache()

		if slog.Default().Enabled(context.TODO(), logutil.LevelTrace) {
			mlx.LogArrays()
			r.cache.log()
		}
	}()

	enableCompile := true
	if modelCompile, ok := r.Model.(interface{ EnableCompile() bool }); ok {
		enableCompile = modelCompile.EnableCompile()
	}
	if enableCompile {
		mlx.EnableCompile()
	} else {
		mlx.DisableCompile()
	}
	mlx.ResetPeakMemory()

	inputs := r.Tokenizer.Encode(request.Prompt, true)
	if len(inputs) == 0 {
		return errors.New("empty prompt")
	}

	session := r.cache.begin(r.Model, inputs)
	defer session.close()

	caches := session.caches
	tokens := session.remaining

	total, processed := len(tokens), 0
	for total-processed > 1 {
		if err := request.Ctx.Err(); err != nil {
			return err
		}

		n := min(2<<10, total-processed-1)
		r.Model.Forward(mlx.FromValues(tokens[processed:processed+n], n).ExpandDims(0), caches)
		mlx.Sweep()
		mlx.Eval(func() []*mlx.Array {
			s := make([]*mlx.Array, 2*len(caches))
			for i, c := range caches {
				s[2*i], s[2*i+1] = c.State()
			}
			return s
		}()...)
		processed += n
		slog.Info("Prompt processing progress", "processed", processed, "total", total)
		mlx.ClearCache()
	}

	step := func(token *mlx.Array) (*mlx.Array, *mlx.Array) {
		fwd := r.Model.Forward(token.ExpandDims(0), caches)
		logits := r.Model.Unembed(fwd)
		logits = logits.Slice(mlx.Slice(), mlx.Slice(logits.Dim(1)-1), mlx.Slice()).Squeeze(1)

		logprobs := logits.Subtract(logits.Logsumexp(true))
		sample := request.Sample(logprobs)

		mlx.Pin(sample, logprobs)
		mlx.Sweep()
		mlx.AsyncEval(sample, logprobs)

		return sample, logprobs
	}

	sample, logprobs = step(mlx.FromValues(tokens[processed:], total-processed))

	var b bytes.Buffer

	now := time.Now()
	final := CompletionResponse{Done: true, PromptEvalCount: total, EvalCount: request.Options.MaxTokens, DoneReason: 1}
	for i := range request.Options.MaxTokens {
		if err := request.Ctx.Err(); err != nil {
			return err
		}

		nextSample, nextLogprobs = step(sample)

		if i == 0 {
			mlx.Eval(sample)
			final.PromptEvalDuration = time.Since(now)
			now = time.Now()
		}

		output := int32(sample.Int())
		session.outputs = append(session.outputs, output)

		if r.Tokenizer.IsEOS(output) {
			final.DoneReason = 0
			final.EvalCount = i
			break
		}

		select {
		case <-request.Ctx.Done():
			return request.Ctx.Err()
		case request.Responses <- CompletionResponse{
			Content: r.Decode(output, &b),
		}:
		}

		mlx.Unpin(sample, logprobs)
		sample, logprobs = nextSample, nextLogprobs
		nextSample, nextLogprobs = nil, nil

		if i%256 == 0 {
			mlx.ClearCache()
		}
	}

	final.EvalDuration = time.Since(now)
	final.PeakMemory = uint64(mlx.PeakMemory())
	select {
	case <-request.Ctx.Done():
		return request.Ctx.Err()
	case request.Responses <- final:
		return nil
	}
}

func (r Runner) Decode(sample int32, b *bytes.Buffer) string {
	token := r.Tokenizer.Decode([]int32{sample})

	if _, err := b.WriteString(token); err != nil {
		slog.Error("Failed to write token to buffer", "error", err)
		return ""
	}

	return flushValidUTF8Prefix(b)
}
