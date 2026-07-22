package mlxrunner

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	sampler "github.com/ollama/ollama/x/mlxrunner/sample"
)

// genSlot is one parallel generation sequence in the continuous batcher.
type genSlot struct {
	id         int
	req        Request
	position   int
	promptLen  int
	generated  int
	promptEval time.Duration
	detok      detokenizer
	done       bool
	err        error
	// nextInput is the token id array shape [1] to feed on the next decode step.
	nextInput *mlx.Array
	started   time.Time
}

// runContinuousBatcher admits up to numParallel requests and fuses decode
// steps across active slots. Prefills run one slot at a time into a shared
// MultiSeq cache. Speculation and prefix-trie reuse are disabled.
//
// ponytail: rotating/recurrent caches are not wrapped — only enabled when
// WrapParallelCaches succeeded.
func (r *Runner) runContinuousBatcher(ctx context.Context) error {
	slots := make([]*genSlot, r.numParallel)
	caches := r.parallelCaches

	for {
		select {
		case <-ctx.Done():
			return nil
		default:
		}

		admitted := false
		for i := range slots {
			if slots[i] != nil {
				continue
			}
			select {
			case <-ctx.Done():
				return nil
			case req := <-r.Requests:
				admitted = true
				if err := r.resetParallelSeq(caches, i); err != nil {
					r.failRequest(req, err)
					continue
				}
				slot, err := r.prefillSlot(req.Ctx, caches, i, req)
				if err != nil {
					r.failRequest(req, err)
					continue
				}
				slots[i] = slot
			default:
			}
		}

		active := make([]*genSlot, 0, r.numParallel)
		for _, s := range slots {
			if s != nil && !s.done {
				active = append(active, s)
			}
		}
		if len(active) == 0 {
			if admitted {
				continue
			}
			select {
			case <-ctx.Done():
				return nil
			case req := <-r.Requests:
				if err := r.resetParallelSeq(caches, 0); err != nil {
					r.failRequest(req, err)
					continue
				}
				slot, err := r.prefillSlot(req.Ctx, caches, 0, req)
				if err != nil {
					r.failRequest(req, err)
					continue
				}
				slots[0] = slot
			}
			continue
		}

		if err := r.decodeBatchStep(caches, active); err != nil {
			for _, s := range active {
				if !s.done {
					s.err = err
					s.done = true
				}
			}
		}

		for i, s := range slots {
			if s == nil || !s.done {
				continue
			}
			r.finishSlot(s)
			r.Sampler.Remove(s.id)
			slots[i] = nil
		}
	}
}

func (r *Runner) failRequest(req Request, err error) {
	slog.Info("Request terminated", "error", err)
	var statusErr api.StatusError
	if !errors.As(err, &statusErr) {
		statusErr = api.StatusError{
			StatusCode:   http.StatusInternalServerError,
			ErrorMessage: err.Error(),
		}
	}
	select {
	case req.Responses <- CompletionResponse{Error: &statusErr}:
	case <-req.Ctx.Done():
	}
	close(req.Responses)
}

func (r *Runner) resetParallelSeq(caches []cache.Cache, seq int) error {
	for _, c := range caches {
		ms, ok := c.(*cache.MultiSeq)
		if !ok {
			return fmt.Errorf("expected MultiSeq cache")
		}
		if err := ms.ResetSeq(seq); err != nil {
			return err
		}
	}
	return nil
}

func (r *Runner) prefillSlot(ctx context.Context, caches []cache.Cache, seq int, req Request) (*genSlot, error) {
	mlx.ResetPeakMemory()
	tokens := req.Tokens
	position := 0
	start := time.Now()
	prefillChunk := prefillChunkSize()

	materialize := func() {
		state := make([]*mlx.Array, 0, 2*len(caches))
		for _, c := range caches {
			state = append(state, c.State()...)
		}
		if len(state) > 0 {
			mlx.Eval(state...)
		}
	}

	total, processed := len(tokens), 0
	for total-processed > 1 {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		n := min(prefillChunk, total-processed-1)
		chunkIDs := mlx.FromValues(tokens[processed:processed+n], 1, n)
		_ = r.Model.Forward(&batch.Batch{
			InputIDs:     chunkIDs,
			SeqOffsets:   []int32{int32(position)},
			SeqQueryLens: []int32{int32(n)},
			SeqIDs:       []int32{int32(seq)},
		}, caches)
		mlx.Sweep()
		materialize()
		processed += n
		position += n
		slog.Info("Prompt processing progress", "slot", seq, "processed", processed, "total", total)
		mlx.ClearCache()
	}

	seed := mlx.NewArrayInt32([]int32{tokens[processed]}, []int32{1})
	r.Sampler.Add(seq, req.SamplerOpts, req.Tokens)

	return &genSlot{
		id:         seq,
		req:        req,
		position:   position,
		promptLen:  len(req.Tokens),
		promptEval: time.Since(start),
		nextInput:  seed,
		started:    time.Now(),
		detok: detokenizer{
			tokenizer:       r.Tokenizer,
			wantLogprobs:    req.SamplerOpts.Logprobs,
			wantTopLogprobs: req.SamplerOpts.TopLogprobs,
		},
	}, nil
}

func (r *Runner) decodeBatchStep(caches []cache.Cache, active []*genSlot) error {
	B := len(active)
	tokenIDs := make([]int32, B)
	offsets := make([]int32, B)
	qLens := make([]int32, B)
	seqIDs := make([]int32, B)
	slotIDs := make([]int, B)

	for i, s := range active {
		if err := s.req.Ctx.Err(); err != nil {
			s.done = true
			s.err = err
			continue
		}
		tokenIDs[i] = int32(s.nextInput.Int())
		offsets[i] = int32(s.position)
		qLens[i] = 1
		seqIDs[i] = int32(s.id)
		slotIDs[i] = s.id
	}

	// Drop slots cancelled before the forward.
	alive := active[:0]
	aliveIDs := tokenIDs[:0]
	aliveOff := offsets[:0]
	aliveQ := qLens[:0]
	aliveSeq := seqIDs[:0]
	aliveSlots := slotIDs[:0]
	for i, s := range active {
		if s.done {
			continue
		}
		alive = append(alive, s)
		aliveIDs = append(aliveIDs, tokenIDs[i])
		aliveOff = append(aliveOff, offsets[i])
		aliveQ = append(aliveQ, qLens[i])
		aliveSeq = append(aliveSeq, seqIDs[i])
		aliveSlots = append(aliveSlots, slotIDs[i])
	}
	active = alive
	if len(active) == 0 {
		return nil
	}
	B = len(active)

	input := mlx.FromValues(aliveIDs, B, 1)
	hidden := r.Model.Forward(&batch.Batch{
		InputIDs:     input,
		SeqOffsets:   aliveOff,
		SeqQueryLens: aliveQ,
		SeqIDs:       aliveSeq,
	}, caches)
	for _, s := range active {
		s.position++
	}

	logits := r.Model.Unembed(hidden)
	rowLogits := logits.Slice(mlx.Slice(), mlx.Slice(logits.Dim(1)-1), mlx.Slice()).Squeeze(1)
	next := r.Sampler.Sample(aliveSlots, rowLogits)
	mlx.Pin(next.Arrays()...)
	mlx.Sweep()
	mlx.Eval(next.Arrays()...)

	for i, s := range active {
		tok := next.Token.Slice(mlx.Slice(i)).Squeeze(0)
		id := int32(tok.Int())
		s.nextInput = mlx.NewArrayInt32([]int32{id}, []int32{1})

		if r.Tokenizer.IsEOS(id) {
			s.done = true
			r.emitFinal(s, 0)
			continue
		}
		s.generated++
		res := sampler.Result{Token: tok}
		if next.Logprob != nil {
			res.Logprob = next.Logprob.Slice(mlx.Slice(i), mlx.Slice())
		}
		resp, ok := s.detok.detokenize(res)
		if ok {
			select {
			case <-s.req.Ctx.Done():
				s.done = true
				s.err = s.req.Ctx.Err()
				continue
			case s.req.Responses <- resp:
			}
		}
		if s.generated >= s.req.Options.NumPredict {
			s.done = true
			r.emitFinal(s, 1)
		}
	}
	mlx.Unpin(next.Arrays()...)
	mlx.ClearCache()
	return nil
}

func (r *Runner) emitFinal(s *genSlot, reason int) {
	final := CompletionResponse{
		Done:               true,
		PromptEvalCount:    s.promptLen,
		EvalCount:          s.generated,
		DoneReason:         reason,
		PromptEvalDuration: s.promptEval,
		EvalDuration:       time.Since(s.started),
	}
	select {
	case <-s.req.Ctx.Done():
		s.err = s.req.Ctx.Err()
	case s.req.Responses <- final:
	}
}

func (r *Runner) finishSlot(s *genSlot) {
	close(s.req.Responses)
	slog.Info("peak memory", "slot", s.id, "size", mlx.PrettyBytes(mlx.PeakMemory()))
}
