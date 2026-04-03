package mlxrunner

import (
	"bytes"
	"context"
	"errors"
	"log/slog"
	"net/http"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

// activeSeq tracks a single sequence in the decode batch.
type activeSeq struct {
	seqID   int
	session *cacheSession
	request Request

	// Decode state — pinned arrays from the previous step.
	sample, logprobs *mlx.Array

	buf       bytes.Buffer
	generated int
	final     CompletionResponse
	decodeAt  time.Time // set after prefill completes
}

func (s *activeSeq) cleanup() {
	if s.request.Sampler != nil {
		s.request.Sampler.Free()
	}
	mlx.Unpin(s.sample, s.logprobs)
}

const maxParallel = 4

// scheduler manages prefill and decode for all active sequences.
type scheduler struct {
	runner *Runner
	active []*activeSeq
	used   [maxParallel]bool // seqID slot allocation
}

func (r *Runner) newScheduler() *scheduler {
	return &scheduler{runner: r}
}

// allocSeqID returns the lowest free seqID slot.
func (s *scheduler) allocSeqID() int {
	for i, used := range s.used {
		if !used {
			s.used[i] = true
			return i
		}
	}
	panic("no free sequence slots")
}

// freeSeqID returns a seqID slot to the pool.
func (s *scheduler) freeSeqID(seqID int) {
	s.used[seqID] = false
}

func (s *scheduler) run(ctx context.Context) error {
	r := s.runner

	enableCompile := true
	if modelCompile, ok := r.Model.(interface{ EnableCompile() bool }); ok {
		enableCompile = modelCompile.EnableCompile()
	}
	if enableCompile {
		mlx.EnableCompile()
	} else {
		mlx.DisableCompile()
	}

	for {
		if len(s.active) == 0 {
			// No active sequences — block waiting for a request.
			select {
			case <-ctx.Done():
				return nil
			case request := <-r.Requests:
				s.admitRequest(ctx, request)
			}
		} else {
			// Active sequences decoding — check for new requests non-blocking.
			select {
			case <-ctx.Done():
				s.finishAll()
				return nil
			case request := <-r.Requests:
				s.admitRequest(ctx, request)
			default:
			}

			// Run one decode step for all active sequences.
			s.decodeStep(ctx)
		}
	}
}

// admitRequest prefills a new request and adds it to the decode batch.
func (s *scheduler) admitRequest(ctx context.Context, request Request) {
	mlx.ResetPeakMemory()

	seqID := s.allocSeqID()

	seq := &activeSeq{
		seqID:   seqID,
		request: request,
		final: CompletionResponse{
			Done:            true,
			PromptEvalCount: len(request.Tokens),
			EvalCount:       request.Options.MaxTokens,
			DoneReason:      1,
		},
	}

	// Ensure caches exist with all pool slots registered. SetSeqs is
	// a no-op after the first call since the slot set never changes.
	s.runner.cache.ensureCaches(s.runner.Model)
	allSlots := make([]int, maxParallel)
	for i := range allSlots {
		allSlots[i] = i
	}
	for _, kv := range s.runner.cache.caches {
		if kv != nil {
			kv.SetSeqs(allSlots)
		}
	}

	if err := s.prefill(ctx, seq); err != nil {
		slog.Info("Prefill failed", "seq", seqID, "error", err)
		seq.cleanup()
		s.freeSeqID(seqID)
		s.sendError(request, err)
		return
	}

	// Materialize all cache state so existing sequences' decode steps
	// see clean buffer data (not lazy graphs from prefill/restore).
	s.materializeCaches()

	s.active = append(s.active, seq)
}

func (s *scheduler) prefill(ctx context.Context, seq *activeSeq) error {
	r := s.runner
	inputs := seq.request.Tokens
	seq.request.Sampler.ResetHistory(inputs)

	session := r.cache.begin(seq.seqID, r.Model, inputs)
	seq.session = session

	caches := session.caches
	tokens := session.remaining

	// Schedule periodic snapshots during prefill.
	const snapshotInterval = 8192
	for offset := snapshotInterval; offset < len(inputs); offset += snapshotInterval {
		session.requestSnapshot(offset)
	}
	const preThinking = 4
	if end := len(inputs) - preThinking; end > 0 {
		session.requestSnapshot(end)
	}

	prefillChunk := prefillChunkSize()
	total, processed := len(tokens), 0
	for total-processed > 1 {
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := seq.request.Ctx.Err(); err != nil {
			return err
		}

		n := min(prefillChunk, total-processed-1)

		if snapOffset := session.nextPendingSnapshot(); snapOffset > 0 {
			baseOffset := len(session.inputs) - len(tokens)
			tokensUntilSnapshot := snapOffset - (baseOffset + processed)
			if tokensUntilSnapshot > 0 && tokensUntilSnapshot < n {
				n = tokensUntilSnapshot
			}
		}

		r.Model.Forward(&batch.ForwardBatch{
			InputIDs: mlx.FromValues(tokens[processed:processed+n], n).ExpandDims(0),
			SeqIDs:   []int{seq.seqID},
			SeqLens:  []int{n},
		}, caches)
		mlx.Sweep()
		s.materializeCaches()
		processed += n
		slog.Info("Prompt processing progress", "seq", seq.seqID, "processed", processed, "total", total)

		if snapOffset := session.nextPendingSnapshot(); snapOffset > 0 {
			baseOffset := len(session.inputs) - len(tokens)
			if baseOffset+processed >= snapOffset {
				session.snapshot()
			}
		}

		mlx.ClearCache()
	}

	// First decode step: process final token(s) and get initial sample.
	// Eval the sample AND the cache state so everything is materialized
	// before any cache transitions (snapshot/restore/rebuild).
	seq.sample, seq.logprobs = s.singleStep(seq, mlx.FromValues(tokens[processed:], total-processed))
	evalArrays := []*mlx.Array{seq.sample, seq.logprobs}
	for _, c := range caches {
		evalArrays = append(evalArrays, c.State()...)
	}
	mlx.Eval(evalArrays...)
	seq.decodeAt = time.Now()

	return nil
}

// singleStep runs a single-sequence forward+sample (used during prefill's
// final token and as fallback).
func (s *scheduler) singleStep(seq *activeSeq, token *mlx.Array) (*mlx.Array, *mlx.Array) {
	r := s.runner
	caches := seq.session.caches

	fwd := r.Model.Forward(&batch.ForwardBatch{
		InputIDs: token.ExpandDims(0),
		SeqIDs:   []int{seq.seqID},
		SeqLens:  []int{1},
	}, caches)
	logits := r.Model.Unembed(fwd)
	logits = logits.Slice(mlx.Slice(), mlx.Slice(logits.Dim(1)-1), mlx.Slice()).Squeeze(1)

	logprobs := logits.Subtract(logits.Logsumexp(true))
	sample := seq.request.Sampler.Sample(logprobs)

	mlx.Pin(sample, logprobs)
	mlx.Sweep()
	mlx.AsyncEval(sample, logprobs)

	return sample, logprobs
}

// decodeStep runs one batched decode iteration for all active sequences.
func (s *scheduler) decodeStep(ctx context.Context) {
	r := s.runner

	// Check for cancelled sequences and remove them.
	s.reapCancelled(ctx)
	if len(s.active) == 0 {
		return
	}

	// Read token values from previous step's samples. This forces
	// evaluation of the lazy computation from the prior step.
	inputTokens := make([]int32, len(s.active))
	for i, seq := range s.active {
		if seq.generated == 0 {
			mlx.Eval(seq.sample)
			seq.final.PromptEvalDuration = time.Since(seq.decodeAt)
			seq.decodeAt = time.Now()
		}
		inputTokens[i] = int32(seq.sample.Int())
	}

	// Process previous step's outputs: stream tokens, check EOS.
	var completed []*activeSeq
	for i, seq := range s.active {
		output := inputTokens[i]
		seq.session.outputs = append(seq.session.outputs, output)
		seq.generated++

		if r.Tokenizer.IsEOS(output) {
			seq.final.DoneReason = 0
			seq.final.EvalCount = seq.generated - 1
			completed = append(completed, seq)
			continue
		}

		if seq.generated >= seq.request.Options.MaxTokens {
			seq.final.EvalCount = seq.generated
			completed = append(completed, seq)
			continue
		}

		// Stream token to client.
		select {
		case <-seq.request.Ctx.Done():
			completed = append(completed, seq)
		case seq.request.Responses <- CompletionResponse{
			Content: r.Decode(output, &seq.buf),
		}:
		}
	}

	// Finish completed sequences and remove from active list.
	if len(completed) > 0 {
		completedSet := make(map[int]bool, len(completed))
		for _, seq := range completed {
			s.finishSeq(seq)
			completedSet[seq.seqID] = true
		}
		alive := s.active[:0]
		for _, seq := range s.active {
			if !completedSet[seq.seqID] {
				alive = append(alive, seq)
			}
		}
		s.active = alive
		mlx.ClearCache()
	}

	if len(s.active) == 0 {
		return
	}

	// Batched forward pass: one token per sequence.
	seqIDs := make([]int, len(s.active))
	seqLens := make([]int, len(s.active))
	nextTokens := make([]int32, len(s.active))
	for i, seq := range s.active {
		seq.request.Sampler.AppendToken(seq.sample)
		nextTokens[i] = int32(seq.sample.Int())
		seqIDs[i] = seq.seqID
		seqLens[i] = 1
		mlx.Unpin(seq.sample, seq.logprobs)
		seq.sample, seq.logprobs = nil, nil
	}

	fwd := r.Model.Forward(&batch.ForwardBatch{
		InputIDs: mlx.FromValues(nextTokens, len(nextTokens)).ExpandDims(0),
		SeqIDs:   seqIDs,
		SeqLens:  seqLens,
	}, r.cache.caches)
	logits := r.Model.Unembed(fwd)

	for i, seq := range s.active {
		seqLogits := logits.Slice(mlx.Slice(), mlx.Slice(i, i+1), mlx.Slice()).Squeeze(1)
		lp := seqLogits.Subtract(seqLogits.Logsumexp(true))
		sample := seq.request.Sampler.Sample(lp)
		mlx.Pin(sample, lp)
		seq.sample = sample
		seq.logprobs = lp
	}

	mlx.Sweep()

	evalArrays := make([]*mlx.Array, 0, 2*len(s.active))
	for _, seq := range s.active {
		evalArrays = append(evalArrays, seq.sample, seq.logprobs)
	}
	mlx.AsyncEval(evalArrays...)
}

// reapCancelled removes sequences whose request context has been cancelled.
func (s *scheduler) reapCancelled(ctx context.Context) {
	var alive []*activeSeq
	for _, seq := range s.active {
		if ctx.Err() != nil || seq.request.Ctx.Err() != nil {
			s.finishSeq(seq)
		} else {
			alive = append(alive, seq)
		}
	}
	if len(alive) != len(s.active) {
		s.active = alive
	}
}

// finishSeq sends the final response, saves to trie, and cleans up.
// It does NOT remove from s.active — the caller is responsible for that.
func (s *scheduler) finishSeq(seq *activeSeq) {
	seq.final.EvalDuration = time.Since(seq.decodeAt)

	// Send final response.
	if seq.request.Ctx.Err() == nil {
		select {
		case seq.request.Responses <- seq.final:
		case <-seq.request.Ctx.Done():
		}
	}

	// Save to trie and clean up.
	if seq.session != nil && seq.generated > 0 {
		seq.session.close()
	}
	s.freeSeqID(seq.seqID)
	seq.cleanup()
	close(seq.request.Responses)

	if slog.Default().Enabled(context.TODO(), logutil.LevelTrace) {
		s.runner.cache.dumpTree()
	}
	slog.Info("sequence complete", "seq", seq.seqID, "generated", seq.generated,
		"peak_memory", mlx.PrettyBytes(mlx.PeakMemory()))
}

func (s *scheduler) sendError(request Request, err error) {
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
	close(request.Responses)
}

func (s *scheduler) finishAll() {
	for _, seq := range s.active {
		s.finishSeq(seq)
	}
	s.active = nil
}

func (s *scheduler) materializeCaches() {
	state := make([]*mlx.Array, 0, 2*len(s.runner.cache.caches))
	for _, c := range s.runner.cache.caches {
		state = append(state, c.State()...)
	}
	if len(state) == 0 {
		return
	}
	mlx.Eval(state...)
}
