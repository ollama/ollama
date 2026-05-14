package mlxrunner

import (
	"context"
	"fmt"
	"log/slog"
	"time"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
	sampler "github.com/ollama/ollama/x/mlxrunner/sample"
)

type dflashStats struct {
	iterations       int
	drafted          int
	accepted         int
	mismatches       int
	allAccepted      int
	batched          int
	serial           int
	targetDuration   time.Duration
	draftDuration    time.Duration
	validateDuration time.Duration
}

type dflashDecodeMode string

const (
	dflashDecodeDisabled dflashDecodeMode = ""
	dflashDecodeGreedy   dflashDecodeMode = "greedy"
	dflashDecodeSample   dflashDecodeMode = "sample"
)

func (m dflashDecodeMode) enabled() bool {
	return m != dflashDecodeDisabled
}

func newDFlashTargetCaches(m base.Model) []cache.Cache {
	if cacheFactory, ok := m.(interface{ NewCaches() []cache.Cache }); ok {
		return cacheFactory.NewCaches()
	}
	caches := make([]cache.Cache, m.NumLayers())
	for i := range caches {
		caches[i] = cache.NewKVCache()
	}
	return caches
}

func freeCacheSet(caches []cache.Cache) {
	for _, c := range caches {
		if c != nil {
			c.Free()
		}
	}
}

func (r *Runner) dflashGate(opts sampler.Options) (dflashDecodeMode, string) {
	if r.Draft == nil {
		return dflashDecodeDisabled, "no_draft"
	}
	if _, ok := r.Draft.(base.DFlashDraftModel); !ok {
		return dflashDecodeDisabled, "draft_not_dflash"
	}
	if _, ok := r.Model.(base.DFlashTargetModel); !ok {
		return dflashDecodeDisabled, "target_not_dflash"
	}
	if _, ok := r.Model.(base.MTPEmbeddingModel); !ok {
		return dflashDecodeDisabled, "target_embeddings_missing"
	}
	if opts.Logprobs || opts.TopLogprobs > 0 {
		return dflashDecodeDisabled, "logprobs_requested"
	}

	if opts.Temperature > 0 || dflashUsesSamplerHistory(opts) {
		return dflashDecodeSample, ""
	}

	return dflashDecodeGreedy, ""
}

func dflashUsesSamplerHistory(opts sampler.Options) bool {
	if opts.RepeatLastN == 0 {
		return false
	}

	repeatPenalty := opts.RepeatPenalty
	if repeatPenalty <= 0 {
		repeatPenalty = 1
	}
	return repeatPenalty != 1 || opts.PresencePenalty != 0 || opts.FrequencyPenalty != 0
}

func (r *Runner) runGreedyDFlashDecode(ctx context.Context, request Request, session *cacheSession, targetCaches []cache.Cache, draftCaches []cache.Cache, seed []int32, position *int, started time.Time) error {
	target := r.Model.(base.DFlashTargetModel)
	draft := r.Draft.(base.DFlashDraftModel)
	stats := dflashStats{}
	slog.Info("DFlash greedy decode enabled", "block_size", draft.BlockSize(), "target_layers", draft.TargetLayerIDs())

	targetForward := func(token *mlx.Array) (*mlx.Array, *mlx.Array) {
		hidden, targetHidden := target.ForwardDFlash(&batch.Batch{
			InputIDs:     token,
			SeqOffsets:   []int32{int32(*position)},
			SeqQueryLens: []int32{int32(token.Dim(1))},
		}, targetCaches, draft.TargetLayerIDs())
		*position += token.Dim(1)
		return hidden, targetHidden
	}

	t0 := time.Now()
	hidden, targetHidden := targetForward(mlx.FromValues(seed, 1, len(seed)))
	draft.AppendContext(targetHidden, draftCaches)
	current := sampler.Result{Token: greedyTokenFromLogits(r.lastLogits(hidden))}
	mlx.Pin(current.Arrays()...)
	mlx.Sweep()
	mlx.AsyncEval(current.Arrays()...)
	stats.targetDuration += time.Since(t0)
	defer func() {
		mlx.Unpin(current.Arrays()...)
	}()

	dec := decoder{tokenizer: r.Tokenizer}
	final := CompletionResponse{Done: true, PromptEvalCount: len(request.Tokens), DoneReason: 1}
	now := started
	generated := 0

	for generated < request.Options.NumPredict {
		if err := ctx.Err(); err != nil {
			return err
		}

		if generated == 0 {
			mlx.Eval(current.Arrays()...)
			final.PromptEvalDuration = time.Since(now)
			now = time.Now()
		}

		done, err := r.emitMTPToken(ctx, request, session, &dec, current, &final)
		if err != nil {
			return err
		}
		if !done {
			generated++
		}
		if done || generated >= request.Options.NumPredict {
			break
		}

		draftCount := min(draft.BlockSize()-1, request.Options.NumPredict-generated)
		if draftCount <= 0 {
			t0 = time.Now()
			hidden, targetHidden := targetForward(mtpTokenInput(current.Token))
			draft.AppendContext(targetHidden, draftCaches)
			stats.targetDuration += time.Since(t0)
			next := sampler.Result{Token: greedyTokenFromLogits(r.lastLogits(hidden))}
			mlx.Pin(next.Arrays()...)
			old := current
			current = next
			mlx.Unpin(old.Arrays()...)
			mlx.Sweep()
			mlx.AsyncEval(current.Arrays()...)
			continue
		}

		stats.iterations++
		t0 = time.Now()
		draftTokens := r.generateDFlashDrafts(draft, current.Token, draftCaches, draftCount)
		mlx.Pin(draftTokens)
		mlx.Eval(draftTokens)
		stats.draftDuration += time.Since(t0)
		stats.drafted += draftCount

		t0 = time.Now()
		next, accepted, done, err := r.acceptDFlashDrafts(ctx, request, session, &dec, target, draft, targetCaches, draftCaches, position, current, draftTokens, &final, &generated, &stats)
		stats.validateDuration += time.Since(t0)
		mlx.Unpin(draftTokens)
		if err != nil {
			return err
		}
		stats.accepted += accepted
		if accepted == draftCount {
			stats.allAccepted++
		} else {
			stats.mismatches++
		}
		if done || generated >= request.Options.NumPredict {
			break
		}

		mlx.Pin(next.Arrays()...)
		old := current
		current = next
		mlx.Unpin(old.Arrays()...)
		mlx.Sweep()
		mlx.AsyncEval(current.Arrays()...)

		if generated%256 == 0 {
			mlx.ClearCache()
		}
	}

	final.EvalCount = generated
	final.EvalDuration = time.Since(now)
	acceptance := 0.0
	if stats.drafted > 0 {
		acceptance = float64(stats.accepted) / float64(stats.drafted)
	}
	avgDraft := 0.0
	avgAccepted := 0.0
	if stats.iterations > 0 {
		avgDraft = float64(stats.drafted) / float64(stats.iterations)
		avgAccepted = float64(stats.accepted) / float64(stats.iterations)
	}
	slog.Info("DFlash decode stats", "mode", "greedy", "generated", generated, "drafted", stats.drafted, "accepted", stats.accepted, "acceptance", acceptance, "iterations", stats.iterations, "avg_draft", avgDraft, "avg_accepted", avgAccepted, "batched", stats.batched, "serial", stats.serial, "mismatches", stats.mismatches, "all_accepted", stats.allAccepted, "max_draft", draft.BlockSize()-1, "block_size", draft.BlockSize(), "target_layers", draft.TargetLayerIDs(), "target_duration", stats.targetDuration, "draft_duration", stats.draftDuration, "validate_duration", stats.validateDuration)

	select {
	case <-ctx.Done():
		return ctx.Err()
	case request.Responses <- final:
		return nil
	}
}

func (r *Runner) runSampleDFlashDecode(ctx context.Context, request Request, session *cacheSession, targetCaches []cache.Cache, draftCaches []cache.Cache, seed []int32, position *int, started time.Time) error {
	target := r.Model.(base.DFlashTargetModel)
	draft := r.Draft.(base.DFlashDraftModel)
	stats := dflashStats{}
	slog.Info("DFlash sample decode enabled",
		"block_size", draft.BlockSize(),
		"target_layers", draft.TargetLayerIDs(),
		"temperature", request.SamplerOpts.Temperature,
		"top_p", request.SamplerOpts.TopP,
		"top_k", request.SamplerOpts.TopK,
		"min_p", request.SamplerOpts.MinP,
		"repeat_penalty", request.SamplerOpts.RepeatPenalty,
		"presence_penalty", request.SamplerOpts.PresencePenalty,
		"frequency_penalty", request.SamplerOpts.FrequencyPenalty,
	)

	targetForward := func(token *mlx.Array) (*mlx.Array, *mlx.Array) {
		hidden, targetHidden := target.ForwardDFlash(&batch.Batch{
			InputIDs:     token,
			SeqOffsets:   []int32{int32(*position)},
			SeqQueryLens: []int32{int32(token.Dim(1))},
		}, targetCaches, draft.TargetLayerIDs())
		*position += token.Dim(1)
		return hidden, targetHidden
	}

	t0 := time.Now()
	hidden, targetHidden := targetForward(mlx.FromValues(seed, 1, len(seed)))
	draft.AppendContext(targetHidden, draftCaches)
	current := r.Sampler.Sample([]int{pipelineSlot}, r.lastLogits(hidden))
	mlx.Pin(current.Arrays()...)
	mlx.Sweep()
	mlx.AsyncEval(current.Arrays()...)
	stats.targetDuration += time.Since(t0)
	defer func() {
		mlx.Unpin(current.Arrays()...)
	}()

	dec := decoder{tokenizer: r.Tokenizer}
	final := CompletionResponse{Done: true, PromptEvalCount: len(request.Tokens), DoneReason: 1}
	now := started
	generated := 0

	for generated < request.Options.NumPredict {
		if err := ctx.Err(); err != nil {
			return err
		}

		if generated == 0 {
			mlx.Eval(current.Arrays()...)
			final.PromptEvalDuration = time.Since(now)
			now = time.Now()
		}

		done, err := r.emitMTPToken(ctx, request, session, &dec, current, &final)
		if err != nil {
			return err
		}
		if !done {
			generated++
		}
		if done || generated >= request.Options.NumPredict {
			break
		}

		draftCount := min(draft.BlockSize()-1, request.Options.NumPredict-generated)
		if draftCount <= 0 {
			t0 = time.Now()
			hidden, targetHidden := targetForward(mtpTokenInput(current.Token))
			draft.AppendContext(targetHidden, draftCaches)
			stats.targetDuration += time.Since(t0)
			next := r.Sampler.Sample([]int{pipelineSlot}, r.lastLogits(hidden))
			mlx.Pin(next.Arrays()...)
			old := current
			current = next
			mlx.Unpin(old.Arrays()...)
			mlx.Sweep()
			mlx.AsyncEval(current.Arrays()...)
			continue
		}

		stats.iterations++
		t0 = time.Now()
		candidates := r.generateDFlashDraftCandidates(draft, current.Token, draftCaches, draftCount)
		var candidateArrays []*mlx.Array
		if candidates != nil {
			draftCount = candidates.tokens.Dim(1)
			candidateArrays = candidates.Arrays()
			mlx.Pin(candidateArrays...)
			mlx.Sweep()
		}
		stats.draftDuration += time.Since(t0)
		stats.drafted += draftCount

		var next sampler.Result
		if draftCount == 0 {
			t0 = time.Now()
			hidden, targetHidden := targetForward(mtpTokenInput(current.Token))
			draft.AppendContext(targetHidden, draftCaches)
			stats.targetDuration += time.Since(t0)
			next = r.Sampler.Sample([]int{pipelineSlot}, r.lastLogits(hidden))
		} else {
			var accepted int
			t0 = time.Now()
			next, accepted, done, err = r.acceptSampleDFlashDrafts(ctx, request, session, &dec, target, draft, targetCaches, draftCaches, position, current, candidates, &final, &generated, &stats)
			stats.validateDuration += time.Since(t0)
			mlx.Unpin(candidateArrays...)
			if err != nil {
				return err
			}
			stats.accepted += accepted
			if accepted == draftCount {
				stats.allAccepted++
			} else {
				stats.mismatches++
			}
			if next.Token == nil {
				mlx.Sweep()
			}
			if done || generated >= request.Options.NumPredict {
				break
			}
		}

		mlx.Pin(next.Arrays()...)
		old := current
		current = next
		mlx.Unpin(old.Arrays()...)
		mlx.Sweep()
		mlx.AsyncEval(current.Arrays()...)

		if generated%256 == 0 {
			mlx.ClearCache()
		}
	}

	final.EvalCount = generated
	final.EvalDuration = time.Since(now)
	acceptance := 0.0
	if stats.drafted > 0 {
		acceptance = float64(stats.accepted) / float64(stats.drafted)
	}
	avgDraft := 0.0
	avgAccepted := 0.0
	if stats.iterations > 0 {
		avgDraft = float64(stats.drafted) / float64(stats.iterations)
		avgAccepted = float64(stats.accepted) / float64(stats.iterations)
	}
	slog.Info("DFlash decode stats", "mode", "sample", "generated", generated, "drafted", stats.drafted, "accepted", stats.accepted, "acceptance", acceptance, "iterations", stats.iterations, "avg_draft", avgDraft, "avg_accepted", avgAccepted, "batched", stats.batched, "serial", stats.serial, "mismatches", stats.mismatches, "all_accepted", stats.allAccepted, "max_draft", draft.BlockSize()-1, "block_size", draft.BlockSize(), "target_layers", draft.TargetLayerIDs(), "target_duration", stats.targetDuration, "draft_duration", stats.draftDuration, "validate_duration", stats.validateDuration)

	select {
	case <-ctx.Done():
		return ctx.Err()
	case request.Responses <- final:
		return nil
	}
}

func (r *Runner) dflashDraftLogits(draft base.DFlashDraftModel, current *mlx.Array, caches []cache.Cache, draftCount int) *mlx.Array {
	blockLen := draftCount + 1
	values := make([]int32, blockLen)
	values[0] = int32(tokenID(current))
	for i := 1; i < blockLen; i++ {
		values[i] = draft.MaskTokenID()
	}
	block := mlx.FromValues(values, 1, blockLen)
	logits := draft.Draft(block, caches)
	return logits.Slice(mlx.Slice(), mlx.Slice(1, blockLen), mlx.Slice())
}

func (r *Runner) generateDFlashDrafts(draft base.DFlashDraftModel, current *mlx.Array, caches []cache.Cache, draftCount int) *mlx.Array {
	logits := r.dflashDraftLogits(draft, current, caches, draftCount)
	return logits.Argmax(-1, false).AsType(mlx.DTypeInt32)
}

type dflashDraftCandidates struct {
	tokens *mlx.Array
	dist   sampler.Distribution
}

func (c *dflashDraftCandidates) Arrays() []*mlx.Array {
	if c == nil {
		return nil
	}
	return append([]*mlx.Array{c.tokens}, c.dist.Arrays()...)
}

func (r *Runner) generateDFlashDraftCandidates(draft base.DFlashDraftModel, current *mlx.Array, caches []cache.Cache, draftCount int) *dflashDraftCandidates {
	if draftCount <= 0 {
		return nil
	}

	logits := r.dflashDraftLogits(draft, current, caches, draftCount)
	draftTokens := make([]*mlx.Array, 0, draftCount)
	draftDists := make([]sampler.Distribution, 0, draftCount)
	var prefix *mlx.Array

	for i := range draftCount {
		rows := logits.Slice(mlx.Slice(), mlx.Slice(0, i+1), mlx.Slice())
		dist := r.Sampler.Distribution(pipelineSlot, rows, prefix).SliceRows(i, i+1)
		nextToken := mtpTokenVector(r.Sampler.SampleDistribution(pipelineSlot, dist))
		nextInput := mtpTokenInput(nextToken)

		draftTokens = append(draftTokens, nextInput)
		draftDists = append(draftDists, dist)
		if prefix == nil {
			prefix = nextInput
		} else {
			prefix = prefix.Concatenate(1, nextInput)
		}
	}
	if len(draftTokens) == 0 {
		return nil
	}
	return &dflashDraftCandidates{
		tokens: mlx.Concatenate(draftTokens, 1),
		dist:   sampler.ConcatenateDistributions(draftDists),
	}
}

func (r *Runner) acceptDFlashDrafts(ctx context.Context, request Request, session *cacheSession, dec *decoder, target base.DFlashTargetModel, draft base.DFlashDraftModel, targetCaches []cache.Cache, draftCaches []cache.Cache, position *int, current sampler.Result, draftTokens *mlx.Array, final *CompletionResponse, generated *int, stats *dflashStats) (sampler.Result, int, bool, error) {
	specCaches, spec, ok := cache.BeginSpeculation(targetCaches)
	if !ok {
		stats.serial++
		return r.acceptDFlashDraftsSerial(ctx, request, session, dec, target, draft, targetCaches, draftCaches, position, current, draftTokens, final, generated)
	}
	stats.batched++
	return r.acceptDFlashDraftsBatched(ctx, request, session, dec, target, draft, specCaches, spec, draftCaches, position, current, draftTokens, final, generated)
}

func (r *Runner) acceptDFlashDraftsBatched(ctx context.Context, request Request, session *cacheSession, dec *decoder, target base.DFlashTargetModel, draft base.DFlashDraftModel, specCaches []cache.Cache, spec *cache.Speculation, draftCaches []cache.Cache, position *int, current sampler.Result, draftTokens *mlx.Array, final *CompletionResponse, generated *int) (sampler.Result, int, bool, error) {
	before := *position
	draftCount := draftTokens.Dim(1)
	verifyInput := mtpTokenInput(current.Token).Concatenate(1, draftTokens)
	hiddenSeq, targetHiddenSeq := target.ForwardDFlash(&batch.Batch{
		InputIDs:     verifyInput,
		SeqOffsets:   []int32{int32(before)},
		SeqQueryLens: []int32{int32(verifyInput.Dim(1))},
	}, specCaches, draft.TargetLayerIDs())

	selectedTokens := r.Model.Unembed(hiddenSeq).Argmax(-1, false).AsType(mlx.DTypeInt32)
	mlx.Eval(draftTokens, selectedTokens)

	draftIDs := draftTokens.Ints()
	selectedIDs := selectedTokens.Ints()
	if len(selectedIDs) < draftCount+1 {
		spec.Commit(0)
		return sampler.Result{}, 0, false, fmt.Errorf("dflash validation produced %d tokens for %d draft tokens", len(selectedIDs), draftCount)
	}

	accepted := 0
	for i, id := range draftIDs {
		if selectedIDs[i] != id {
			break
		}
		accepted++
		if r.Tokenizer.IsEOS(int32(id)) {
			break
		}
	}

	commitN := accepted + 1
	spec.Commit(0)

	done := false
	for _, id := range draftIDs[:accepted] {
		if *generated >= request.Options.NumPredict {
			done = true
			break
		}
		res := sampler.Result{Token: mlx.FromValues([]int32{int32(id)}, 1)}
		var err error
		done, err = r.emitMTPToken(ctx, request, session, dec, res, final)
		if err != nil {
			return sampler.Result{}, accepted, done, err
		}
		if !done {
			(*generated)++
		}
		if done {
			break
		}
	}

	spec.Commit(commitN)
	*position = before + commitN
	draft.AppendContext(targetHiddenSeq.Slice(mlx.Slice(), mlx.Slice(0, commitN), mlx.Slice()), draftCaches)

	if done || *generated >= request.Options.NumPredict {
		return sampler.Result{}, accepted, true, nil
	}

	nextIndex := accepted
	if nextIndex >= len(selectedIDs) {
		nextIndex = len(selectedIDs) - 1
	}
	return sampler.Result{Token: mlx.FromValues([]int32{int32(selectedIDs[nextIndex])}, 1)}, accepted, false, nil
}

func (r *Runner) acceptDFlashDraftsSerial(ctx context.Context, request Request, session *cacheSession, dec *decoder, target base.DFlashTargetModel, draft base.DFlashDraftModel, targetCaches []cache.Cache, draftCaches []cache.Cache, position *int, current sampler.Result, draftTokens *mlx.Array, final *CompletionResponse, generated *int) (sampler.Result, int, bool, error) {
	targetForward := func(token *mlx.Array) *mlx.Array {
		hidden, targetHidden := target.ForwardDFlash(&batch.Batch{
			InputIDs:     token,
			SeqOffsets:   []int32{int32(*position)},
			SeqQueryLens: []int32{int32(token.Dim(1))},
		}, targetCaches, draft.TargetLayerIDs())
		*position += token.Dim(1)
		draft.AppendContext(targetHidden, draftCaches)
		return r.lastLogits(hidden)
	}

	logits := targetForward(mtpTokenInput(current.Token))
	accepted := 0
	for _, id := range draftTokens.Ints() {
		selected := greedyTokenFromLogits(logits)
		mlx.Eval(selected)
		selectedID := tokenID(selected)
		if selectedID != id {
			return sampler.Result{Token: mlx.FromValues([]int32{int32(selectedID)}, 1)}, accepted, false, nil
		}

		res := sampler.Result{Token: mlx.FromValues([]int32{int32(id)}, 1)}
		done, err := r.emitMTPToken(ctx, request, session, dec, res, final)
		if err != nil {
			return sampler.Result{}, accepted, done, err
		}
		accepted++
		if !done {
			(*generated)++
		}
		if done || *generated >= request.Options.NumPredict {
			return sampler.Result{}, accepted, true, nil
		}

		logits = targetForward(mtpTokenInput(res.Token))
	}

	return sampler.Result{Token: greedyTokenFromLogits(logits)}, accepted, false, nil
}

func (r *Runner) acceptSampleDFlashDrafts(ctx context.Context, request Request, session *cacheSession, dec *decoder, target base.DFlashTargetModel, draft base.DFlashDraftModel, targetCaches []cache.Cache, draftCaches []cache.Cache, position *int, current sampler.Result, candidates *dflashDraftCandidates, final *CompletionResponse, generated *int, stats *dflashStats) (sampler.Result, int, bool, error) {
	specCaches, spec, ok := cache.BeginSpeculation(targetCaches)
	if !ok {
		stats.serial++
		return r.acceptSampleDFlashDraftsSerial(ctx, request, session, dec, target, draft, targetCaches, draftCaches, position, current, candidates, final, generated)
	}
	stats.batched++
	return r.acceptSampleDFlashDraftsBatched(ctx, request, session, dec, target, draft, specCaches, spec, draftCaches, position, current, candidates, final, generated)
}

func (r *Runner) acceptSampleDFlashDraftsBatched(ctx context.Context, request Request, session *cacheSession, dec *decoder, target base.DFlashTargetModel, draft base.DFlashDraftModel, specCaches []cache.Cache, spec *cache.Speculation, draftCaches []cache.Cache, position *int, current sampler.Result, candidates *dflashDraftCandidates, final *CompletionResponse, generated *int) (sampler.Result, int, bool, error) {
	before := *position
	draftCount := candidates.tokens.Dim(1)
	verifyInput := mtpTokenInput(current.Token).Concatenate(1, candidates.tokens)
	hiddenSeq, targetHiddenSeq := target.ForwardDFlash(&batch.Batch{
		InputIDs:     verifyInput,
		SeqOffsets:   []int32{int32(before)},
		SeqQueryLens: []int32{int32(verifyInput.Dim(1))},
	}, specCaches, draft.TargetLayerIDs())

	targetDist := r.Sampler.Distribution(pipelineSlot, r.Model.Unembed(hiddenSeq), candidates.tokens)
	draftDist := candidates.dist
	acceptedMask := r.mtpSampleAcceptedMask(targetDist.SliceRows(0, draftCount), draftDist, candidates.tokens)
	mlx.Eval(candidates.tokens, acceptedMask)

	draftIDs := candidates.tokens.Ints()
	acceptedFlags := acceptedMask.Ints()
	accepted := 0
	for _, ok := range acceptedFlags {
		if ok == 0 {
			break
		}
		accepted++
	}
	if accepted > draftCount {
		spec.Commit(0)
		return sampler.Result{}, 0, false, fmt.Errorf("dflash sample validation accepted %d tokens for %d draft tokens", accepted, draftCount)
	}

	commitIDs := make([]int32, 0, accepted+1)
	done := false
	for i, id := range draftIDs[:accepted] {
		commitIDs = append(commitIDs, int32(id))
		if r.Tokenizer.IsEOS(int32(id)) {
			done = true
			accepted = i + 1
			commitIDs = commitIDs[:accepted]
			break
		}
	}

	commitN := accepted + 1
	spec.Commit(0)

	for _, id := range draftIDs[:accepted] {
		if *generated >= request.Options.NumPredict {
			done = true
			break
		}
		res := sampler.Result{Token: mlx.FromValues([]int32{int32(id)}, 1)}
		var err error
		done, err = r.emitMTPToken(ctx, request, session, dec, res, final)
		if err != nil {
			return sampler.Result{}, accepted, done, err
		}
		if !done {
			(*generated)++
		}
		if done {
			break
		}
	}

	spec.Commit(commitN)
	*position = before + commitN
	draft.AppendContext(targetHiddenSeq.Slice(mlx.Slice(), mlx.Slice(0, commitN), mlx.Slice()), draftCaches)

	if done || *generated >= request.Options.NumPredict {
		r.Sampler.Commit(pipelineSlot, commitIDs)
		return sampler.Result{}, accepted, true, nil
	}

	var nextToken *mlx.Array
	if accepted == draftCount {
		nextToken = r.mtpSampleTokenAt(targetDist, draftCount)
	} else {
		nextToken = r.mtpSampleResidualToken(targetDist, draftDist, accepted)
	}
	mlx.Eval(nextToken)
	nextID := int32(tokenID(nextToken))
	commitIDs = append(commitIDs, nextID)
	r.Sampler.Commit(pipelineSlot, commitIDs)

	return sampler.Result{Token: nextToken}, accepted, false, nil
}

func (r *Runner) acceptSampleDFlashDraftsSerial(ctx context.Context, request Request, session *cacheSession, dec *decoder, target base.DFlashTargetModel, draft base.DFlashDraftModel, targetCaches []cache.Cache, draftCaches []cache.Cache, position *int, current sampler.Result, candidates *dflashDraftCandidates, final *CompletionResponse, generated *int) (sampler.Result, int, bool, error) {
	targetForward := func(token *mlx.Array) *mlx.Array {
		hidden, targetHidden := target.ForwardDFlash(&batch.Batch{
			InputIDs:     token,
			SeqOffsets:   []int32{int32(*position)},
			SeqQueryLens: []int32{int32(token.Dim(1))},
		}, targetCaches, draft.TargetLayerIDs())
		*position += token.Dim(1)
		draft.AppendContext(targetHidden, draftCaches)
		return r.lastLogits(hidden)
	}

	mlx.Eval(candidates.tokens)
	draftIDs := candidates.tokens.Ints()
	logits := targetForward(mtpTokenInput(current.Token))
	accepted := 0

	for i, id := range draftIDs {
		targetDist := r.Sampler.Distribution(pipelineSlot, logits, nil)
		draftDist := candidates.dist.SliceRows(i, i+1)
		draftToken := mlx.FromValues([]int32{int32(id)}, 1)
		acceptedMask := r.mtpSampleAcceptedMask(targetDist, draftDist, draftToken)
		mlx.Eval(acceptedMask)

		if acceptedMask.Ints()[0] == 0 {
			nextToken := mtpTokenVector(r.Sampler.SampleDistribution(pipelineSlot, targetDist.ResidualAgainst(draftDist)))
			mlx.Eval(nextToken)
			r.Sampler.Commit(pipelineSlot, []int32{int32(tokenID(nextToken))})
			return sampler.Result{Token: nextToken}, accepted, false, nil
		}

		accepted++
		r.Sampler.Commit(pipelineSlot, []int32{int32(id)})
		res := sampler.Result{Token: mlx.FromValues([]int32{int32(id)}, 1)}
		done, err := r.emitMTPToken(ctx, request, session, dec, res, final)
		if err != nil {
			return sampler.Result{}, accepted, done, err
		}
		if !done {
			(*generated)++
		}
		if done || *generated >= request.Options.NumPredict {
			return sampler.Result{}, accepted, true, nil
		}

		logits = targetForward(mtpTokenInput(res.Token))
	}

	targetDist := r.Sampler.Distribution(pipelineSlot, logits, nil)
	nextToken := mtpTokenVector(r.Sampler.SampleDistribution(pipelineSlot, targetDist))
	mlx.Eval(nextToken)
	r.Sampler.Commit(pipelineSlot, []int32{int32(tokenID(nextToken))})
	return sampler.Result{Token: nextToken}, accepted, false, nil
}
