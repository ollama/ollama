package mlxrunner

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
	sampler "github.com/ollama/ollama/x/mlxrunner/sample"
)

const (
	mtpDefaultInitialDraftTokens = 4
	mtpDefaultMaxDraftTokens     = 16
)

type mtpDraftSchedule string

const (
	mtpDraftScheduleHeuristic mtpDraftSchedule = "heuristic"
	mtpDraftScheduleConstant  mtpDraftSchedule = "constant"
)

type mtpStats struct {
	iterations            int
	drafted               int
	accepted              int
	mismatches            int
	allAccepted           int
	batched               int
	serial                int
	compared              int
	batchSerialMismatches int
	maxDraft              int
	targetDuration        time.Duration
	draftDuration         time.Duration
	validateDuration      time.Duration
}

type mtpOptions struct {
	initialDraftTokens    int
	maxDraftTokens        int
	draftSchedule         mtpDraftSchedule
	serialValidate        bool
	compareSerialValidate bool
}

func (r *Runner) mtpDefaults(sample bool) base.MTPDefaults {
	defaults := base.MTPDefaults{
		InitialDraftTokens: mtpDefaultInitialDraftTokens,
		MaxDraftTokens:     mtpDefaultMaxDraftTokens,
		Enabled:            true,
	}
	if p, ok := r.Model.(base.MTPDefaultsProvider); ok {
		defaults = p.MTPDraftDefaults(sample)
	}
	if defaults.InitialDraftTokens <= 0 {
		defaults.InitialDraftTokens = mtpDefaultInitialDraftTokens
	}
	if defaults.MaxDraftTokens <= 0 {
		defaults.MaxDraftTokens = mtpDefaultMaxDraftTokens
	}
	return defaults
}

func (r *Runner) loadMTPOptions(sample bool) mtpOptions {
	defaults := r.mtpDefaults(sample)

	opts := mtpOptions{
		initialDraftTokens: defaults.InitialDraftTokens,
		maxDraftTokens:     defaults.MaxDraftTokens,
		draftSchedule:      mtpDraftScheduleConstant,
	}
	if v := positiveEnvInt("OLLAMA_MLX_MTP_MAX_DRAFT_TOKENS"); v > 0 {
		opts.maxDraftTokens = v
	}
	if v := positiveEnvInt("OLLAMA_MLX_MTP_INITIAL_DRAFT_TOKENS"); v > 0 {
		opts.initialDraftTokens = v
	}
	if opts.initialDraftTokens > opts.maxDraftTokens {
		opts.initialDraftTokens = opts.maxDraftTokens
	}
	if b, err := strconv.ParseBool(os.Getenv("OLLAMA_MLX_MTP_SERIAL_VALIDATE")); err == nil {
		opts.serialValidate = b
	}
	if b, err := strconv.ParseBool(os.Getenv("OLLAMA_MLX_MTP_COMPARE_SERIAL_VALIDATE")); err == nil {
		opts.compareSerialValidate = b
	}
	switch schedule := strings.ToLower(strings.TrimSpace(os.Getenv("OLLAMA_MLX_MTP_DRAFT_SCHEDULE"))); schedule {
	case "", string(mtpDraftScheduleConstant):
		opts.draftSchedule = mtpDraftScheduleConstant
	case string(mtpDraftScheduleHeuristic):
		opts.draftSchedule = mtpDraftScheduleHeuristic
	default:
		slog.Warn("invalid MTP env setting", "key", "OLLAMA_MLX_MTP_DRAFT_SCHEDULE", "value", schedule)
	}
	return opts
}

func positiveEnvInt(key string) int {
	raw := os.Getenv(key)
	if raw == "" {
		return 0
	}
	v, err := strconv.Atoi(raw)
	if err != nil || v <= 0 {
		slog.Warn("invalid MTP env setting", "key", key, "value", raw)
		return 0
	}
	return v
}

func (r *Runner) useGreedyMTP(opts sampler.Options) bool {
	if r.Draft == nil {
		return false
	}
	if _, ok := r.Draft.(base.MTPDraftModel); !ok {
		return false
	}
	if _, ok := r.Model.(base.MTPEmbeddingModel); !ok {
		return false
	}
	if !r.mtpDefaults(false).Enabled {
		return false
	}
	if opts.Logprobs || opts.TopLogprobs > 0 {
		return false
	}
	if opts.Temperature != 0 {
		return false
	}
	repeatPenaltyNeutral := opts.RepeatPenalty <= 0 || opts.RepeatPenalty == 1
	topPNeutral := opts.TopP <= 0 || opts.TopP >= 1
	topKNeutral := opts.TopK <= 0
	return repeatPenaltyNeutral && opts.PresencePenalty == 0 && opts.FrequencyPenalty == 0 && topPNeutral && topKNeutral && opts.MinP == 0
}

func (r *Runner) useSampleMTP(opts sampler.Options) bool {
	if serial, err := strconv.ParseBool(os.Getenv("OLLAMA_MLX_MTP_SERIAL_VALIDATE")); err == nil && serial {
		return false
	}
	if compare, err := strconv.ParseBool(os.Getenv("OLLAMA_MLX_MTP_COMPARE_SERIAL_VALIDATE")); err == nil && compare {
		return false
	}
	if r.Draft == nil {
		return false
	}
	if _, ok := r.Draft.(base.MTPDraftModel); !ok {
		return false
	}
	if _, ok := r.Model.(base.MTPEmbeddingModel); !ok {
		return false
	}
	if !r.mtpDefaults(true).Enabled {
		return false
	}
	if opts.Logprobs || opts.TopLogprobs > 0 {
		return false
	}
	return opts.Temperature != 0
}

func (r *Runner) runGreedyMTPDecode(ctx context.Context, request Request, session *cacheSession, caches []cache.Cache, seed []int32, position *int, started time.Time) error {
	targetEmbeddings := r.Model.(base.MTPEmbeddingModel)
	draft := r.Draft.(base.MTPDraftModel)
	mtpOpts := r.loadMTPOptions(false)
	stats := mtpStats{maxDraft: mtpOpts.initialDraftTokens}
	draftLimit := mtpOpts.initialDraftTokens
	slog.Info("MTP greedy decode enabled", "initial_draft_tokens", mtpOpts.initialDraftTokens, "max_draft_tokens", mtpOpts.maxDraftTokens, "draft_schedule", mtpOpts.draftSchedule, "serial_validate", mtpOpts.serialValidate, "compare_serial_validate", mtpOpts.compareSerialValidate)

	targetForward := func(token *mlx.Array) *mlx.Array {
		fwd := r.Model.Forward(&batch.Batch{
			InputIDs:     token,
			SeqOffsets:   []int32{int32(*position)},
			SeqQueryLens: []int32{int32(token.Dim(1))},
		}, caches)
		*position += token.Dim(1)
		return fwd
	}

	hidden := targetForward(mlx.FromValues(seed, 1, len(seed)))
	current := sampler.Result{Token: greedyTokenFromLogits(r.lastLogits(hidden))}
	mlx.Pin(current.Arrays()...)
	mlx.Sweep()
	mlx.AsyncEval(current.Arrays()...)
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

		t0 := time.Now()
		hidden = targetForward(current.Token.ExpandDims(-1))
		baseLogits := r.lastLogits(hidden)
		stats.targetDuration += time.Since(t0)

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

		stats.iterations++
		maxDraft := min(draftLimit, request.Options.NumPredict-generated)
		t0 = time.Now()
		draftTokens := r.generateMTPDrafts(draft, targetEmbeddings, current.Token, hidden, caches, int32(*position-1), maxDraft)
		draftCount := 0
		if draftTokens != nil {
			draftCount = draftTokens.Dim(1)
			mlx.Pin(baseLogits, draftTokens)
			mlx.Eval(draftTokens)
			mlx.Sweep()
		}
		stats.draftDuration += time.Since(t0)
		stats.drafted += draftCount
		var next sampler.Result
		if draftCount == 0 {
			next = sampler.Result{Token: greedyTokenFromLogits(baseLogits)}
		} else {
			var accepted int
			t0 = time.Now()
			next, accepted, done, err = r.acceptMTPDrafts(ctx, request, session, &dec, caches, position, baseLogits, draftTokens, &final, &generated, &stats, mtpOpts)
			stats.validateDuration += time.Since(t0)
			mlx.Unpin(baseLogits, draftTokens)
			if err != nil {
				return err
			}
			stats.accepted += accepted
			switch {
			case mtpOpts.draftSchedule == mtpDraftScheduleConstant:
			case accepted == draftCount:
				stats.allAccepted++
				draftLimit = min(mtpOpts.maxDraftTokens, draftLimit+2)
			default:
				stats.mismatches++
				draftLimit = max(1, draftLimit-1)
			}
			if mtpOpts.draftSchedule == mtpDraftScheduleConstant {
				if accepted == draftCount {
					stats.allAccepted++
				} else {
					stats.mismatches++
				}
			}
			stats.maxDraft = max(stats.maxDraft, draftLimit)
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
	slog.Info("MTP decode stats", "generated", generated, "drafted", stats.drafted, "accepted", stats.accepted, "acceptance", acceptance, "iterations", stats.iterations, "avg_draft", avgDraft, "avg_accepted", avgAccepted, "batched", stats.batched, "serial", stats.serial, "compared", stats.compared, "batch_serial_mismatches", stats.batchSerialMismatches, "mismatches", stats.mismatches, "all_accepted", stats.allAccepted, "max_draft", stats.maxDraft, "draft_schedule", mtpOpts.draftSchedule, "target_duration", stats.targetDuration, "draft_duration", stats.draftDuration, "validate_duration", stats.validateDuration)
	select {
	case <-ctx.Done():
		return ctx.Err()
	case request.Responses <- final:
		return nil
	}
}

func (r *Runner) runSampleMTPDecode(ctx context.Context, request Request, session *cacheSession, caches []cache.Cache, seed []int32, position *int, started time.Time) error {
	targetEmbeddings := r.Model.(base.MTPEmbeddingModel)
	draft := r.Draft.(base.MTPDraftModel)
	mtpOpts := r.loadMTPOptions(true)
	stats := mtpStats{maxDraft: mtpOpts.initialDraftTokens}
	draftLimit := mtpOpts.initialDraftTokens
	slog.Info("MTP sample decode enabled", "initial_draft_tokens", mtpOpts.initialDraftTokens, "max_draft_tokens", mtpOpts.maxDraftTokens, "draft_schedule", mtpOpts.draftSchedule, "serial_validate", mtpOpts.serialValidate)

	targetForward := func(token *mlx.Array) *mlx.Array {
		fwd := r.Model.Forward(&batch.Batch{
			InputIDs:     token,
			SeqOffsets:   []int32{int32(*position)},
			SeqQueryLens: []int32{int32(token.Dim(1))},
		}, caches)
		*position += token.Dim(1)
		return fwd
	}

	hidden := targetForward(mlx.FromValues(seed, 1, len(seed)))
	current := r.Sampler.Sample([]int{pipelineSlot}, r.lastLogits(hidden))
	mlx.Pin(current.Arrays()...)
	mlx.Sweep()
	mlx.AsyncEval(current.Arrays()...)
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

		t0 := time.Now()
		hidden = targetForward(mtpTokenInput(current.Token))
		baseLogits := r.lastLogits(hidden)
		stats.targetDuration += time.Since(t0)

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

		stats.iterations++
		maxDraft := min(draftLimit, request.Options.NumPredict-generated)
		t0 = time.Now()
		candidates := r.generateMTPDraftCandidates(draft, targetEmbeddings, current.Token, hidden, caches, int32(*position-1), maxDraft)
		draftCount := 0
		var candidateArrays []*mlx.Array
		if candidates != nil {
			draftCount = candidates.tokens.Dim(1)
			candidateArrays = append([]*mlx.Array{baseLogits}, candidates.Arrays()...)
			mlx.Pin(candidateArrays...)
			mlx.Sweep()
		}
		stats.draftDuration += time.Since(t0)
		stats.drafted += draftCount

		var next sampler.Result
		if draftCount == 0 {
			next = r.Sampler.Sample([]int{pipelineSlot}, baseLogits)
		} else {
			var accepted int
			t0 = time.Now()
			next, accepted, done, err = r.acceptSampleMTPDrafts(ctx, request, session, &dec, caches, position, baseLogits, candidates, &final, &generated, &stats)
			stats.validateDuration += time.Since(t0)
			mlx.Unpin(candidateArrays...)
			if err != nil {
				return err
			}
			stats.accepted += accepted
			switch {
			case mtpOpts.draftSchedule == mtpDraftScheduleConstant:
			case accepted == draftCount:
				stats.allAccepted++
				draftLimit = min(mtpOpts.maxDraftTokens, draftLimit+2)
			default:
				stats.mismatches++
				draftLimit = max(1, draftLimit-1)
			}
			if mtpOpts.draftSchedule == mtpDraftScheduleConstant {
				if accepted == draftCount {
					stats.allAccepted++
				} else {
					stats.mismatches++
				}
			}
			stats.maxDraft = max(stats.maxDraft, draftLimit)
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
	slog.Info("MTP decode stats", "mode", "sample", "generated", generated, "drafted", stats.drafted, "accepted", stats.accepted, "acceptance", acceptance, "iterations", stats.iterations, "avg_draft", avgDraft, "avg_accepted", avgAccepted, "batched", stats.batched, "serial", stats.serial, "mismatches", stats.mismatches, "all_accepted", stats.allAccepted, "max_draft", stats.maxDraft, "draft_schedule", mtpOpts.draftSchedule, "target_duration", stats.targetDuration, "draft_duration", stats.draftDuration, "validate_duration", stats.validateDuration)
	select {
	case <-ctx.Done():
		return ctx.Err()
	case request.Responses <- final:
		return nil
	}
}

type mtpDraftCandidates struct {
	tokens *mlx.Array
	// dist is the proposal distribution used to sample each drafted token.
	dist sampler.Distribution
}

func (c *mtpDraftCandidates) Arrays() []*mlx.Array {
	if c == nil {
		return nil
	}
	return append([]*mlx.Array{c.tokens}, c.dist.Arrays()...)
}

func (r *Runner) generateMTPDrafts(draft base.MTPDraftModel, target base.MTPEmbeddingModel, token *mlx.Array, hidden *mlx.Array, caches []cache.Cache, position int32, maxDraft int) *mlx.Array {
	if maxDraft <= 0 {
		return nil
	}

	lastToken := token.ExpandDims(-1)
	lastHidden := hidden
	draftTokens := make([]*mlx.Array, 0, maxDraft)

	// Gemma4 assistant MTP is trained as "single-position" drafting:
	// keep the RoPE/cache position anchored at the last target-seen token
	// while the proposed token and projected hidden state advance.
	for range maxDraft {
		tokenEmbedding := target.TokenEmbeddings(lastToken)
		inputs := tokenEmbedding.Concatenate(-1, lastHidden)
		logits, projected := draft.Draft(inputs, position, caches)
		stepLogits := r.lastLogitsFromLogits(logits)
		nextToken := greedyTokenFromLogits(stepLogits)

		lastToken = nextToken.ExpandDims(-1)
		lastHidden = projected
		draftTokens = append(draftTokens, lastToken)
	}
	if len(draftTokens) == 0 {
		return nil
	}
	return mlx.Concatenate(draftTokens, 1)
}

func (r *Runner) generateMTPDraftCandidates(draft base.MTPDraftModel, target base.MTPEmbeddingModel, token *mlx.Array, hidden *mlx.Array, caches []cache.Cache, position int32, maxDraft int) *mtpDraftCandidates {
	if maxDraft <= 0 {
		return nil
	}

	lastToken := mtpTokenInput(token)
	lastHidden := hidden
	draftTokens := make([]*mlx.Array, 0, maxDraft)
	draftDists := make([]sampler.Distribution, 0, maxDraft)
	var prefix *mlx.Array

	// Gemma4 assistant MTP is trained as "single-position" drafting:
	// keep the RoPE/cache position anchored at the last target-seen token
	// while the proposed token and projected hidden state advance.
	for range maxDraft {
		tokenEmbedding := target.TokenEmbeddings(lastToken)
		inputs := tokenEmbedding.Concatenate(-1, lastHidden)
		logits, projected := draft.Draft(inputs, position, caches)
		stepLogits := r.lastLogitsFromLogits(logits)
		dist := r.Sampler.Distribution(pipelineSlot, stepLogits, prefix)
		nextToken := r.Sampler.SampleDistribution(pipelineSlot, dist)

		lastToken = mtpTokenInput(nextToken)
		lastHidden = projected
		draftTokens = append(draftTokens, lastToken)
		draftDists = append(draftDists, dist)
		if prefix == nil {
			prefix = lastToken
		} else {
			prefix = prefix.Concatenate(1, lastToken)
		}
	}
	if len(draftTokens) == 0 {
		return nil
	}
	return &mtpDraftCandidates{
		tokens: mlx.Concatenate(draftTokens, 1),
		dist:   sampler.ConcatenateDistributions(draftDists),
	}
}

func (r *Runner) acceptMTPDrafts(ctx context.Context, request Request, session *cacheSession, dec *decoder, caches []cache.Cache, position *int, baseLogits *mlx.Array, draftTokens *mlx.Array, final *CompletionResponse, generated *int, stats *mtpStats, opts mtpOptions) (sampler.Result, int, bool, error) {
	if opts.serialValidate {
		stats.serial++
		return r.acceptMTPDraftsSerial(ctx, request, session, dec, caches, position, baseLogits, draftTokens, final, generated)
	}

	specCaches, spec, ok := cache.BeginSpeculation(caches)
	if ok {
		stats.batched++
		return r.acceptMTPDraftsBatched(ctx, request, session, dec, caches, specCaches, spec, position, baseLogits, draftTokens, final, generated, stats, opts)
	}

	stats.serial++
	return r.acceptMTPDraftsSerial(ctx, request, session, dec, caches, position, baseLogits, draftTokens, final, generated)
}

func (r *Runner) acceptMTPDraftsBatched(ctx context.Context, request Request, session *cacheSession, dec *decoder, liveCaches []cache.Cache, caches []cache.Cache, spec *cache.Speculation, position *int, baseLogits *mlx.Array, draftTokens *mlx.Array, final *CompletionResponse, generated *int, stats *mtpStats, opts mtpOptions) (sampler.Result, int, bool, error) {
	before := *position
	draftCount := draftTokens.Dim(1)
	hiddenSeq := r.Model.Forward(&batch.Batch{
		InputIDs:     draftTokens,
		SeqOffsets:   []int32{int32(before)},
		SeqQueryLens: []int32{int32(draftCount)},
	}, caches)

	accepted := 0
	var next sampler.Result
	done := false

	selectedTokens := r.mtpValidationTokens(baseLogits, hiddenSeq)
	mlx.Eval(draftTokens, selectedTokens)
	draftIDs := draftTokens.Ints()
	selectedIDs := selectedTokens.Ints()
	if len(selectedIDs) < draftCount+1 {
		return sampler.Result{}, accepted, false, fmt.Errorf("mtp validation produced %d tokens for %d draft tokens", len(selectedIDs), draftCount)
	}

	for i, id := range draftIDs {
		if selectedIDs[i] != id {
			next = sampler.Result{Token: mtpTokenAt(selectedTokens, i)}
			break
		}
		accepted++
		if r.Tokenizer.IsEOS(int32(id)) {
			done = true
			break
		}
	}

	if opts.compareSerialValidate {
		spec.Commit(0)
		r.compareMTPBatchedWithSerial(ctx, liveCaches, before, baseLogits, hiddenSeq, draftIDs, selectedIDs, accepted, draftCount, stats)
	}
	spec.Commit(accepted)
	*position = before + accepted

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

	if done || *generated >= request.Options.NumPredict {
		return sampler.Result{}, accepted, true, nil
	}
	if next.Token == nil {
		next = sampler.Result{Token: mtpTokenAt(selectedTokens, draftCount)}
	}
	return next, accepted, false, nil
}

func (r *Runner) acceptSampleMTPDrafts(ctx context.Context, request Request, session *cacheSession, dec *decoder, caches []cache.Cache, position *int, baseLogits *mlx.Array, candidates *mtpDraftCandidates, final *CompletionResponse, generated *int, stats *mtpStats) (sampler.Result, int, bool, error) {
	specCaches, spec, ok := cache.BeginSpeculation(caches)
	if !ok {
		stats.serial++
		return r.Sampler.Sample([]int{pipelineSlot}, baseLogits), 0, false, nil
	}
	stats.batched++

	before := *position
	draftCount := candidates.tokens.Dim(1)
	hiddenSeq := r.Model.Forward(&batch.Batch{
		InputIDs:     candidates.tokens,
		SeqOffsets:   []int32{int32(before)},
		SeqQueryLens: []int32{int32(draftCount)},
	}, specCaches)

	targetDist := r.Sampler.Distribution(pipelineSlot, r.mtpValidationLogits(baseLogits, hiddenSeq), candidates.tokens)
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
		return sampler.Result{}, 0, false, fmt.Errorf("mtp sample validation accepted %d tokens for %d draft tokens", accepted, draftCount)
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

	spec.Commit(accepted)
	*position = before + accepted

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

func (r *Runner) mtpSampleAcceptedMask(targetDist, draftDist sampler.Distribution, draftTokens *mlx.Array) *mlx.Array {
	p := targetDist.Prob(draftTokens)
	q := draftDist.Prob(draftTokens)
	acceptP := mlx.Minimum(p.Divide(q), mlx.FromValue(float32(1)))
	return r.Sampler.Bernoulli(pipelineSlot, acceptP).AsType(mlx.DTypeInt32)
}

func (r *Runner) mtpSampleTokenAt(dist sampler.Distribution, index int) *mlx.Array {
	return mtpTokenVector(r.Sampler.SampleDistribution(pipelineSlot, dist.SliceRows(index, index+1)))
}

func (r *Runner) mtpSampleResidualToken(targetDist, draftDist sampler.Distribution, index int) *mlx.Array {
	residual := targetDist.SliceRows(index, index+1).ResidualAgainst(draftDist.SliceRows(index, index+1))
	return mtpTokenVector(r.Sampler.SampleDistribution(pipelineSlot, residual))
}

func mtpTokenInput(token *mlx.Array) *mlx.Array {
	switch token.NumDims() {
	case 0:
		return token.Reshape(1, 1)
	case 1:
		return token.ExpandDims(-1)
	case 2:
		return token
	default:
		panic(fmt.Sprintf("mtp token must be rank 0, 1, or 2, got rank %d", token.NumDims()))
	}
}

func mtpTokenVector(token *mlx.Array) *mlx.Array {
	switch token.NumDims() {
	case 0:
		return token.Reshape(1)
	case 1:
		return token
	default:
		panic(fmt.Sprintf("mtp sampled token must be rank 0 or 1, got rank %d", token.NumDims()))
	}
}

func (r *Runner) compareMTPBatchedWithSerial(ctx context.Context, caches []cache.Cache, before int, baseLogits, hiddenSeq *mlx.Array, draftIDs, selectedIDs []int, accepted, draftCount int, stats *mtpStats) {
	serialCaches, ok := cache.BeginIsolatedSpeculation(caches)
	if !ok {
		return
	}

	compareCount := accepted + 1
	if accepted == draftCount {
		// Include the target bonus token when every draft was accepted.
		compareCount = draftCount + 1
	}

	serialLogits := baseLogits
	for i := range compareCount {
		if err := ctx.Err(); err != nil {
			return
		}
		if i >= len(selectedIDs) {
			return
		}

		batchedLogits := baseLogits
		if i > 0 {
			batchedLogits = r.targetLogitsAt(hiddenSeq, i-1)
		}

		batchedToken := greedyTokenFromLogits(batchedLogits)
		serialToken := greedyTokenFromLogits(serialLogits)
		mlx.Eval(batchedToken, serialToken)

		batchedID := tokenID(batchedToken)
		vectorizedID := selectedIDs[i]
		serialID := tokenID(serialToken)
		stats.compared++
		if vectorizedID != serialID {
			firstMismatch := stats.batchSerialMismatches == 0
			stats.batchSerialMismatches++
			if !firstMismatch {
				return
			}

			draftID := -1
			if i < draftCount {
				draftID = draftIDs[i]
			}
			batchedTop := top2FromLogits(batchedLogits)
			serialTop := top2FromLogits(serialLogits)
			slog.Warn("MTP batched validation differs from serial validation",
				"position", before+i,
				"draft", draftID,
				"batched", vectorizedID,
				"batched_slice", batchedID,
				"serial", serialID,
				"batched_slice_top1", batchedTop.firstToken,
				"batched_slice_top2", batchedTop.secondToken,
				"batched_slice_margin", batchedTop.margin,
				"serial_top1", serialTop.firstToken,
				"serial_top2", serialTop.secondToken,
				"serial_margin", serialTop.margin,
			)
			return
		}

		if i >= draftCount || i >= accepted {
			return
		}

		hidden := r.Model.Forward(&batch.Batch{
			InputIDs:     mlx.FromValues([]int32{int32(draftIDs[i])}, 1, 1),
			SeqOffsets:   []int32{int32(before + i)},
			SeqQueryLens: []int32{1},
		}, serialCaches)
		serialLogits = r.lastLogits(hidden)
	}
}

type mtpTop2 struct {
	firstToken  int
	secondToken int
	margin      float64
}

func top2FromLogits(logits *mlx.Array) mtpTop2 {
	indices := logits.Negative().ArgsortAxis(-1).Slice(mlx.Slice(), mlx.Slice(0, 2))
	indices32 := indices.AsType(mlx.DTypeInt32)
	values := logits.TakeAlongAxis(indices, -1).AsType(mlx.DTypeFloat32)
	mlx.Eval(indices32, values)

	tokenIDs := indices32.Ints()
	logitValues := values.Floats()
	if len(tokenIDs) < 2 || len(logitValues) < 2 {
		return mtpTop2{}
	}
	return mtpTop2{
		firstToken:  tokenIDs[0],
		secondToken: tokenIDs[1],
		margin:      float64(logitValues[0] - logitValues[1]),
	}
}

func (r *Runner) acceptMTPDraftsSerial(ctx context.Context, request Request, session *cacheSession, dec *decoder, caches []cache.Cache, position *int, baseLogits *mlx.Array, draftTokens *mlx.Array, final *CompletionResponse, generated *int) (sampler.Result, int, bool, error) {
	logits := baseLogits
	accepted := 0
	draftIDs := draftTokens.Ints()

	for _, id := range draftIDs {
		selected := greedyTokenFromLogits(logits)
		mlx.Eval(selected)
		selectedID := tokenID(selected)
		if selectedID != id {
			return sampler.Result{Token: mlx.FromValues([]int32{int32(selectedID)}, 1)}, accepted, false, nil
		}

		hidden := r.Model.Forward(&batch.Batch{
			InputIDs:     mlx.FromValues([]int32{int32(id)}, 1, 1),
			SeqOffsets:   []int32{int32(*position)},
			SeqQueryLens: []int32{1},
		}, caches)
		(*position)++
		accepted++

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

		logits = r.lastLogits(hidden)
	}

	return sampler.Result{Token: greedyTokenFromLogits(logits)}, accepted, false, nil
}

func (r *Runner) emitMTPToken(ctx context.Context, request Request, session *cacheSession, dec *decoder, res sampler.Result, final *CompletionResponse) (bool, error) {
	output := int32(tokenID(res.Token))
	session.outputs = append(session.outputs, output)

	if r.Tokenizer.IsEOS(output) {
		final.DoneReason = 0
		return true, nil
	}

	if resp, ok := dec.decode(res); ok {
		select {
		case <-ctx.Done():
			return false, ctx.Err()
		case request.Responses <- resp:
		}
	}
	return false, nil
}

func (r *Runner) lastLogits(hidden *mlx.Array) *mlx.Array {
	logits := r.Model.Unembed(hidden)
	return r.lastLogitsFromLogits(logits)
}

func (r *Runner) targetLogitsAt(hiddenSeq *mlx.Array, index int) *mlx.Array {
	hidden := hiddenSeq.Slice(mlx.Slice(), mlx.Slice(index), mlx.Slice())
	return r.lastLogits(hidden)
}

func (r *Runner) mtpValidationTokens(baseLogits, hiddenSeq *mlx.Array) *mlx.Array {
	return greedyTokenFromLogits(r.mtpValidationLogits(baseLogits, hiddenSeq))
}

func (r *Runner) mtpValidationLogits(baseLogits, hiddenSeq *mlx.Array) *mlx.Array {
	seqLogits := r.Model.Unembed(hiddenSeq)
	return baseLogits.ExpandDims(1).Concatenate(1, seqLogits)
}

func mtpTokenAt(tokens *mlx.Array, index int) *mlx.Array {
	return tokens.Slice(mlx.Slice(), mlx.Slice(index)).Squeeze(0)
}

func (r *Runner) lastLogitsFromLogits(logits *mlx.Array) *mlx.Array {
	return logits.Slice(mlx.Slice(), mlx.Slice(logits.Dim(1)-1), mlx.Slice()).Squeeze(1)
}

func greedyTokenFromLogits(logits *mlx.Array) *mlx.Array {
	return logits.Argmax(-1, false).AsType(mlx.DTypeInt32)
}

func tokenID(token *mlx.Array) int {
	if token == nil {
		return -1
	}
	if token.DType() == mlx.DTypeInt32 {
		ids := token.Ints()
		if len(ids) > 0 {
			return ids[0]
		}
	}
	return token.Int()
}
