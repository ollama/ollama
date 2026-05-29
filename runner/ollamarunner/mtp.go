package ollamarunner

import (
	"log/slog"

	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/model"
	"github.com/ollama/ollama/tokenizer"
)

const mtpDefaultDraftTokens = 4

func isMTPEligible(m model.Model, seq *Sequence) bool {
	mtpModel, ok := m.(model.MTPModel)
	if !ok || !mtpModel.HasDraft() {
		return false
	}
	if !seq.sampler.IsGreedy() {
		return false
	}
	if seq.logprobs {
		return false
	}
	slog.Debug("MTP eligible, attempting draft cycle")
	return true
}

func runMTPCycle(
	s *Server,
	seq *Sequence,
	inputToken int32,
	sampledToken int32,
	logits []float32,
	hiddenFloats []float32,
	hiddenDim int,
	position int32,
	tok tokenizer.Tokenizer,
) (acceptedTokens []int32, nextToken int32, ok bool) {
	mtpModel, valid := s.model.(model.MTPModel)
	if !valid {
		return nil, sampledToken, false
	}

	maxDraft := mtpDefaultDraftTokens
	if seq.numPredict > 0 {
		remaining := seq.numPredict - seq.numPredicted
		if remaining <= 1 {
			return nil, sampledToken, false
		}
		if maxDraft > remaining-1 {
			maxDraft = remaining - 1
		}
	}

	cache := s.model.Config().Cache
	wc, isWrapper := cache.(*kvcache.WrapperCache)
	seqID := seq.cache.Id

	// Draft phase: use the INPUT token (position P) so predictions start
	// at P+1, matching baseChoice. reserve=true avoids cache cell allocation.
	draftCtx := s.model.Backend().NewContext()
	draftTokens, err := mtpModel.MTPDraft(draftCtx, inputToken, hiddenFloats, hiddenDim, position, seqID, cache, maxDraft)
	draftCtx.Close()
	if err != nil {
		slog.Warn("MTP draft failed", "error", err)
		return nil, sampledToken, false
	}

	if len(draftTokens) == 0 {
		return nil, sampledToken, false
	}

	// Begin speculation for verify — cells allocated here are committed or
	// rolled back based on how many tokens are accepted.
	if isWrapper {
		wc.BeginSpeculation(seqID)
	}

	// Verify: draftTokens[0] predicts P+1 (same as baseChoice). If it matches,
	// the verify batch runs the target on draftTokens to check further tokens.
	// draftTokens[0] = sampledToken, so we pass sampledToken for the verify batch.
	verifyCtx := s.model.Backend().NewContext()
	accepted, nextAfter, err := mtpModel.MTPVerify(verifyCtx, logits, sampledToken, draftTokens, seqID, position, cache)
	verifyCtx.Close()
	if err != nil {
		slog.Warn("MTP verification failed", "error", err)
		if isWrapper {
			wc.Rollback()
		}
		return nil, sampledToken, false
	}

	// Rollback the verify's KV entries — the accepted tokens will be
	// re-processed by the normal pipeline through seq.inputs.
	if isWrapper {
		wc.Rollback()
	}

	if accepted > 0 {
		slog.Debug("MTP accepted", "count", accepted, "total_drafted", len(draftTokens))
	}

	// draftTokens[0] = sampledToken (already accounted for by the runner).
	// Return only the ADDITIONAL accepted tokens: draftTokens[1:accepted].
	if accepted <= 1 {
		return nil, nextAfter, true
	}
	return draftTokens[1:accepted], nextAfter, true
}
