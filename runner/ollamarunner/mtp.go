package ollamarunner

import (
	"log/slog"

	"github.com/ollama/ollama/kvcache"
	"github.com/ollama/ollama/ml"
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
	return true
}

func runMTPCycle(
	s *Server,
	seq *Sequence,
	token int32,
	logits []float32,
	hidden ml.Tensor,
	position int32,
	tok tokenizer.Tokenizer,
) (acceptedTokens []int32, nextToken int32, ok bool) {
	mtpModel, valid := s.model.(model.MTPModel)
	if !valid {
		return nil, token, false
	}

	maxDraft := mtpDefaultDraftTokens
	if seq.numPredict > 0 {
		remaining := seq.numPredict - seq.numPredicted
		if remaining <= 1 {
			return nil, token, false
		}
		if maxDraft > remaining-1 {
			maxDraft = remaining - 1
		}
	}

	cache := s.model.Config().Cache
	wc, isWrapper := cache.(*kvcache.WrapperCache)

	if isWrapper {
		wc.BeginSpeculation(seq.cache.Id)
	}

	draftCtx := s.model.Backend().NewContext()
	defer draftCtx.Close()

	draftTokens, err := mtpModel.MTPDraft(draftCtx, token, hidden, position, cache, maxDraft)
	if err != nil {
		slog.Warn("MTP draft failed", "error", err)
		if isWrapper {
			wc.Rollback()
		}
		return nil, token, false
	}

	if len(draftTokens) == 0 {
		if isWrapper {
			wc.Rollback()
		}
		return nil, token, false
	}

	verifyCtx := s.model.Backend().NewContext()
	defer verifyCtx.Close()

	accepted, nextAfter, err := mtpModel.MTPVerify(verifyCtx, logits, draftTokens, seq.cache.Id, position, cache)
	if err != nil {
		slog.Warn("MTP verification failed", "error", err)
		if isWrapper {
			wc.Rollback()
		}
		return nil, token, false
	}

	if isWrapper {
		wc.Commit(accepted)
	}

	if accepted > 0 {
		slog.Debug("MTP accepted", "count", accepted, "total_drafted", len(draftTokens))
	}

	return draftTokens[:accepted], nextAfter, true
}
