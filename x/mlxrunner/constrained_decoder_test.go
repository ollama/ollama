package mlxrunner

import (
	"errors"
	"fmt"
	"math"
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	"github.com/ollama/ollama/x/mlxrunner/model/base"
	sampler "github.com/ollama/ollama/x/mlxrunner/sample"
	"github.com/ollama/ollama/x/tokenizer"
)

type constrainedDecoderTestModel struct {
	matcher                   *constrainedDecoderTestMatcher
	forwards                  int
	secondForwardBeforeAccept bool
}

func (*constrainedDecoderTestModel) LoadWeights(map[string]*mlx.Array) error { return nil }
func (*constrainedDecoderTestModel) NewCaches() []cache.Cache                { return nil }
func (m *constrainedDecoderTestModel) Forward(*batch.Batch, []cache.Cache) (*mlx.Array, *mlx.Array) {
	m.forwards++
	if m.forwards == 2 {
		m.secondForwardBeforeAccept = len(m.matcher.accepted) == 0
	}
	// Token 3 wins unmasked; token 2 wins when only 1 and 2 are allowed.
	return mlx.FromValues([]float32{-4, 1, 5, 20}, 1, 1, 4), nil
}
func (*constrainedDecoderTestModel) Unembed(hidden *mlx.Array) *mlx.Array { return hidden }
func (*constrainedDecoderTestModel) Tokenizer() *tokenizer.Tokenizer      { return nil }
func (*constrainedDecoderTestModel) MaxContextLength() int                { return 32 }

var _ base.Model = (*constrainedDecoderTestModel)(nil)

type constrainedDecoderTestMatcher struct {
	accepted  []int32
	needsMask bool
	acceptAny bool
	fills     int
	failFill  int
}

func (*constrainedDecoderTestMatcher) VocabSize() int { return 4 }

func (m *constrainedDecoderTestMatcher) Fill() ([]int32, bool, error) {
	m.fills++
	if m.fills == m.failFill {
		return nil, false, errors.New("test fill failure")
	}
	if !m.needsMask {
		return nil, false, nil
	}
	if len(m.accepted) > 0 {
		return []int32{int32(uint32(1) << 1)}, true, nil
	}
	return []int32{int32(uint32(1)<<1 | uint32(1)<<2)}, true, nil
}

func (m *constrainedDecoderTestMatcher) Accept(id int32) error {
	if !m.acceptAny && id != 1 && id != 2 {
		return fmt.Errorf("token %d is not allowed", id)
	}
	m.accepted = append(m.accepted, id)
	return nil
}

func (*constrainedDecoderTestMatcher) Close() {}

func TestConstrainedDecoderMasksBeforeSampling(t *testing.T) {
	skipIfNoMLX(t)
	matcher := &constrainedDecoderTestMatcher{needsMask: true}
	model := &constrainedDecoderTestModel{matcher: matcher}
	r := &Runner{
		Model:     model,
		Tokenizer: newTestTokenizer(t, nil),
		Sampler:   sampler.New(32),
	}
	r.Sampler.Add(pipelineSlot, sampler.Options{
		Temperature: 0,
		Logprobs:    true,
		TopLogprobs: 4,
	}, []int32{0})
	defer r.Sampler.Remove(pipelineSlot)

	decoder, err := r.constrainedDecoder(nil, nil, mlx.FromValues([]int32{0}, 1), 0, nil, matcher)
	if err != nil {
		t.Fatal(err)
	}
	defer decoder.close()
	results, err := decoder.next(1)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 1 || int32(results[0].Token.Int()) != 2 {
		t.Fatalf("sampled token = %v, want allowed token 2", results)
	}
	if len(matcher.accepted) != 1 || matcher.accepted[0] != 2 {
		t.Fatalf("accepted tokens = %v, want [2]", matcher.accepted)
	}
	if model.forwards != 2 || !model.secondForwardBeforeAccept {
		t.Fatalf("forwards = %d, second before accept = %v; want pipelined forward before grammar acceptance", model.forwards, model.secondForwardBeforeAccept)
	}
	drained, position := decoder.drain()
	if len(drained) != 1 || int32(drained[0].Token.Int()) != 1 || position != 2 {
		t.Fatalf("drain = %v at %d, want next grammar-valid token 1 at position 2", drained, position)
	}
	if len(matcher.accepted) != 1 {
		t.Fatalf("drain accepted tokens = %v, want only the emitted token", matcher.accepted)
	}

	logprobs := buildLogprob(results[0], true, 4, func(ids []int32) string {
		return fmt.Sprintf("%d", ids[0])
	})
	if len(logprobs) != 1 || len(logprobs[0].TopLogprobs) != 4 {
		t.Fatalf("unconstrained top logprobs = %+v, want all four tokens", logprobs)
	}
	removeMaskedTopLogprobs(logprobs)
	if len(logprobs) != 1 || len(logprobs[0].TopLogprobs) != 2 {
		t.Fatalf("top logprobs = %+v, want the two allowed tokens only", logprobs)
	}
	for _, alternative := range logprobs[0].TopLogprobs {
		if math.IsInf(alternative.Logprob, 0) || math.IsNaN(alternative.Logprob) {
			t.Fatalf("non-finite constrained logprob: %+v", alternative)
		}
	}
}

func TestConstrainedDecoderSkipsAllTrueMask(t *testing.T) {
	skipIfNoMLX(t)
	matcher := &constrainedDecoderTestMatcher{acceptAny: true}
	model := &constrainedDecoderTestModel{matcher: matcher}
	r := &Runner{Model: model, Tokenizer: newTestTokenizer(t, []int32{3}), Sampler: sampler.New(32)}
	r.Sampler.Add(pipelineSlot, sampler.Options{Temperature: 0}, []int32{0})
	defer r.Sampler.Remove(pipelineSlot)

	decoder, err := r.constrainedDecoder(nil, nil, mlx.FromValues([]int32{0}, 1), 0, nil, matcher)
	if err != nil {
		t.Fatal(err)
	}
	defer decoder.close()
	results, err := decoder.next(1)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 1 || int32(results[0].Token.Int()) != 3 {
		t.Fatalf("sample = %v, want EOS token 3", results)
	}
	drained, position := decoder.drain()
	if len(drained) != 0 || position != 2 {
		t.Fatalf("drain = %v at %d, want no lookahead after EOS at position 2", drained, position)
	}
	if len(matcher.accepted) != 1 || matcher.accepted[0] != 3 {
		t.Fatalf("accepted tokens = %v, want [3]", matcher.accepted)
	}
	if matcher.fills != 1 {
		t.Fatalf("mask fills = %d, want no fill after accepting EOS", matcher.fills)
	}
}

func TestConstrainedDecoderPreservesSampleAfterFillError(t *testing.T) {
	skipIfNoMLX(t)
	matcher := &constrainedDecoderTestMatcher{needsMask: true, failFill: 2}
	model := &constrainedDecoderTestModel{matcher: matcher}
	r := &Runner{Model: model, Tokenizer: newTestTokenizer(t, nil), Sampler: sampler.New(32)}
	r.Sampler.Add(pipelineSlot, sampler.Options{Temperature: 0}, []int32{0})
	defer r.Sampler.Remove(pipelineSlot)

	decoder, err := r.constrainedDecoder(nil, nil, mlx.FromValues([]int32{0}, 1), 0, nil, matcher)
	if err != nil {
		t.Fatal(err)
	}
	defer decoder.close()
	if _, err := decoder.next(1); err == nil || err.Error() != "test fill failure" {
		t.Fatalf("next error = %v, want test fill failure", err)
	}

	drained, position := decoder.drain()
	if len(drained) != 1 || int32(drained[0].Token.Int()) != 2 || position != 2 {
		t.Fatalf("drain = %v at %d, want preserved token 2 at position 2", drained, position)
	}
}
