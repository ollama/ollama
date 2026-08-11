package mlxrunner

import (
	"fmt"
	"math"

	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
	sampler "github.com/ollama/ollama/x/mlxrunner/sample"
)

type constrainedDecoder struct {
	r             *Runner
	spec          *speculationSession
	caches        []cache.Cache
	layout        []any
	matcher       requestConstraint
	position      int
	tokenMask     []bool
	pendingSample sampler.Result
	eosForwarded  bool
}

func (r *Runner) constrainedDecoder(spec *speculationSession, caches []cache.Cache, seed *mlx.Array, position int, layout []any, matcher requestConstraint) (*constrainedDecoder, error) {
	d := &constrainedDecoder{
		r: r, spec: spec, caches: caches, layout: layout,
		matcher: matcher, position: position, tokenMask: make([]bool, matcher.VocabSize()),
	}
	logits, err := d.startForward(seed.ExpandDims(-1))
	if err != nil {
		return nil, err
	}
	d.pendingSample, err = d.sampleLogits(logits)
	if err != nil {
		return nil, err
	}
	return d, nil
}

func (d *constrainedDecoder) startForward(token *mlx.Array) (*mlx.Array, error) {
	hidden, auxHidden := d.r.Model.Forward(&batch.Batch{
		InputIDs:     token,
		SeqOffsets:   []int32{int32(d.position)},
		SeqQueryLens: []int32{int32(token.Dim(1))},
		Layout:       d.layout,
	}, d.caches)
	d.spec.committed(token, auxHidden, d.position, nil)
	d.position += token.Dim(1)

	logits := d.r.Model.Unembed(hidden)
	if logits.NumDims() != 3 {
		return nil, fmt.Errorf("constraint model logits have %d dimensions, want 3", logits.NumDims())
	}
	logits = logits.Slice(mlx.Slice(), mlx.Slice(logits.Dim(1)-1), mlx.Slice()).Squeeze(1)
	if got, want := logits.Dim(logits.NumDims()-1), d.matcher.VocabSize(); got != want {
		return nil, fmt.Errorf("constraint vocabulary size %d does not match model logits %d", want, got)
	}
	mlx.Pin(logits)
	mlx.Sweep()
	mlx.AsyncEval(logits)
	return logits, nil
}

func (d *constrainedDecoder) sampleLogits(logits *mlx.Array) (sampler.Result, error) {
	source := logits
	packed, needsApply, err := d.matcher.Fill()
	if err != nil {
		mlx.Unpin(source)
		return sampler.Result{}, err
	}
	if needsApply {
		unpackTokenMaskInto(d.tokenMask, packed)
		mask := mlx.FromValues(d.tokenMask, d.matcher.VocabSize())
		negInf := mlx.FromValue(float32(math.Inf(-1))).AsType(logits.DType())
		logits = mlx.Where(mask, logits, negInf)
	}
	next := d.r.Sampler.Sample([]int{pipelineSlot}, logits)
	mlx.Pin(next.Arrays()...)
	mlx.Unpin(source)
	mlx.Sweep()
	mlx.AsyncEval(next.Arrays()...)
	return next, nil
}

func (d *constrainedDecoder) next(int) ([]sampler.Result, error) {
	out := d.pendingSample

	// Start the independent forward before reading the sampled token back to the host.
	logits, err := d.startForward(out.Token.ExpandDims(-1))
	if err != nil {
		return nil, err
	}
	id := int32(out.Token.Int())
	if err := d.matcher.Accept(id); err != nil {
		mlx.Unpin(logits)
		return nil, err
	}
	if d.r.Tokenizer.IsEOS(id) {
		mlx.Unpin(logits)
		mlx.Unpin(out.Arrays()...)
		d.pendingSample = sampler.Result{}
		d.eosForwarded = true
		return []sampler.Result{out}, nil
	}
	next, err := d.sampleLogits(logits)
	if err != nil {
		return nil, err
	}
	d.pendingSample = next
	mlx.Unpin(out.Arrays()...)
	return []sampler.Result{out}, nil
}

func (d *constrainedDecoder) drain() ([]sampler.Result, int) {
	if d.eosForwarded {
		return nil, d.position
	}
	return []sampler.Result{d.pendingSample}, d.position
}

func (d *constrainedDecoder) close() {
	if d.eosForwarded {
		return
	}
	d.spec.settle(d.pendingSample.Token)
	mlx.Unpin(d.pendingSample.Arrays()...)
}

func unpackTokenMaskInto(mask []bool, packed []int32) {
	for id := range mask {
		mask[id] = uint32(packed[id/32])&(uint32(1)<<uint(id%32)) != 0
	}
}
