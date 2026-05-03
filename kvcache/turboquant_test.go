package kvcache

import (
	"testing"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/model/input"
)

type turboQuantTestBaseCache struct {
	canResume        bool
	canResumeCalls   int
	canResumeLastSeq int
	canResumeLastPos int32
}

func (c *turboQuantTestBaseCache) SetLayer(layer int) {}

func (c *turboQuantTestBaseCache) Get(ctx ml.Context) (ml.Tensor, ml.Tensor, ml.Tensor) {
	return nil, nil, nil
}

func (c *turboQuantTestBaseCache) Put(ctx ml.Context, key, value ml.Tensor) {}

func (c *turboQuantTestBaseCache) SetConfig(config ml.CacheConfig) {}

func (c *turboQuantTestBaseCache) Init(backend ml.Backend, dtype ml.DType, maxSequences, capacity, maxBatch int) {
}

func (c *turboQuantTestBaseCache) Close() {}

func (c *turboQuantTestBaseCache) StartForward(ctx ml.Context, batch input.Batch, reserve bool) error {
	return nil
}

func (c *turboQuantTestBaseCache) CopyPrefix(srcSeq, dstSeq int, len int32) {}

func (c *turboQuantTestBaseCache) CanResume(seq int, pos int32) bool {
	c.canResumeCalls++
	c.canResumeLastSeq = seq
	c.canResumeLastPos = pos
	return c.canResume
}

func (c *turboQuantTestBaseCache) Remove(seq int, beginIndex, endIndex int32) error {
	return nil
}

type turboQuantTestCheckpointCache struct {
	turboQuantTestBaseCache
	restorePos      int32
	restoreOK       bool
	prepareCalls    int
	prepareLastSeq  int
	prepareLastPos  int32
}

func (c *turboQuantTestCheckpointCache) PrepareRestore(seq int, targetPos int32) (int32, bool) {
	c.prepareCalls++
	c.prepareLastSeq = seq
	c.prepareLastPos = targetPos
	return c.restorePos, c.restoreOK
}

func TestTurboQuantPrepareRestorePassthrough(t *testing.T) {
	inner := &turboQuantTestCheckpointCache{
		restorePos: 7,
		restoreOK:  true,
	}

	w := NewTurboQuantWrapper(inner, ml.DTypeTQ4)

	gotPos, gotOK := w.PrepareRestore(2, 11)
	if !gotOK || gotPos != 7 {
		t.Fatalf("PrepareRestore() = (%d, %v), want (7, true)", gotPos, gotOK)
	}

	if inner.prepareCalls != 1 {
		t.Fatalf("inner PrepareRestore calls = %d, want 1", inner.prepareCalls)
	}

	if inner.prepareLastSeq != 2 || inner.prepareLastPos != 11 {
		t.Fatalf("inner PrepareRestore args = (%d, %d), want (2, 11)", inner.prepareLastSeq, inner.prepareLastPos)
	}
}

func TestTurboQuantPrepareRestoreFallsBackToCanResume(t *testing.T) {
	inner := &turboQuantTestBaseCache{canResume: true}
	w := NewTurboQuantWrapper(inner, ml.DTypeTQ3)

	gotPos, gotOK := w.PrepareRestore(3, 9)
	if !gotOK || gotPos != 9 {
		t.Fatalf("PrepareRestore() = (%d, %v), want (9, true)", gotPos, gotOK)
	}

	if inner.canResumeCalls != 1 {
		t.Fatalf("inner CanResume calls = %d, want 1", inner.canResumeCalls)
	}

	if inner.canResumeLastSeq != 3 || inner.canResumeLastPos != 9 {
		t.Fatalf("inner CanResume args = (%d, %d), want (3, 9)", inner.canResumeLastSeq, inner.canResumeLastPos)
	}
}

func TestTurboQuantPrepareRestoreFallbackFailure(t *testing.T) {
	inner := &turboQuantTestBaseCache{canResume: false}
	w := NewTurboQuantWrapper(inner, ml.DTypeTQ4)

	gotPos, gotOK := w.PrepareRestore(4, 13)
	if gotOK || gotPos != 0 {
		t.Fatalf("PrepareRestore() = (%d, %v), want (0, false)", gotPos, gotOK)
	}

	if inner.canResumeCalls != 1 {
		t.Fatalf("inner CanResume calls = %d, want 1", inner.canResumeCalls)
	}
}

