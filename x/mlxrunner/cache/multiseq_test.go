package cache

import "testing"

func TestWrapParallelCaches(t *testing.T) {
	caches := []Cache{NewKVCache(), NewKVCache()}
	wrapped, ok := WrapParallelCaches(caches, 4)
	if !ok {
		t.Fatal("expected WrapParallelCaches to succeed for KVCache layers")
	}
	if len(wrapped) != 2 {
		t.Fatalf("len=%d want 2", len(wrapped))
	}
	for i, c := range wrapped {
		ms, ok := c.(*MultiSeq)
		if !ok {
			t.Fatalf("layer %d: got %T, want *MultiSeq", i, c)
		}
		if ms.NumSeq() != 4 {
			t.Fatalf("layer %d: NumSeq=%d want 4", i, ms.NumSeq())
		}
	}
}

func TestWrapParallelCachesRejectsRotating(t *testing.T) {
	caches := []Cache{NewRotatingKVCache(128)}
	if _, ok := WrapParallelCaches(caches, 2); ok {
		t.Fatal("expected WrapParallelCaches to reject rotating cache")
	}
}

func TestMultiSeqResetSeq(t *testing.T) {
	ms := NewMultiSeq([]Attention{NewKVCache(), NewKVCache()})
	if err := ms.ResetSeq(1); err != nil {
		t.Fatal(err)
	}
	if err := ms.ResetSeq(2); err == nil {
		t.Fatal("expected out of range error")
	}
}
