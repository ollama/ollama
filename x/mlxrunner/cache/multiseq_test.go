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

func TestWrapParallelCachesRotatingAndRecurrent(t *testing.T) {
	caches := []Cache{
		NewKVCache(),
		NewRotatingKVCache(128),
		NewRecurrentCache(3, 64, 4, 16, 16),
	}
	wrapped, ok := WrapParallelCaches(caches, 2)
	if !ok {
		t.Fatal("expected WrapParallelCaches to succeed for KV/rotating/recurrent mix")
	}
	if _, ok := wrapped[0].(*MultiSeq); !ok {
		t.Fatalf("layer 0: got %T, want *MultiSeq", wrapped[0])
	}
	if _, ok := wrapped[1].(*MultiSeq); !ok {
		t.Fatalf("layer 1: got %T, want *MultiSeq", wrapped[1])
	}
	mr, ok := wrapped[2].(*MultiSeqRecurrent)
	if !ok {
		t.Fatalf("layer 2: got %T, want *MultiSeqRecurrent", wrapped[2])
	}
	if mr.NumSeq() != 2 {
		t.Fatalf("recurrent NumSeq=%d want 2", mr.NumSeq())
	}
}

func TestWrapParallelCachesRejectsUnknown(t *testing.T) {
	caches := []Cache{nil}
	if _, ok := WrapParallelCaches(caches, 2); ok {
		t.Fatal("expected WrapParallelCaches to reject nil layer")
	}
}

func TestMultiSeqResetSeq(t *testing.T) {
	ms := NewMultiSeq([]Attention{NewKVCache(), NewRotatingKVCache(32)})
	if err := ms.ResetSeq(0); err != nil {
		t.Fatal(err)
	}
	if err := ms.ResetSeq(1); err != nil {
		t.Fatal(err)
	}
	if _, ok := ms.seqs[1].(*RotatingKVCache); !ok {
		t.Fatalf("after reset got %T, want *RotatingKVCache", ms.seqs[1])
	}
	if err := ms.ResetSeq(2); err == nil {
		t.Fatal("expected out of range error")
	}
}

func TestMultiSeqRecurrentResetSeq(t *testing.T) {
	mr := NewMultiSeqRecurrent([]*RecurrentCache{
		NewRecurrentCache(3, 64, 4, 16, 16),
		NewRecurrentCache(3, 64, 4, 16, 16),
	})
	if err := mr.ResetSeq(1); err != nil {
		t.Fatal(err)
	}
	if err := mr.ResetSeq(2); err == nil {
		t.Fatal("expected out of range error")
	}
}
