package mlxrunner

import (
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/cache"
)

func TestResetParallelSeqMixedCaches(t *testing.T) {
	r := &Runner{}
	caches := []cache.Cache{
		cache.NewMultiSeq([]cache.Attention{cache.NewKVCache(), cache.NewKVCache()}),
		cache.NewMultiSeq([]cache.Attention{cache.NewRotatingKVCache(16), cache.NewRotatingKVCache(16)}),
		cache.NewMultiSeqRecurrent([]*cache.RecurrentCache{
			cache.NewRecurrentCache(2, 8, 1, 4, 4),
			cache.NewRecurrentCache(2, 8, 1, 4, 4),
		}),
	}
	if err := r.resetParallelSeq(caches, 1); err != nil {
		t.Fatal(err)
	}
	if err := r.resetParallelSeq(caches, 0); err != nil {
		t.Fatal(err)
	}
}

func TestResetParallelSeqRejectsPlain(t *testing.T) {
	r := &Runner{}
	err := r.resetParallelSeq([]cache.Cache{cache.NewKVCache()}, 0)
	if err == nil {
		t.Fatal("expected error for plain KVCache")
	}
}
