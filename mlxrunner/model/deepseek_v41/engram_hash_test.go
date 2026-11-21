package deepseek_v41

import (
	"reflect"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
)

func TestEngramHashCarriesHistoryAndStopsAtImage(t *testing.T) {
	layout, err := newEngramLayout(&Config{
		EngramLayerIDs: []int{1}, EngramNumEmbeddings: []int64{60},
		EngramMaxNGramSize: 3, EngramVocabSize: 11, EngramNumHeads: 2,
	})
	if err != nil {
		t.Fatal(err)
	}
	newHasher := func() *engramHasher {
		h, err := newEngramHasher(layout, []int32{0, 1, 2, 3}, [][]int64{{3, 5, 7}}, 0)
		if err != nil {
			t.Fatal(err)
		}
		return h
	}
	whole, err := newHasher().hashChunk([]int32{1, 2, 3}, nil)
	if err != nil {
		t.Fatal(err)
	}
	h := newHasher()
	first, err := h.hashChunk([]int32{1, 2}, nil)
	if err != nil {
		t.Fatal(err)
	}
	last, err := h.hashChunk([]int32{3}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got := append(first, last...); !reflect.DeepEqual(got, whole) {
		t.Fatalf("chunked hashes = %v, whole = %v", got, whole)
	}
	if got, want := whole[2][0], []int64{3, 14, 28, 45}; !reflect.DeepEqual(got, want) {
		t.Fatalf("third token hash rows = %v, want %v", got, want)
	}
	h = newHasher()
	masked, err := h.hashChunk([]int32{1, 2, 3}, []bool{true, false, true})
	if err != nil {
		t.Fatal(err)
	}
	if reflect.DeepEqual(masked[2], whole[2]) {
		t.Fatal("image boundary did not stop n-gram lookback")
	}
}

func TestEngramHashKeepsBatchedHistoriesSeparate(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		config := &Config{
			EngramLayerIDs: []int{1}, EngramNumEmbeddings: []int64{60},
			EngramMaxNGramSize: 3, EngramVocabSize: 11, EngramNumHeads: 2,
		}
		layout, err := newEngramLayout(config)
		if err != nil {
			t.Fatal(err)
		}
		m := &Model{Config: config, EngramLayout: layout, TokenMap: []int32{0, 1, 2, 3}, HashMultipliers: [][]int64{{3, 5, 7}}}
		seed := func(ids []int32) *cache.KVCache {
			state := cache.NewKVCache()
			b := &batch.Batch{InputIDs: mlx.FromValues(ids, 1, len(ids)), SeqOffsets: []int32{0}, SeqQueryLens: []int32{int32(len(ids))}}
			m.hashCurrentTokens(b, state)
			return state
		}
		one, two := seed([]int32{1, 2}), seed([]int32{3})
		layers, err := cache.Batch([][]cache.Cache{{one}, {two}})
		if err != nil {
			t.Fatal(err)
		}
		b := &batch.Batch{InputIDs: mlx.FromValues([]int32{3, 2}, 2, 1), SeqOffsets: []int32{2, 1}, SeqQueryLens: []int32{1, 1}}
		got := m.hashCurrentTokens(b, layers[0])[1]
		wantOne := m.hashCurrentTokens(b.Row(0), seed([]int32{1, 2}))[1]
		wantTwo := m.hashCurrentTokens(b.Row(1), seed([]int32{3}))[1]
		want := append(wantOne, wantTwo...)
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("batched engram rows = %v, separate rows = %v", got, want)
		}
		if one.Offset() != 3 || two.Offset() != 2 {
			t.Fatalf("batched token histories advanced to %d and %d, want 3 and 2", one.Offset(), two.Offset())
		}
	})
}
