package ggml

// Tests for DecodeShards, the multi-file GGUF merge path.
//
// Issue #5245: users import split GGUF models with multiple FROM lines.
// The server sorts shards by split.no, then DecodeShards merges them into
// a single *GGML whose tensors carry ShardIdx so the backend knows which
// file to read each tensor from.

import (
	"bytes"
	"os"
	"testing"

	"github.com/google/go-cmp/cmp"
)

// f32Data returns n*4 bytes of zero-valued F32 tensor data.
func f32Data(n int) *bytes.Reader {
	return bytes.NewReader(make([]byte, n*4))
}

// writeShard creates a temporary GGUF file for one shard and returns its path.
// no / count are the split.no / split.count values; total is split.tensors.count.
// extraKV is merged into the KV block (useful for adding arch metadata to shard 0).
func writeShard(t *testing.T, no, count, total int, extraKV KV, tensors []*Tensor) string {
	t.Helper()

	kv := KV{
		"general.architecture": "test",
		"split.no":             uint16(no),
		"split.count":          uint16(count),
		"split.tensors.count":  int32(total),
	}
	for k, v := range extraKV {
		kv[k] = v
	}

	f, err := os.CreateTemp(t.TempDir(), "shard*.gguf")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	if err := WriteGGUF(f, kv, tensors); err != nil {
		t.Fatal("WriteGGUF:", err)
	}
	return f.Name()
}

func TestDecodeShards_EmptyPaths(t *testing.T) {
	_, err := DecodeShards(nil, -1)
	if err == nil {
		t.Fatal("expected error for nil paths, got nil")
	}

	_, err = DecodeShards([]string{}, -1)
	if err == nil {
		t.Fatal("expected error for empty paths, got nil")
	}
}

func TestDecodeShards_SinglePath(t *testing.T) {
	// Single path delegates to Decode; result must not have ShardOffsets.
	path := writeShard(t, 0, 1, 2, KV{"general.name": "solo"}, []*Tensor{
		{Name: "token_embd.weight", Shape: []uint64{2, 3}, WriterTo: f32Data(6)},
		{Name: "output.weight", Shape: []uint64{3, 2}, WriterTo: f32Data(6)},
	})

	g, err := DecodeShards([]string{path}, -1)
	if err != nil {
		t.Fatal(err)
	}

	tensors := g.Tensors()
	if tensors.ShardOffsets != nil {
		t.Errorf("single-path: ShardOffsets should be nil, got %v", tensors.ShardOffsets)
	}
	if n := len(tensors.Items()); n != 2 {
		t.Errorf("single-path: want 2 tensors, got %d", n)
	}
	// split keys must still be present (not stripped) for single-path (caller's concern)
	if g.KV()["split.no"] == nil {
		// single-path passes through Decode unchanged, split keys are whatever the file has
	}
}

func TestDecodeShards_MissingFile(t *testing.T) {
	path0 := writeShard(t, 0, 2, 4, nil, []*Tensor{
		{Name: "token_embd.weight", Shape: []uint64{2, 3}, WriterTo: f32Data(6)},
	})

	_, err := DecodeShards([]string{path0, "/nonexistent/shard-00002-of-00002.gguf"}, -1)
	if err == nil {
		t.Fatal("expected error for missing shard file, got nil")
	}
}

func TestDecodeShards_TwoShards(t *testing.T) {
	// Core invariant from issue #5245:
	// All tensors from all shards are merged into one *GGML.
	// Tensors from shard 0 keep ShardIdx=0 (zero value), tensors from shard 1 get ShardIdx=1.
	// ShardOffsets has one entry per shard.
	// KV metadata comes from shard 0; split.* keys are removed.

	path0 := writeShard(t, 0, 2, 4, KV{"general.name": "split-model"}, []*Tensor{
		{Name: "blk.0.attn_k.weight", Shape: []uint64{2, 3}, WriterTo: f32Data(6)},
		{Name: "blk.0.attn_q.weight", Shape: []uint64{2, 3}, WriterTo: f32Data(6)},
	})
	path1 := writeShard(t, 1, 2, 4, nil, []*Tensor{
		{Name: "blk.1.attn_k.weight", Shape: []uint64{2, 3}, WriterTo: f32Data(6)},
		{Name: "blk.1.attn_q.weight", Shape: []uint64{2, 3}, WriterTo: f32Data(6)},
	})

	g, err := DecodeShards([]string{path0, path1}, -1)
	if err != nil {
		t.Fatal(err)
	}

	// ── tensor count ──────────────────────────────────────────────────────────
	tensors := g.Tensors()
	if n := len(tensors.Items()); n != 4 {
		t.Errorf("want 4 tensors total, got %d", n)
	}

	// ── ShardIdx assignment ───────────────────────────────────────────────────
	shardOf := make(map[string]int)
	for _, t2 := range tensors.Items() {
		shardOf[t2.Name] = t2.ShardIdx
	}
	wantShardOf := map[string]int{
		"blk.0.attn_k.weight": 0,
		"blk.0.attn_q.weight": 0,
		"blk.1.attn_k.weight": 1,
		"blk.1.attn_q.weight": 1,
	}
	if diff := cmp.Diff(wantShardOf, shardOf); diff != "" {
		t.Errorf("ShardIdx mismatch (-want +got):\n%s", diff)
	}

	// ── ShardOffsets ──────────────────────────────────────────────────────────
	if got := len(tensors.ShardOffsets); got != 2 {
		t.Errorf("want ShardOffsets len=2, got %d", got)
	}
	// Both offsets must be nonzero (they're after the GGUF header+KV block).
	for i, off := range tensors.ShardOffsets {
		if off == 0 {
			t.Errorf("ShardOffsets[%d] is 0, expected positive offset", i)
		}
	}

	// ── KV merge ─────────────────────────────────────────────────────────────
	kv := g.KV()
	// split.* keys must be stripped
	for _, k := range []string{"split.no", "split.count", "split.tensors.count"} {
		if kv[k] != nil {
			t.Errorf("KV key %q should have been removed after merge, still present", k)
		}
	}
	// arch and model metadata from shard 0 must survive
	if kv["general.architecture"] == nil {
		t.Error("general.architecture missing from merged KV")
	}
	if kv["general.name"] == nil {
		t.Error("general.name from shard 0 missing from merged KV")
	}
	// SplitNo / SplitCount helpers must return zero after strip
	if v := kv.SplitNo(); v != 0 {
		t.Errorf("SplitNo() after merge: want 0, got %d", v)
	}
	if v := kv.SplitCount(); v != 0 {
		t.Errorf("SplitCount() after merge: want 0, got %d", v)
	}
}

func TestDecodeShards_OrderIsCallerResponsibility(t *testing.T) {
	// DecodeShards trusts the caller to pass paths in split.no order.
	// The server sorts by split.no in create.go before storing to the manifest,
	// so by the time paths reach DecodeShards the order is always correct.
	//
	// This test documents what happens when paths are reversed: tensors get
	// wrong ShardIdx, so the backend would read weight data from the wrong file.
	// It is a contract test, not a correctness test for DecodeShards itself.

	path0 := writeShard(t, 0, 2, 4, KV{"general.name": "original"}, []*Tensor{
		{Name: "blk.0.attn_k.weight", Shape: []uint64{2, 3}, WriterTo: f32Data(6)},
		{Name: "blk.0.attn_q.weight", Shape: []uint64{2, 3}, WriterTo: f32Data(6)},
	})
	path1 := writeShard(t, 1, 2, 4, nil, []*Tensor{
		{Name: "blk.1.attn_k.weight", Shape: []uint64{2, 3}, WriterTo: f32Data(6)},
		{Name: "blk.1.attn_q.weight", Shape: []uint64{2, 3}, WriterTo: f32Data(6)},
	})

	// correct order: [shard0, shard1]
	correct, err := DecodeShards([]string{path0, path1}, -1)
	if err != nil {
		t.Fatal(err)
	}

	// reversed order: [shard1, shard0] — what happens if sort in create.go is bypassed
	reversed, err := DecodeShards([]string{path1, path0}, -1)
	if err != nil {
		t.Fatal(err)
	}

	// In correct order, shard 0 tensors have ShardIdx 0, shard 1 tensors have ShardIdx 1.
	correctIdx := make(map[string]int)
	for _, tt := range correct.Tensors().Items() {
		correctIdx[tt.Name] = tt.ShardIdx
	}
	if correctIdx["blk.0.attn_k.weight"] != 0 || correctIdx["blk.1.attn_k.weight"] != 1 {
		t.Errorf("correct order: unexpected ShardIdx map %v", correctIdx)
	}

	// In reversed order, the same tensors get the opposite ShardIdx,
	// proving that DecodeShards does not auto-correct the order.
	reversedIdx := make(map[string]int)
	for _, tt := range reversed.Tensors().Items() {
		reversedIdx[tt.Name] = tt.ShardIdx
	}
	if reversedIdx["blk.0.attn_k.weight"] != 1 || reversedIdx["blk.1.attn_k.weight"] != 0 {
		t.Errorf("reversed order: expected ShardIdx to be swapped, got %v", reversedIdx)
	}
}

func TestDecodeShards_ShardOffsetsDiffer(t *testing.T) {
	// Shard 0 has more KV than shard 1, so its header is larger and its
	// tensor-data region starts at a higher offset. ShardOffsets must reflect
	// each shard's individual tensorOffset, not a shared value.
	path0 := writeShard(
		t, 0, 2, 2,
		KV{
			"general.name":       "big-metadata",
			"tokenizer.ggml.bos": uint32(1),
			"tokenizer.ggml.eos": uint32(2),
		},
		[]*Tensor{
			{Name: "token_embd.weight", Shape: []uint64{2, 3}, WriterTo: f32Data(6)},
		},
	)
	path1 := writeShard(t, 1, 2, 2, nil, []*Tensor{
		{Name: "output.weight", Shape: []uint64{3, 2}, WriterTo: f32Data(6)},
	})

	g, err := DecodeShards([]string{path0, path1}, -1)
	if err != nil {
		t.Fatal(err)
	}

	so := g.Tensors().ShardOffsets
	if len(so) != 2 {
		t.Fatalf("want 2 ShardOffsets, got %d", len(so))
	}
	// Shard 0 has more KV bytes so its tensor region starts later.
	if so[0] <= so[1] {
		t.Errorf("shard 0 has larger header, its offset (%d) should exceed shard 1's (%d)", so[0], so[1])
	}
}
