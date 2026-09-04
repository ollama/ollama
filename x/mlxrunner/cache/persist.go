package cache

import (
	"fmt"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

// LayerKind identifies a cache implementation for prefill persistence.
type LayerKind string

const (
	LayerKV       LayerKind = "kv"
	LayerRotating LayerKind = "rotating"
	LayerRecurrent LayerKind = "recurrent"
)

// LayerPersistMeta holds per-layer metadata saved alongside tensor data.
type LayerPersistMeta struct {
	Kind    LayerKind `json:"kind"`
	Offset  int       `json:"offset"`
	MaxSize int       `json:"max_size,omitempty"`
	Idx     int       `json:"idx,omitempty"`
}

// KindOf returns the persistence kind for a cache layer.
func KindOf(c Cache) LayerKind {
	switch c.(type) {
	case *RotatingKVCache:
		return LayerRotating
	case *RecurrentCache:
		return LayerRecurrent
	default:
		return LayerKV
	}
}

// ExportLayer snapshots live cache state for persistence.
func ExportLayer(c Cache) (map[string]*mlx.Array, LayerPersistMeta, error) {
	meta := LayerPersistMeta{Kind: KindOf(c), Offset: c.Offset()}
	arrays := make(map[string]*mlx.Array)

	switch v := c.(type) {
	case *KVCache:
		state := v.State()
		if len(state) == 0 || meta.Offset == 0 {
			return nil, meta, fmt.Errorf("empty kv cache")
		}
		arrays["keys"] = state[0]
		arrays["values"] = state[1]
	case *RotatingKVCache:
		meta.MaxSize = v.maxSize
		meta.Idx = v.idx
		state := v.State()
		if len(state) == 0 || meta.Offset == 0 {
			return nil, meta, fmt.Errorf("empty rotating cache")
		}
		arrays["keys"] = state[0]
		arrays["values"] = state[1]
	case *RecurrentCache:
		if v.convState == nil && v.deltaState == nil {
			return nil, meta, fmt.Errorf("empty recurrent cache")
		}
		if v.convState != nil {
			arrays["conv"] = v.convState
		}
		if v.deltaState != nil {
			arrays["delta"] = v.deltaState
		}
	default:
		return nil, meta, fmt.Errorf("unsupported cache type %T", c)
	}

	return arrays, meta, nil
}

// ImportLayer restores cache state from persisted arrays.
func ImportLayer(c Cache, arrays map[string]*mlx.Array, meta LayerPersistMeta) error {
	switch v := c.(type) {
	case *KVCache:
		keys, values := arrays["keys"], arrays["values"]
		if keys == nil || values == nil {
			return fmt.Errorf("missing kv tensors")
		}
		snap := &kvSnapshot{
			keys:       keys,
			values:     values,
			fromOffset: 0,
			toOffset:   meta.Offset,
		}
		mlx.Pin(keys, values)
		if !v.Restore(snap, meta.Offset) {
			return fmt.Errorf("kv restore failed")
		}
	case *RotatingKVCache:
		keys, values := arrays["keys"], arrays["values"]
		if keys == nil || values == nil {
			return fmt.Errorf("missing rotating tensors")
		}
		snap := &rotatingSnapshot{
			keys:       keys,
			values:     values,
			fromOffset: 0,
			toOffset:   meta.Offset,
			idx:        meta.Idx,
		}
		mlx.Pin(keys, values)
		if !v.Restore(snap, meta.Offset) {
			return fmt.Errorf("rotating restore failed")
		}
	case *RecurrentCache:
		conv, delta := arrays["conv"], arrays["delta"]
		if conv == nil && delta == nil {
			return fmt.Errorf("missing recurrent tensors")
		}
		snap := &recurrentSnapshot{offset: meta.Offset}
		if conv != nil {
			snap.convState = conv.Clone()
		}
		if delta != nil {
			snap.deltaState = delta.Clone()
		}
		mlx.Pin(snap.convState, snap.deltaState)
		if !v.Restore(snap, meta.Offset) {
			return fmt.Errorf("recurrent restore failed")
		}
	default:
		return fmt.Errorf("unsupported cache type %T", c)
	}
	return nil
}
