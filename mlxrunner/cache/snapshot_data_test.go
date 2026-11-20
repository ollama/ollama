package cache

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
)

// persistedData crosses the actual file/JSON boundary, so the tests exercise
// independently owned restored arrays rather than reusing live tensor handles.
func persistedData(t *mlxtest.T, dir string, data SnapshotData) SnapshotData {
	t.Helper()
	f, err := mlx.NewTensorFile(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	var record struct {
		State   SnapshotData
		Tensors []mlx.TensorExtent
	}
	record.State = data
	for _, a := range data.Arrays {
		writer, err := f.BeginTensor(a)
		if err != nil {
			t.Fatal(err)
		}
		for !writer.Done() {
			chunk, err := writer.Stage(context.Background(), a)
			if err != nil {
				t.Fatal(err)
			}
			if err := writer.Write(context.Background(), chunk); err != nil {
				t.Fatal(err)
			}
		}
		record.Tensors = append(record.Tensors, writer.Extent())
	}
	b, err := json.Marshal(record)
	if err != nil {
		t.Fatal(err)
	}
	if err := f.Seal(b); err != nil {
		t.Fatal(err)
	}
	if err := f.Publish(); err != nil {
		t.Fatal(err)
	}
	reader, metadata, err := mlx.OpenTensorFile(f.Path())
	if err != nil {
		t.Fatal(err)
	}
	defer reader.Close()
	record.State = SnapshotData{}
	if err := json.Unmarshal(metadata, &record); err != nil {
		t.Fatal(err)
	}
	for _, e := range record.Tensors {
		a, err := reader.Read(context.Background(), e)
		if err != nil {
			t.Fatal(err)
		}
		record.State.Arrays = append(record.State.Arrays, a)
	}
	return record.State
}

func TestPersistedKVPrefixContinuity(t *testing.T) {
	dir := t.TempDir()
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.Scoped(func() {
			source, restored := NewKVCache(), NewKVCache()
			defer source.Free()
			defer restored.Free()
			source.PrepareSnapshots([]int{2, 4})
			k := mlx.FromValues([]float32{1, 2, 3, 4}, 1, 1, 4, 1)
			source.Update(newKVBatch(0, 4), k, k.Multiply(mlx.FromValue(float32(2))))
			snaps := source.TakeSnapshots()
			head := persistedData(t, dir, source.SnapshotData(snaps[0]))
			tail := persistedData(t, dir, source.SnapshotData(snaps[1]))
			for _, s := range snaps {
				if s.Size() != 0 {
					t.Fatal("writing a lazy snapshot materialized it")
				}
				s.Close()
			}
			source.Free()
			if restored.RestoreData(tail, 4) {
				t.Fatal("restored suffix without its prefix")
			}
			if !restored.RestoreData(head, 2) || !restored.RestoreData(tail, 4) {
				t.Fatal("could not restore continuous history")
			}
			for i, want := range [][]float32{{1, 2, 3, 4}, {2, 4, 6, 8}} {
				if got := restored.State()[i].Floats(); !slices.Equal(got, want) {
					t.Fatalf("state %d = %v, want %v", i, got, want)
				}
			}
		})
	})
}

func TestPersistedRotatingWindow(t *testing.T) {
	for _, lengths := range [][]int{{3}, {9}, {4, 1, 1, 1}, {9, 1, 1}, {3, 7}} {
		t.Run(fmt.Sprint(lengths), func(t *testing.T) {
			dir := t.TempDir()
			mlxtest.Run(t, func(t *mlxtest.T) {
				mlx.Scoped(func() {
					source, restored := NewRotatingKVCache(4), NewRotatingKVCache(4)
					defer source.Free()
					defer restored.Free()
					for _, n := range lengths {
						values := make([]float32, n)
						for i := range values {
							values[i] = float32(source.Offset() + i)
						}
						a := mlx.FromValues(values, 1, 1, n, 1)
						source.Update(newKVBatch(source.Offset(), n), a, a)
					}
					snap := source.Snapshot(0)
					data := persistedData(t, dir, source.SnapshotData(snap))
					snap.Close()
					if !restored.RestoreData(data, source.Offset()) {
						t.Fatal("window restore failed")
					}
					// Advancing both exercises the persisted write cursor and ring
					// order, including an oversized prefill followed by decode.
					for range 5 {
						a := mlx.FromValues([]float32{float32(source.Offset())}, 1, 1, 1, 1)
						for _, c := range []*RotatingKVCache{source, restored} {
							c.Update(newKVBatch(c.Offset(), 1), a, a)
						}
						if !slices.Equal(source.State()[0].Floats(), restored.State()[0].Floats()) {
							t.Fatal("restored window diverged during decode")
						}
					}
					if data.End > 4 && restored.RestoreData(data, data.End-1) {
						t.Fatal("rewound a wrapped checkpoint")
					}
				})
			})
		})
	}
}

func TestPersistedRecurrentAndHiddenState(t *testing.T) {
	dir := t.TempDir()
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.Scoped(func() {
			r := NewRecurrentCache(2, 3, 1, 2, 2)
			defer r.Free()
			b := newKVBatch(0, 4)
			r.Put(b, []*mlx.Array{mlx.AddScalar(mlx.Zeros(mlx.DTypeFloat16, 1, 2, 3), 7)}, []*mlx.Array{mlx.AddScalar(mlx.Zeros(mlx.DTypeFloat32, 1, 1, 2, 2), 8)})
			snap := r.Snapshot(0)
			data := persistedData(t, dir, r.SnapshotData(snap))
			snap.Close()
			r.Free()
			if r.RestoreData(data, 3) || !r.RestoreData(data, 4) {
				t.Fatal("recurrent restore lost its exact-endpoint restriction")
			}
			if r.State()[0].AsType(mlx.DTypeFloat32).Floats()[0] != 7 || r.State()[1].Floats()[0] != 8 {
				t.Fatal("recurrent state did not round trip")
			}
			wrong := NewRecurrentCache(2, 3, 2, 2, 2)
			defer wrong.Free()
			if wrong.RestoreData(data, 4) || wrong.Offset() != 0 {
				t.Fatal("accepted incompatible recurrent dimensions")
			}

			h := NewHiddenCache()
			defer h.Free()
			h.Append(mlx.FromValues([]float32{1, 2, 3, 4, 5, 6}, 1, 3, 2))
			head, tail := h.Split(h.Snapshot(0), 1)
			headData := persistedData(t, dir, h.SnapshotData(head))
			tailData := persistedData(t, dir, h.SnapshotData(tail))
			head.Close()
			tail.Close()
			h.Free()
			if h.RestoreData(tailData, 3) || !h.RestoreData(headData, 1) || !h.RestoreData(tailData, 3) {
				t.Fatal("hidden state did not preserve prefix continuity")
			}
			if !slices.Equal(h.State()[0].Floats(), []float32{1, 2, 3, 4, 5, 6}) {
				t.Fatal("hidden state did not round trip")
			}
		})
	})
}
