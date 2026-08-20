package cache

import (
	"path/filepath"
	"slices"
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

func TestPersistedSnapshotsRoundTrip(t *testing.T) {
	skipIfNoMLX(t)

	path := filepath.Join(t.TempDir(), "snapshots.safetensors")

	kvKeys := mlx.FromValues([]float32{1, 2, 3, 4}, 1, 1, 2, 2)
	kvVals := mlx.FromValues([]float32{5, 6, 7, 8}, 1, 1, 2, 2)
	rotKeys := mlx.FromValues([]float32{9, 10, 11, 12}, 1, 1, 2, 2)
	rotVals := mlx.FromValues([]float32{13, 14, 15, 16}, 1, 1, 2, 2)
	conv := mlx.FromValues([]float32{1, 2, 3, 4, 5, 6}, 1, 2, 3)
	delta := mlx.FromValues([]float32{7, 8, 9, 10, 11, 12}, 1, 1, 2, 3)
	mlx.Pin(kvKeys, kvVals, rotKeys, rotVals, conv, delta)

	original := []Snapshot{
		&kvSnapshot{keys: kvKeys, values: kvVals, fromOffset: 0, toOffset: 2},
		&rotatingSnapshot{keys: rotKeys, values: rotVals, fromOffset: 2, toOffset: 6, idx: 2},
		&recurrentSnapshot{convState: conv, deltaState: delta, offset: 6},
	}
	defer func() {
		for _, snap := range original {
			if snap != nil {
				snap.Close()
			}
		}
	}()

	if err := SaveSnapshots(path, 6, original); err != nil {
		t.Fatalf("SaveSnapshots() error = %v", err)
	}

	loaded, err := LoadSnapshots(path)
	if err != nil {
		t.Fatalf("LoadSnapshots() error = %v", err)
	}
	defer func() {
		for _, snap := range loaded.Snapshots {
			if snap != nil {
				snap.Close()
			}
		}
	}()

	if loaded.Offset != 6 {
		t.Fatalf("loaded offset = %d, want 6", loaded.Offset)
	}
	if len(loaded.Snapshots) != len(original) {
		t.Fatalf("loaded %d snapshots, want %d", len(loaded.Snapshots), len(original))
	}

	kv := NewKVCache()
	if !kv.Restore(loaded.Snapshots[0], 2) {
		t.Fatal("restoring KV snapshot failed")
	}
	if got := kv.State()[0].Floats(); !slices.Equal(got, []float32{1, 2, 3, 4}) {
		t.Fatalf("kv keys = %v", got)
	}

	rot := NewRotatingKVCache(4)
	if !rot.Restore(loaded.Snapshots[1], 6) {
		t.Fatal("restoring rotating snapshot failed")
	}
	if got := rot.State()[0].Floats(); !slices.Equal(got, []float32{9, 10, 11, 12}) {
		t.Fatalf("rotating keys = %v", got)
	}

	rec := NewRecurrentCache(2, 3, 1, 2, 3)
	if !rec.Restore(loaded.Snapshots[2], 6) {
		t.Fatal("restoring recurrent snapshot failed")
	}
	if got := rec.State()[0].Floats(); !slices.Equal(got, []float32{1, 2, 3, 4, 5, 6}) {
		t.Fatalf("recurrent conv = %v", got)
	}
}
