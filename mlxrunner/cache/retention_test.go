package cache

import (
	"fmt"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
)

func TestPendingSnapshotCopiesAcrossBoundaries(t *testing.T) {
	for _, width := range []int{8, 64} {
		for _, heads := range []int{1, 2} {
			for _, chunk := range []int{64, 256} {
				t.Run(fmt.Sprintf("width%d/heads%d/overwrite%d", width, heads, chunk), func(t *testing.T) {
					mlxtest.Run(t, func(t *mlxtest.T) {
						c := NewKVCache()
						defer c.Free()
						update := func(start, count int) {
							mlx.Scoped(func() {
								k, v := distinctKV(start, count, heads, width)
								c.Update(newKVBatch(c.Offset(), count), k, v)
								mlx.Eval(c.State()...)
							})
							mlx.DefaultStream().Synchronize()
						}
						update(0, 256)
						before := mlx.ActiveMemory()
						var snaps []Snapshot
						var pending int
						for from := 0; from < 256; from += 64 {
							snap := c.lazySnapshot(from, from+64)
							snaps = append(snaps, snap)
							pending += PendingCopyBytes(snap)
						}
						defer func() {
							for _, snap := range snaps {
								if snap != nil {
									snap.Close()
								}
							}
						}()
						if !c.Restore(nil, 0) {
							t.Fatal("rewind failed")
						}
						for offset := 0; offset < 256; offset += chunk {
							update(1000+offset, chunk)
						}
						growth := mlx.ActiveMemory() - before
						t.Logf("pending allowance=%d actual retained growth=%d live capacity=%d", pending, growth, CopyBytes(c))
						if growth > pending {
							t.Fatalf("pending copy allowance %d missed retained growth %d", pending, growth)
						}
						for i, snap := range snaps {
							s := snap.(*kvSnapshot)
							if s.cache != nil || s.keys == nil {
								t.Fatal("overwrite did not materialize every snapshot")
							}
							if got := s.keys.Floats()[0]; got != float32(i*64) {
								t.Fatalf("snapshot %d overwritten: %v", i, got)
							}
							beforeClose := mlx.ActiveMemory()
							snap.Close()
							snaps[i] = nil
							mlx.DefaultStream().Synchronize()
							t.Logf("snapshot %d released %d", i, beforeClose-mlx.ActiveMemory())
						}
					})
				})
			}
		}
	}
}

func TestPressureMergeRequiresCompleteChildState(t *testing.T) {
	cases := []struct {
		cache  Cache
		parent Snapshot
	}{
		{&RotatingKVCache{}, &rotatingSnapshot{}},
		{&RecurrentCache{}, &recurrentSnapshot{}},
	}
	for _, tc := range cases {
		if CanMergeWithoutCopy(tc.cache, tc.parent, nil) {
			t.Fatalf("%T can lose the parent's only restore point", tc.cache)
		}
	}
}
