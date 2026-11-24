package cache

import (
	"fmt"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
)

// readTags returns the per-position tags of a [1, H, n, D] key/value view after
// checking that every element of a position agrees.
func readTags(k, v *mlx.Array) ([]float32, error) {
	k, v = mlx.Contiguous(k, false), mlx.Contiguous(v, false)
	mlx.Eval(k, v)
	H, n, D := k.Dim(1), k.Dim(2), k.Dim(3)
	kf, vf := k.Floats(), v.Floats()
	tags := make([]float32, n)
	for l := range n {
		tags[l] = kf[l*D]
		for h := range H {
			for d := range D {
				i := (h*n+l)*D + d
				if kf[i] != tags[l] || vf[i] != -tags[l] {
					return nil, fmt.Errorf("position %d head %d channel %d holds %v/%v, want %v", l, h, d, kf[i], vf[i], tags[l])
				}
			}
		}
	}
	return tags, nil
}

// TestOrderedWindowHistory drives an ordered cache through short and long
// writes, speculative rollbacks, live rewinds and restores of held snapshots.
// Every write must see the window a contiguous rebuild returns: up to
// window-1 earlier positions and the new rows, as most recently written.
func TestOrderedWindowHistory(t *testing.T) {
	const window = 16
	for _, step := range []int{3, 256} {
		for seed := range uint64(6) {
			mlxtest.RunSubtest(t, fmt.Sprintf("step%d/seed%d", step, seed), func(t *mlxtest.T) {
				mlx.Scoped(func() {
					if err := runOrderedSequence(window, step, seed); err != nil {
						t.Fatal(err)
					}
				})
			})
		}
	}
}

// generationTags offsets every rewrite's tags past all positions a sequence
// writes (80 operations of at most 20 rows), so stale rows never match.
const generationTags = 4096

func runOrderedSequence(window, step int, seed uint64) error {
	rng := rand.New(rand.NewPCG(seed, 11))
	c := NewOrderedWindowKVCache(window)
	c.step = step
	defer c.Free()

	// truth[p] is the tag most recently written at position p and still live.
	var truth []float32
	generation := 0
	type held struct {
		snap   Snapshot
		offset int
		truth  []float32
	}
	var saved []held
	defer func() {
		for _, h := range saved {
			h.snap.Close()
		}
	}()

	write := func(L int) error {
		o := c.Offset()
		truth = truth[:o]
		first := o + 1 + generationTags*generation
		for p := range L {
			truth = append(truth, float32(first+p))
		}
		k, v := distinctKV(first, L, 2, 4)
		h := c.Update(newKVBatch(o, L), k, v)
		got, err := readTags(h.K(), h.V())
		if err != nil {
			return err
		}
		if want := truth[o-min(o, window-1):]; !slices.Equal(got, want) {
			return fmt.Errorf("write of %d at %d: history %v, want %v", L, o, got, want)
		}
		return nil
	}
	check := func(label string) error {
		o := c.Offset()
		if c.keys != nil && (c.idx > c.keys.Dim(2) || c.idx < min(o, window)) {
			return fmt.Errorf("%s: %d live rows in %d slots at offset %d", label, c.idx, c.keys.Dim(2), o)
		}
		want := truth[o-min(o, window) : o]
		if o == 0 {
			return nil
		}
		state := c.State()
		got, err := readTags(state[0], state[1])
		if err != nil {
			return err
		}
		if got = got[max(0, len(got)-window):]; !slices.Equal(got, want) {
			return fmt.Errorf("%s: window at %d holds %v, want %v", label, o, got, want)
		}
		view := c.View(newKVBatch(o, 1))
		if got, err = readTags(view.K(), view.V()); err != nil {
			return err
		}
		if !slices.Equal(got, want) {
			return fmt.Errorf("%s: view at %d holds %v, want %v", label, o, got, want)
		}
		return nil
	}

	for op := range 80 {
		var label string
		switch r := rng.IntN(12); {
		case r < 4:
			label = "short write"
			if err := write(1 + rng.IntN(orderedAppendRows)); err != nil {
				return err
			}
		case r < 5:
			label = "long write"
			if err := write(orderedAppendRows + 1 + rng.IntN(12)); err != nil {
				return err
			}
		case r < 8:
			label = "speculation"
			L, o := 1+rng.IntN(orderedAppendRows), c.Offset()
			offsets := make([]int, L)
			for i := range offsets {
				offsets[i] = o + i
			}
			c.PrepareSnapshots(offsets)
			if err := write(L); err != nil {
				return err
			}
			snaps := c.TakeSnapshots()
			accepted := rng.IntN(L + 1)
			if accepted < L {
				for i, s := range snaps {
					if s != nil && i != accepted {
						s.Close()
						snaps[i] = nil
					}
				}
				live := rng.IntN(2) == 0 && c.Restore(nil, o+accepted)
				if !live && !c.Restore(snaps[accepted], o+accepted) {
					return fmt.Errorf("op %d: rollback to %d failed", op, o+accepted)
				}
				truth = truth[:o+accepted]
				generation++
			}
			for _, s := range snaps {
				if s != nil {
					s.Close()
				}
			}
		case r < 9:
			label = "hold snapshot"
			L, o := 1+rng.IntN(orderedAppendRows+6), c.Offset()
			at := o + rng.IntN(L)
			c.PrepareSnapshots([]int{at})
			if err := write(L); err != nil {
				return err
			}
			if s := c.TakeSnapshots()[0]; s != nil {
				saved = append(saved, held{s, at, slices.Clone(truth[:at])})
			}
		case r < 10:
			label = "restore held"
			if len(saved) == 0 {
				continue
			}
			i := rng.IntN(len(saved))
			h := saved[i]
			saved = slices.Delete(saved, i, i+1)
			if !c.Restore(h.snap, h.offset) {
				h.snap.Close()
				return fmt.Errorf("op %d: restore of held snapshot at %d failed", op, h.offset)
			}
			h.snap.Close()
			truth = h.truth
			generation++
		default:
			label = "live rewind"
			target := max(0, c.Offset()-rng.IntN(12))
			if c.Restore(nil, target) {
				truth = truth[:target]
				generation++
			}
		}
		if err := check(fmt.Sprintf("op %d (%s)", op, label)); err != nil {
			return err
		}
	}
	return nil
}

// TestOrderedSnapshotHoldsOneWindow checks that a snapshot of an ordered buffer
// with spare rows after in-place appends clones only the newest window, and
// that restoring it gives later writes the same history.
func TestOrderedSnapshotHoldsOneWindow(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		const window = 16
		mlx.Scoped(func() {
			c := NewOrderedWindowKVCache(window)
			defer c.Free()
			var tags []float32
			write := func(c *RotatingKVCache, L int) ([]float32, error) {
				o := c.Offset()
				for p := o; p < o+L; p++ {
					tags = append(tags[:p], float32(p+1))
				}
				k, v := distinctKV(o+1, L, 2, 4)
				h := c.Update(newKVBatch(o, L), k, v)
				return readTags(h.K(), h.V())
			}
			for _, L := range []int{20, 8, 8, 8} {
				if _, err := write(c, L); err != nil {
					t.Fatal(err)
				}
			}
			if c.idx <= window {
				t.Fatalf("buffer holds %d live rows, want spare rows past the %d-row window", c.idx, window)
			}
			snap := c.Snapshot(0)
			defer snap.Close()
			if got := snap.(*rotatingSnapshot).keys.Dim(2); got != window {
				t.Fatalf("snapshot holds %d rows, want %d", got, window)
			}
			restored := NewOrderedWindowKVCache(window)
			defer restored.Free()
			if !restored.Restore(snap, c.Offset()) {
				t.Fatal("restore failed")
			}
			o := restored.Offset()
			got, err := write(restored, 3)
			if err != nil {
				t.Fatal(err)
			}
			if want := tags[o-(window-1) : o+3]; !slices.Equal(got, want) {
				t.Fatalf("history after restore %v, want %v", got, want)
			}
		})
	})
}

// TestOrderedSnapshotOwnsItsWindow checks that a short prefix snapshot keeps
// only its own rows alive, not the buffer's spare capacity, so the prefix
// cache's byte count matches the memory it holds.
func TestOrderedSnapshotOwnsItsWindow(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.ClearCache()
		before := mlx.ActiveMemory()
		var snap Snapshot
		capacity := 0
		mlx.Scoped(func() {
			c := NewOrderedWindowKVCache(1024)
			k := mlx.Zeros(mlx.DTypeBFloat16, 1, 8, 1, 128)
			c.Update(newKVBatch(0, 1), k, k)
			// Serving evaluates the buffer before the prefix cache saves it.
			mlx.Eval(c.keys, c.values)
			snap = c.Snapshot(0)
			capacity = c.keys.Dim(2)
			c.Free()
		})
		defer snap.Close()
		if capacity < 8 {
			t.Fatalf("buffer holds %d rows, want spare capacity past the prefix", capacity)
		}
		mlx.DefaultStream().Synchronize()
		mlx.ClearCache()
		// Allocator rounding keeps a little beyond the snapshot's rows.
		if retained, counted := mlx.ActiveMemory()-before, snap.Size(); retained > counted+64<<10 {
			t.Fatalf("snapshot counts %d bytes but keeps %d alive", counted, retained)
		}
	})
}

// TestOrderedWindowAppendsInPlace checks that a full window takes short writes
// without rebuilding it, including speculative rounds that schedule rollback
// points and rewind. Rebuilding would allocate a new window for keys and values.
func TestOrderedWindowAppendsInPlace(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		const window, H, D = 1024, 8, 128
		rows := func(L int) (*mlx.Array, *mlx.Array) {
			k := mlx.AddScalar(mlx.Zeros(mlx.DTypeFloat32, 1, H, L, D), 1).AsType(mlx.DTypeBFloat16)
			v := mlx.AddScalar(mlx.Zeros(mlx.DTypeFloat32, 1, H, L, D), 2).AsType(mlx.DTypeBFloat16)
			mlx.Eval(k, v)
			return k, v
		}
		c := NewOrderedWindowKVCache(window)
		// Little spare capacity, so speculative rounds also compact and their
		// rollback points must move with the kept rows instead of copying.
		c.step = 32
		defer c.Free()
		mlx.Scoped(func() {
			k, v := rows(window + 7)
			h := c.Update(newKVBatch(0, k.Dim(2)), k, v)
			mlx.Eval(h.K(), h.V())
		})
		windowBytes := 2 * window * H * D * 2
		measured, compactedSpeculating := 0, 0
		for i := range 96 {
			L, speculate := []int{1, 4, 8, 3}[i%4], i%3 != 0
			mlx.Scoped(func() {
				k, v := rows(L)
				o := c.Offset()
				inPlace := c.idx+L <= c.keys.Dim(2)
				if speculate && !inPlace {
					compactedSpeculating++
				}
				// Measure from an empty allocator cache so earlier writes' buffers do not count.
				mlx.ClearCache()
				before := mlx.ActiveMemory()
				peak := mlx.MeasurePeakMemory(func() {
					if speculate {
						offsets := make([]int, L)
						for j := range offsets {
							offsets[j] = o + j
						}
						c.PrepareSnapshots(offsets)
					}
					h := c.Update(newKVBatch(o, L), k, v)
					if h.K().Dim(2) != window-1+L {
						t.Fatalf("write %d: history has %d rows, want %d", i, h.K().Dim(2), window-1+L)
					}
					mlx.Eval(h.K(), h.V())
					if speculate {
						snaps := c.TakeSnapshots()
						if !c.Restore(nil, o+L/2) {
							t.Fatalf("write %d: live rewind to %d failed", i, o+L/2)
						}
						for _, s := range snaps {
							if s != nil {
								if s.Size() != 0 {
									t.Fatalf("write %d: rollback point owns %d bytes", i, s.Size())
								}
								s.Close()
							}
						}
					}
				})
				if inPlace {
					measured++
					if grew := peak - before; grew >= windowBytes/4 {
						t.Fatalf("write %d of %d rows allocated %d bytes; a window is %d bytes", i, L, grew, windowBytes)
					}
				}
			})
		}
		if measured < 64 {
			t.Fatalf("only %d of 96 writes fit the buffer", measured)
		}
		if compactedSpeculating == 0 {
			t.Fatal("no speculative write compacted the buffer")
		}
	})
}
