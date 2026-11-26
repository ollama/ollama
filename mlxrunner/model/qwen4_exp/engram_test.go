package qwen4_exp

import (
	"fmt"
	"slices"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
)

func TestEngramHashes(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		p := &PLE{
			LayerMultipliers: mlx.FromValues([]int64{3, 5, 7}, 3),
			HeadVocabSizes:   mlx.FromValues([]int64{11, 13, 17, 19}, 4),
			HeadOffsets:      mlx.FromValues([]int64{0, 11, 24, 41}, 4),
		}
		cfg := &Config{NGramSize: 3, HeadsPerNGram: 2, EOSTokenID: 9}
		b := &batch.Batch{InputIDs: mlx.FromValues([]int32{1, 9, 2, 3}, 1, 4)}
		history := mlx.FromValues([]int64{7, 8}, 1, 2)

		got, _ := p.hashes(b, history, cfg)
		got = got.AsType(mlx.DTypeInt32)
		mlx.Eval(got)
		values := got.Ints()
		want := []int32{10, 15, 33, 48, 8, 15, 28, 41, 10, 15, 27, 42, 3, 14, 33, 44}
		if !slices.Equal(values, want) {
			t.Fatalf("hashes = %v, want %v", values, want)
		}
	})
}

func TestEngramCacheCarriesChunkHistory(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		c := newEngramCache(2, 3, 1, 9)
		t.Cleanup(c.Free)
		b := &batch.Batch{
			InputIDs:     mlx.FromValues([]int32{1, 2}, 1, 2),
			SeqQueryLens: []int32{2},
		}
		c.put(b, mlx.FromValues([]int64{1, 2}, 1, 2), mlx.FromValues([]float32{10, 20}, 1, 2, 1))
		b.InputIDs = mlx.FromValues([]int32{3, 4}, 1, 2)
		c.put(b, mlx.FromValues([]int64{3, 4}, 1, 2), mlx.FromValues([]float32{30, 40}, 1, 2, 1))

		history := c.history.AsType(mlx.DTypeInt32)
		mlx.Eval(history, c.convHistory)
		if got, want := history.Ints(), []int32{3, 4}; !slices.Equal(got, want) {
			t.Fatalf("token history = %v, want %v", got, want)
		}
		if got, want := c.convHistory.Floats(), []float32{20, 30, 40}; !slices.Equal(got, want) {
			t.Fatalf("conv history = %v, want %v", got, want)
		}
	})
}

func TestEngramCacheRestoresScheduledSnapshot(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		c := newEngramCache(2, 3, 1, 9)
		t.Cleanup(c.Free)
		b := &batch.Batch{
			InputIDs:     mlx.FromValues([]int32{1, 2}, 1, 2),
			SeqQueryLens: []int32{2},
		}
		c.put(b, mlx.FromValues([]int64{1, 2}, 1, 2), mlx.FromValues([]float32{10, 20}, 1, 2, 1))

		c.PrepareSnapshots([]int{3, 4})
		b.InputIDs = mlx.FromValues([]int32{3, 4}, 1, 2)
		inputIDs := mlx.FromValues([]int64{3, 4}, 1, 2)
		convInput := mlx.FromValues([]float32{30, 40}, 1, 2, 1)
		c.put(b, inputIDs, convInput)
		snapshots := c.TakeSnapshots()
		if len(snapshots) != 2 || snapshots[0] == nil || snapshots[1] == nil {
			t.Fatalf("TakeSnapshots() = %v, want two captured snapshots", snapshots)
		}
		for _, snapshot := range snapshots {
			t.Cleanup(snapshot.Close)
		}
		first := snapshots[0].(*engramSnapshot)
		mlx.Eval(first.history, first.convHistory)
		if got, want := first.convHistory.Floats(), []float32{10, 20, 30}; !slices.Equal(got, want) {
			t.Fatalf("captured convolution history = %v, want %v", got, want)
		}

		b.InputIDs = mlx.FromValues([]int32{5, 6}, 1, 2)
		c.put(b, mlx.FromValues([]int64{5, 6}, 1, 2), mlx.FromValues([]float32{50, 60}, 1, 2, 1))
		if !c.Restore(snapshots[0], 3) {
			t.Fatal("Restore(snapshot, 3) failed")
		}
		history := c.history.AsType(mlx.DTypeInt32)
		mlx.Eval(history, c.convHistory)
		if got, want := history.Ints(), []int32{2, 3}; !slices.Equal(got, want) {
			t.Fatalf("restored token history = %v, want %v", got, want)
		}
		if got, want := c.convHistory.Floats(), []float32{10, 20, 30}; !slices.Equal(got, want) {
			t.Fatalf("restored convolution history = %v, want %v", got, want)
		}

		parent, child := c.Split(snapshots[1], 3)
		if parent != nil || child != snapshots[1] {
			t.Fatalf("Split(snapshot) = (%v, %v), want (nil, snapshot)", parent, child)
		}
		if merged := c.Merge(nil, child); merged != child {
			t.Fatalf("Merge(nil, child) = %v, want child", merged)
		}
	})
}

func TestEngramIndependentOwners(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		a, b := newEngramCache(2, 3, 1, 9), newEngramCache(2, 3, 1, 9)
		defer a.Free()
		defer b.Free()
		for i, c := range []*engramCache{a, b} {
			ids := mlx.FromValues([]int32{int32(i + 1), int32(i + 2)}, 1, 2)
			batch := &batch.Batch{InputIDs: ids, SeqOffsets: []int32{0}, SeqQueryLens: []int32{2}}
			c.put(batch, ids.AsType(mlx.DTypeInt64), mlx.MulScalar(ids.AsType(mlx.DTypeFloat32).ExpandDims(2), 10))
			c.PrepareSnapshots([]int{3})
		}
		layers, err := cache.Batch([][]cache.Cache{{b}, {a}})
		if err != nil {
			t.Fatal(err)
		}
		view := layers[0].(engramState)
		rb := &batch.Batch{InputIDs: mlx.FromValues([]int32{4, 7}, 2, 1), SeqOffsets: []int32{2, 2}, SeqQueryLens: []int32{1, 1}}
		history := view.get(rb).AsType(mlx.DTypeInt32)
		mlx.Eval(history)
		if got := history.Ints(); !slices.Equal(got, []int32{2, 3, 1, 2}) {
			t.Fatalf("gathered histories = %v", got)
		}
		view.put(rb, rb.InputIDs.AsType(mlx.DTypeInt64), mlx.FromValues([]float32{40, 70}, 2, 1, 1))
		for i, c := range []*engramCache{b, a} {
			snapshot := c.TakeSnapshots()[0]
			if snapshot == nil {
				t.Fatal("missing row snapshot")
			}
			defer snapshot.Close()
			if !c.Restore(snapshot, 3) {
				t.Fatal("failed to restore batched state")
			}
			h := c.history.AsType(mlx.DTypeInt32)
			mlx.Eval(h, c.convHistory)
			wantTokens := [][]int32{{3, 4}, {2, 7}}[i]
			wantConv := [][]float32{{20, 30, 40}, {10, 20, 70}}[i]
			if !slices.Equal(h.Ints(), wantTokens) || !slices.Equal(c.convHistory.Floats(), wantConv) {
				t.Fatalf("row %d history=%v conv=%v", i, h.Ints(), c.convHistory.Floats())
			}
		}
		incompatible := newEngramCache(2, 4, 1, 9)
		defer incompatible.Free()
		if _, err := cache.Batch([][]cache.Cache{{a}, {incompatible}}); err == nil {
			t.Fatal("accepted incompatible Engram owners")
		}
	})
}

func TestEngramSnapshotRetainsOnlyTail(t *testing.T) {
	for _, mode := range []string{"direct", "scheduled", "live"} {
		for _, dtype := range []mlx.DType{mlx.DTypeFloat32, mlx.DTypeBFloat16} {
			for _, tokens := range []int{128, 1024} {
				t.Run(fmt.Sprintf("%s/%s/%d", mode, dtype, tokens), func(t *testing.T) {
					mlxtest.Run(t, func(t *mlxtest.T) {
						if !mlx.MetalIsAvailable() {
							t.Skip("requires Metal allocation accounting")
						}
						const channels = 10240
						measure := func(check bool) {
							before := engramActiveMemory()
							c := newEngramCache(2, 9, channels, 9)
							defer c.Free()
							end := tokens
							if mode == "scheduled" {
								end = tokens / 2
								c.PrepareSnapshots([]int{end})
							}
							putEngramChunk(c, 0, tokens, dtype)
							if c.Offset() != tokens {
								t.Fatalf("live offset = %d, want %d", c.Offset(), tokens)
							}
							history, conv, close := c.history, c.convHistory, c.Free
							size := history.NumBytes() + conv.NumBytes()
							if mode != "live" {
								var s *engramSnapshot
								if mode == "scheduled" {
									s = takeEngramSnapshots(t, c, end)[0]
								} else {
									s = c.Snapshot(0).(*engramSnapshot)
									defer s.Close()
								}
								if s.offset != end {
									t.Fatalf("snapshot offset = %d, want %d", s.offset, end)
								}
								history, conv, close, size = s.history, s.convHistory, s.Close, s.Size()
								c.Free()
							}
							checkEngramTail(t, history, conv, end, channels, dtype)
							bytesPerValue := 4
							if dtype == mlx.DTypeBFloat16 {
								bytesPerValue = 2
							}
							if want := 2*8 + 9*channels*bytesPerValue; size != want {
								t.Fatalf("logical size = %d, want %d", size, want)
							}
							if check {
								checkEngramMemory(t, before, size, 1)
							}
							close()
							if check {
								checkEngramMemory(t, before, 0, 0)
							}
						}
						measure(false) // Warm the same operations before taking the baseline.
						measure(true)
					})
				})
			}
		}
	}
}

func TestEngramSnapshotBoundariesAndRestore(t *testing.T) {
	for _, dtype := range []mlx.DType{mlx.DTypeFloat32, mlx.DTypeBFloat16} {
		t.Run(dtype.String(), func(t *testing.T) {
			mlxtest.Run(t, func(t *mlxtest.T) {
				c, cold := newEngramCache(2, 9, 2, 9), newEngramCache(2, 9, 2, 9)
				defer c.Free()
				defer cold.Free()
				c.PrepareSnapshots([]int{0, 1})
				putEngramChunk(c, 0, 1, dtype)
				snapshots := takeEngramSnapshots(t, c, 0, 1)
				c.PrepareSnapshots([]int{1, 4}) // Current offset, then a short chunk's end.
				putEngramChunk(c, 1, 3, dtype)
				snapshots = append(snapshots, takeEngramSnapshots(t, c, 1, 4)...)
				for i, end := range []int{0, 1, 1, 4} {
					checkEngramTail(t, snapshots[i].history, snapshots[i].convHistory, end, 2, dtype)
				}
				if c.Offset() != 4 {
					t.Fatalf("offset = %d, want 4", c.Offset())
				}
				checkEngramTail(t, c.history, c.convHistory, 4, 2, dtype)

				putEngramChunk(c, 4, 16, dtype)
				if c.Offset() != 20 || !c.Restore(snapshots[1], 1) || c.Offset() != 1 {
					t.Fatalf("failed to restore the earlier checkpoint: offset = %d", c.Offset())
				}
				snapshots[1].Close() // Restored state owns its handles independently.
				checkEngramTail(t, c.history, c.convHistory, 1, 2, dtype)
				putEngramChunk(c, 1, 2, dtype)
				putEngramChunk(cold, 0, 2, dtype)
				putEngramChunk(cold, 2, 1, dtype)
				if c.Offset() != 3 || cold.Offset() != 3 {
					t.Fatalf("continued offsets = (%d, %d), want (3, 3)", c.Offset(), cold.Offset())
				}
				checkEngramTail(t, c.history, c.convHistory, 3, 2, dtype)
				mlx.Scoped(func() {
					if !slices.Equal(c.history.AsType(mlx.DTypeInt32).Ints(), cold.history.AsType(mlx.DTypeInt32).Ints()) ||
						!slices.Equal(c.convHistory.AsType(mlx.DTypeFloat32).Floats(), cold.convHistory.AsType(mlx.DTypeFloat32).Floats()) {
						t.Fatal("continued state differs from cold chunked state")
					}
				})
			})
		})
	}
}

func TestEngramSnapshotsSurviveCloseOrder(t *testing.T) {
	for _, order := range [][]int{{0, 1, 2}, {2, 1, 0}, {1, 0, 2}} {
		t.Run(fmt.Sprint(order), func(t *testing.T) {
			mlxtest.Run(t, func(t *mlxtest.T) {
				if !mlx.MetalIsAvailable() {
					t.Skip("requires Metal allocation accounting")
				}
				const tokens, channels = 1024, 256
				measure := func(check bool) {
					before := engramActiveMemory()
					c := newEngramCache(2, 9, channels, 9)
					defer c.Free()
					c.PrepareSnapshots([]int{tokens / 2, tokens})
					putEngramChunk(c, 0, tokens, mlx.DTypeBFloat16)
					snapshots := takeEngramSnapshots(t, c, tokens/2, tokens)
					s := c.Snapshot(0).(*engramSnapshot) // Shares the live end checkpoint.
					defer s.Close()
					snapshots = append(snapshots, s)
					putEngramChunk(c, tokens, 1, mlx.DTypeBFloat16)
					c.Free()
					for step := range len(order) + 1 {
						size, owners := 0, 0
						for i, s := range snapshots {
							if s != nil {
								end := []int{tokens / 2, tokens, tokens}[i]
								if s.offset != end {
									t.Fatalf("snapshot %d offset = %d, want %d", i, s.offset, end)
								}
								checkEngramTail(t, s.history, s.convHistory, end, channels, mlx.DTypeBFloat16)
								size += s.Size()
								owners++
							}
						}
						if check {
							checkEngramMemory(t, before, size, owners)
						}
						if step < len(order) {
							i := order[step]
							snapshots[i].Close()
							snapshots[i] = nil
						}
					}
				}
				measure(false)
				measure(true)
			})
		})
	}
}

func TestEngramBatchedSnapshotsRetainOnlyTails(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		if !mlx.MetalIsAvailable() {
			t.Skip("requires Metal allocation accounting")
		}
		const tokens, channels = 1024, 256
		measure := func(check bool) {
			before := engramActiveMemory()
			a, b := newEngramCache(2, 9, channels, 9), newEngramCache(2, 9, channels, 9)
			defer a.Free()
			defer b.Free()
			a.PrepareSnapshots([]int{tokens})
			b.PrepareSnapshots([]int{tokens / 2})
			layers, err := cache.Batch([][]cache.Cache{{b}, {a}})
			if err != nil {
				t.Fatal(err)
			}
			mlx.Scoped(func() {
				idsB, valuesB := engramChunkValues(5, tokens, channels)
				idsA, valuesA := engramChunkValues(0, tokens, channels)
				rb := &batch.Batch{
					InputIDs:   mlx.FromValues(append(idsB, idsA...), 2, tokens),
					SeqOffsets: []int32{0, 0}, SeqQueryLens: []int32{tokens, tokens},
				}
				layers[0].(*engramBatch).put(rb, rb.InputIDs.AsType(mlx.DTypeInt64),
					mlx.FromValues(append(valuesB, valuesA...), 2, tokens, channels).AsType(mlx.DTypeBFloat16))
			})
			mlx.Eval(layers[0].State()...)
			sa, sb := takeEngramSnapshots(t, a, tokens)[0], takeEngramSnapshots(t, b, tokens/2)[0]
			if a.Offset() != tokens || b.Offset() != tokens {
				t.Fatalf("batched offsets = (%d, %d), want (%d, %d)", a.Offset(), b.Offset(), tokens, tokens)
			}
			putEngramChunk(a, tokens, 1, mlx.DTypeBFloat16)
			b.Free()
			checkEngramTail(t, sa.history, sa.convHistory, tokens, channels, mlx.DTypeBFloat16)
			checkEngramTail(t, sb.history, sb.convHistory, 5+tokens/2, channels, mlx.DTypeBFloat16)
			a.Free()
			if check {
				checkEngramMemory(t, before, sa.Size()+sb.Size(), 2)
			}
			sa.Close()
			checkEngramTail(t, sb.history, sb.convHistory, 5+tokens/2, channels, mlx.DTypeBFloat16)
			if check {
				checkEngramMemory(t, before, sb.Size(), 1)
			}
			sb.Close()
			if check {
				checkEngramMemory(t, before, 0, 0)
			}
		}
		measure(false)
		measure(true)
	})
}

func engramChunkValues(start, tokens, channels int) ([]int32, []float32) {
	ids, values := make([]int32, tokens), make([]float32, tokens*channels)
	for row := range tokens {
		ids[row] = int32(100 + start + row)
		for col := range channels {
			// Small integers are exact in both FP32 and BF16.
			values[row*channels+col] = float32((start+row)%16*8 + col%8)
		}
	}
	return ids, values
}

func putEngramChunk(c *engramCache, start, tokens int, dtype mlx.DType) {
	mlx.Scoped(func() {
		ids, values := engramChunkValues(start, tokens, c.convDim)
		b := &batch.Batch{InputIDs: mlx.FromValues(ids, 1, tokens), SeqQueryLens: []int32{int32(tokens)}}
		c.put(b, b.InputIDs.AsType(mlx.DTypeInt64), mlx.FromValues(values, 1, tokens, c.convDim).AsType(dtype))
	})
	mlx.Eval(c.State()...)
}

func takeEngramSnapshots(t *mlxtest.T, c *engramCache, offsets ...int) []*engramSnapshot {
	t.Helper()
	captured := c.TakeSnapshots()
	for _, s := range captured {
		if s != nil {
			t.Cleanup(s.Close)
		}
	}
	if len(captured) != len(offsets) {
		t.Fatalf("captured %d snapshots, want %d", len(captured), len(offsets))
	}
	result := make([]*engramSnapshot, len(captured))
	for i, s := range captured {
		value, ok := s.(*engramSnapshot)
		if !ok || value.offset != offsets[i] {
			t.Fatalf("snapshot %d = %v, want offset %d", i, s, offsets[i])
		}
		result[i] = value
	}
	return result
}

func checkEngramTail(t *mlxtest.T, history, conv *mlx.Array, end, channels int, dtype mlx.DType) {
	t.Helper()
	if !slices.Equal(history.Dims(), []int{1, 2}) || history.DType() != mlx.DTypeInt64 ||
		!slices.Equal(conv.Dims(), []int{1, 9, channels}) || conv.DType() != dtype {
		t.Fatalf("history = %v/%v, conv = %v/%v", history.Dims(), history.DType(), conv.Dims(), conv.DType())
	}
	mlx.Scoped(func() {
		for i, got := range history.AsType(mlx.DTypeInt32).Ints() {
			want := int32(9)
			if row := end - 2 + i; row >= 0 {
				want = int32(100 + row)
			}
			if got != want {
				t.Fatalf("token history[%d] = %d, want %d", i, got, want)
			}
		}
		for i, got := range conv.AsType(mlx.DTypeFloat32).Floats() {
			want := float32(0)
			if row, col := end-9+i/channels, i%channels; row >= 0 {
				want = float32(row%16*8 + col%8)
			}
			if got != want {
				t.Fatalf("convolution history[%d] = %v, want %v", i, got, want)
			}
		}
	})
}

func engramActiveMemory() int {
	mlx.DefaultStream().Synchronize()
	mlx.ClearCache()
	return mlx.ActiveMemory()
}

func checkEngramMemory(t *mlxtest.T, before, size, owners int) {
	t.Helper()
	// Fixed per-owner slack covers small-buffer sharing and allocation rounding,
	// not the processed chunk length. The pinned GPU may share up to 16 KiB extra.
	const slack = 64 << 10
	retained := engramActiveMemory() - before
	lower := 0
	if owners == 0 {
		lower = -slack
	}
	t.Logf("owners=%d logical=%d retained=%d", owners, size, retained)
	if retained < lower || retained > size+max(1, owners)*slack {
		t.Errorf("retained %d bytes for %d owners with %d logical bytes", retained, owners, size)
	}
}
