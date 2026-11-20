package mlxrunner

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/batch"
	"github.com/ollama/ollama/mlxrunner/cache"
	"github.com/ollama/ollama/mlxrunner/model"
)

func checkpointTestCaches() []cache.Cache {
	return []cache.Cache{cache.NewKVCache(), cache.NewRotatingKVCache(4), cache.NewRecurrentCache(2, 3, 1, 2, 2), cache.NewHiddenCache(), nil}
}

func checkpointTestFeed(cs []cache.Cache, tokens []int32) {
	for _, token := range tokens {
		mlx.Scoped(func() {
			offset := cs[0].Offset()
			b := &batch.Batch{InputIDs: mlx.FromValues([]int32{token}, 1, 1), SeqOffsets: []int32{int32(offset)}, SeqQueryLens: []int32{1}}
			a := mlx.FromValues([]float32{float32(token)}, 1, 1, 1, 1)
			for _, c := range cs {
				switch c := c.(type) {
				case cache.Attention:
					c.Update(b, a, a)
				case *cache.RecurrentCache:
					c.Put(b, []*mlx.Array{mlx.AddScalar(mlx.Zeros(mlx.DTypeFloat32, 1, 2, 3), float32(offset))}, []*mlx.Array{mlx.AddScalar(mlx.Zeros(mlx.DTypeFloat32, 1, 1, 2, 2), float32(token))})
				case *cache.HiddenCache:
					c.Append(a.Reshape(1, 1, 1))
				}
				if c != nil {
					mlx.Eval(c.State()...)
				}
			}
		})
	}
}

func drainCheckpoint(t *mlxtest.T, d *checkpointCache) error {
	t.Helper()
	deadline := time.NewTimer(10 * time.Second)
	defer deadline.Stop()
	for d.job != nil {
		var progressed bool
		var err error
		mlx.Scoped(func() { progressed, err = d.step() })
		if err != nil {
			return err
		}
		if !progressed {
			select {
			case <-d.ready:
			case <-deadline.C:
				t.Fatal("checkpoint writer did not finish")
			}
		}
	}
	return nil
}

func TestCheckpointRestartAndRestore(t *testing.T) {
	root := t.TempDir()
	ctx := t.Context()
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.Scoped(func() {
			c := newPrefixCache(checkpointTestCaches())
			defer c.close()
			s := c.beginScore([]int32{1, 2, 3, 4, 5, 6, 7}, nil)
			s.schedulePrefillSnapshots([]int{4})
			checkpointTestFeed(c.caches, s.remaining)
			s.close()
			store, err := openCacheDiskStore(ctx, root, "restart")
			if err != nil {
				t.Fatal(err)
			}
			d, err := openCheckpointCache(context.Background(), store, "restart", c.caches, make(chan struct{}, 1))
			if err != nil {
				t.Fatal(err)
			}
			for _, node := range c.activePath[1:] {
				if err := d.start(c, node, nil); err != nil {
					t.Fatal(err)
				}
				if err := drainCheckpoint(t, d); err != nil {
					t.Fatal(err)
				}
			}
			if len(d.saved) != 2 {
				t.Fatalf("saved %d checkpoints, want prompt and completion", len(d.saved))
			}
			c.close()
			d.close()

			c = newPrefixCache(checkpointTestCaches())
			defer c.close()
			store, err = openCacheDiskStore(ctx, root, "restart")
			if err != nil {
				t.Fatal(err)
			}
			d, err = openCheckpointCache(context.Background(), store, "restart", c.caches, make(chan struct{}, 1))
			if err != nil {
				t.Fatal(err)
			}
			defer d.close()
			for _, tokens := range [][]int32{{1, 2, 3, 4, 99}, {1, 2, 3, 4, 5, 6, 7, 8}} {
				c.close()
				p := d.restore(c, tokens, nil, len(tokens)-1, 0)
				if p == nil {
					t.Fatalf("restart missed %v", tokens)
				}
				for {
					var done bool
					var err error
					mlx.Scoped(func() { done, err = p.step(context.Background()) })
					if err != nil {
						t.Fatal(err)
					}
					if done {
						break
					}
				}
				if err := p.commit(); err != nil {
					t.Fatal(err)
				}
				if c.minCacheOffset() != len(tokens)-1 {
					t.Fatalf("restored %d tokens, want %d", c.minCacheOffset(), len(tokens)-1)
				}
				got := c.caches[0].State()[0].AsType(mlx.DTypeInt32).Ints()
				if !slices.Equal(got, tokens[:len(tokens)-1]) {
					t.Fatalf("restored KV %v, want %v", got, tokens[:len(tokens)-1])
				}
				fresh := checkpointTestCaches()
				checkpointTestFeed(fresh, tokens[:len(tokens)-1])
				for i, layer := range c.caches {
					if layer == nil {
						continue
					}
					for j, a := range layer.State() {
						if !slices.Equal(a.Floats(), fresh[i].State()[j].Floats()) {
							t.Fatalf("layer %d array %d differs from recomputed prefix", i, j)
						}
					}
					fresh[i].Free()
				}
				if len(tokens) == 5 {
					// Restoring a prompt checkpoint must preserve its retention
					// boundary. Completing a reply must not merge it away and
					// turn the next repeat into another disk read or recomputation.
					next := c.begin(tokens, nil)
					checkpointTestFeed(c.caches, next.remaining)
					next.close()
					repeat := c.begin(tokens, nil)
					if len(repeat.remaining) != 1 {
						t.Fatalf("restored prompt was not retained in RAM: %d tokens left", len(repeat.remaining))
					}
				}
			}
		})
	})
}

func TestCheckpointInterruptedByNextRequest(t *testing.T) {
	root := t.TempDir()
	ctx := t.Context()
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.Scoped(func() {
			c := newPrefixCache(checkpointTestCaches())
			defer c.close()
			s := c.beginScore([]int32{1, 2, 3}, nil)
			checkpointTestFeed(c.caches, s.remaining)
			s.close()
			store, err := openCacheDiskStore(ctx, root, "interrupt")
			if err != nil {
				t.Fatal(err)
			}
			d, err := openCheckpointCache(context.Background(), store, "interrupt", c.caches, make(chan struct{}, 1))
			if err != nil {
				t.Fatal(err)
			}
			defer d.close()
			guard, err := store.lock(ctx)
			if err != nil {
				t.Fatal(err)
			}
			defer guard.Close()
			if err := d.start(c, c.activePath[len(c.activePath)-1], nil); err != nil {
				t.Fatal(err)
			}
			// A peer holds disk admission. Neither maintenance nor a new
			// request may wait for the obsolete checkpoint's I/O.
			if progressed, err := d.step(); progressed || err != nil {
				t.Fatalf("blocked writer progressed=%v, error=%v", progressed, err)
			}
			next := c.beginScore([]int32{10, 11, 12}, nil)
			checkpointTestFeed(c.caches, next.remaining)
			next.close()
			if got := c.caches[0].State()[0].AsType(mlx.DTypeInt32).Ints(); !slices.Equal(got, []int32{10, 11, 12}) {
				t.Fatalf("next request did not replace live state: %v", got)
			}
			if progressed, err := d.step(); progressed || err != nil {
				t.Fatalf("resumed slot waited for disk: progressed=%v, error=%v", progressed, err)
			}
			guard.Close()
			if err := drainCheckpoint(t, d); !errors.Is(err, context.Canceled) {
				t.Fatalf("stale write was not canceled: %v", err)
			}
			files, err := os.ReadDir(store.dir)
			if err != nil {
				t.Fatal(err)
			}
			if len(files) != 1 || files[0].Name() != ".active" {
				t.Fatalf("interrupted checkpoint left files: %v", files)
			}
		})
	})
}

func TestCheckpointDiskChurn(t *testing.T) {
	root := t.TempDir()
	ctx := t.Context()
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.Scoped(func() {
			c := newPrefixCache(checkpointTestCaches())
			defer c.close()
			store, err := openCacheDiskStore(ctx, root, "churn")
			if err != nil {
				t.Fatal(err)
			}
			// Leave room for the metadata reservation and only a few tiny
			// checkpoints, forcing eviction without large test allocations.
			store.limit = mlx.TensorFileMetadataLimit + 2<<20
			d, err := openCheckpointCache(ctx, store, "churn", c.caches, make(chan struct{}, 1))
			if err != nil {
				t.Fatal(err)
			}
			defer d.close()
			var latest []int32
			for i := range 12 {
				tokens := []int32{int32(i + 1), 20, 30}
				s := c.beginScore(tokens, nil)
				checkpointTestFeed(c.caches, s.remaining)
				s.close()
				if err := d.start(c, c.activePath[len(c.activePath)-1], nil); err != nil {
					t.Fatal(err)
				}
				if i%2 == 0 {
					// Resume the slot after at least one tensor has reached disk.
					// A staged write must not publish the new conversation under
					// the previous conversation's keys or strand its reservation.
					deadline := time.After(10 * time.Second)
					for d.job.layer == 0 {
						progressed, err := d.step()
						if err != nil {
							t.Fatal(err)
						}
						if !progressed {
							select {
							case <-d.ready:
							case <-deadline:
								t.Fatal("write did not stage its first tensor")
							}
						}
					}
					next := c.beginScore([]int32{100, 101, 102}, nil)
					checkpointTestFeed(c.caches, next.remaining)
					next.close()
					if err := drainCheckpoint(t, d); !errors.Is(err, context.Canceled) {
						t.Fatalf("resumed write returned %v", err)
					}
				} else {
					if err := drainCheckpoint(t, d); err != nil {
						t.Fatal(err)
					}
					latest = tokens
				}
				entries, err := os.ReadDir(store.dir)
				if err != nil {
					t.Fatal(err)
				}
				var bytes int64
				for _, e := range entries {
					if e.Name() == ".active" {
						continue
					}
					if filepath.Ext(e.Name()) != ".kv" {
						t.Fatalf("unfinished write left %s", e.Name())
					}
					info, err := e.Info()
					if err != nil {
						t.Fatal(err)
					}
					bytes += info.Size()
				}
				if bytes != d.bytes || bytes > store.limit {
					t.Fatalf("disk accounting: files=%d, charged=%d, limit=%d", bytes, d.bytes, store.limit)
				}
			}
			if len(d.saved) == 0 || len(d.saved) >= 6 {
				t.Fatalf("expected bounded retained checkpoints, got %d", len(d.saved))
			}
			c.close()
			p := d.restore(c, latest, nil, len(latest), 0)
			if p == nil {
				t.Fatal("latest completed conversation was lost during churn")
			}
			for done := false; !done; {
				done, err = p.step(ctx)
				if err != nil {
					t.Fatal(err)
				}
			}
			if err := p.commit(); err != nil {
				t.Fatal(err)
			}
			if got := c.caches[0].State()[0].AsType(mlx.DTypeInt32).Ints(); !slices.Equal(got, latest) {
				t.Fatalf("restored %v, want %v", got, latest)
			}
		})
	})
}

func TestCheckpointRequiresStrongMediaIdentity(t *testing.T) {
	digest := persistentMediaDigest([]byte("original"), []int{1, 2})
	r := checkpointRecord{Media: []checkpointMedia{{Position: 1, Length: 2, Digest: digest}}}
	items := []mediaItem{{pos: 1, length: 2, fold: 1 << 31, digest: digest}}
	if !r.matchesMedia(items, 3) {
		t.Fatal("matching media rejected")
	}
	items[0].digest = persistentMediaDigest([]byte("different"), []int{1, 2})
	if r.matchesMedia(items, 3) {
		t.Fatal("short fold collision accepted a different image")
	}
	items[0].digest = persistentMediaDigest([]byte("original"), []int{2, 1})
	if r.matchesMedia(items, 3) || r.matchesMedia(nil, 3) {
		t.Fatal("different or absent media accepted")
	}
}

func TestCheckpointPartialRestoreFailure(t *testing.T) {
	for _, failure := range []string{"cancel", "truncate", "corrupt"} {
		t.Run(failure, func(t *testing.T) {
			root := t.TempDir()
			ctx := t.Context()
			mlxtest.Run(t, func(t *mlxtest.T) {
				mlx.Scoped(func() {
					c := newPrefixCache(checkpointTestCaches())
					defer c.close()
					tokens := []int32{1, 2, 3, 4, 5, 6, 7}
					s := c.beginScore(tokens, nil)
					checkpointTestFeed(c.caches, s.remaining)
					s.close()
					store, err := openCacheDiskStore(ctx, root, "failure")
					if err != nil {
						t.Fatal(err)
					}
					d, err := openCheckpointCache(context.Background(), store, "failure", c.caches, make(chan struct{}, 1))
					if err != nil {
						t.Fatal(err)
					}
					defer d.close()
					if err := d.start(c, c.activePath[len(c.activePath)-1], nil); err != nil {
						t.Fatal(err)
					}
					if err := drainCheckpoint(t, d); err != nil {
						t.Fatal(err)
					}
					saved := d.saved[0]
					c.close()
					p := d.restore(c, tokens, nil, len(tokens), 0)
					if p == nil {
						t.Fatal("checkpoint not selected")
					}
					if done, err := p.step(context.Background()); done || err != nil || c.caches[0].Offset() != len(tokens) {
						t.Fatalf("first layer did not restore: done=%v, err=%v", done, err)
					}
					ctx, cancel := context.WithCancel(context.Background())
					defer cancel()
					switch failure {
					case "cancel":
						cancel()
					case "truncate":
						if err := os.Truncate(saved.file.Path(), 1); err != nil {
							t.Fatal(err)
						}
					case "corrupt":
						// Damage the next layer's payload without changing metadata.
						var extent struct{ Offset int64 }
						data, err := json.Marshal(saved.record.Layers[1].Tensors[0])
						if err != nil {
							t.Fatal(err)
						}
						if err := json.Unmarshal(data, &extent); err != nil {
							t.Fatal(err)
						}
						f, err := os.OpenFile(saved.file.Path(), os.O_WRONLY, 0)
						if err != nil {
							t.Fatal(err)
						}
						_, err = f.WriteAt([]byte{0xff}, extent.Offset)
						f.Close()
						if err != nil {
							t.Fatal(err)
						}
					}
					if done, err := p.step(ctx); !done || err == nil {
						t.Fatalf("failed restore continued: done=%v, err=%v", done, err)
					}
					if saved.pins != 0 {
						t.Fatal("failed restore retained a file pin")
					}
					for i, layer := range c.caches {
						if layer != nil && layer.Offset() != 0 {
							t.Fatalf("partial state survived in layer %d", i)
						}
					}
					if (len(d.saved) == 1) != (failure == "cancel") {
						t.Fatalf("wrong recovery after %s: %d checkpoints", failure, len(d.saved))
					}
					retry := c.beginScore(tokens, nil)
					if len(retry.remaining) != len(tokens) {
						t.Fatal("partial restore was counted as a cache hit")
					}
					checkpointTestFeed(c.caches, retry.remaining)
					retry.close()
					if c.minCacheOffset() != len(tokens) {
						t.Fatal("recomputation failed after damaged checkpoint")
					}
				})
			})
		})
	}
}

func TestCheckpointAtomicMediaBoundary(t *testing.T) {
	c := newPrefixCache(nil)
	tokens := []int32{1, 0, 0, 0, 2}
	items := []mediaItem{{pos: 1, length: 3, fold: 1 << 31, digest: persistentMediaDigest([]byte("image"), nil), item: &model.PreparedItem{}}}
	keys := c.key(effectiveKeyTokens(tokens, items))
	for _, end := range []int{1, 2, 4} {
		saved := &savedCheckpoint{record: checkpointRecord{Keys: keys[:end]}}
		if end > 1 {
			saved.record.Media = []checkpointMedia{{Position: 1, Length: 3, Digest: items[0].digest}}
		}
		d := &checkpointCache{saved: []*savedCheckpoint{saved}}
		p := d.restore(c, tokens, items, len(tokens), 0)
		if (p != nil) != (end != 2) {
			t.Fatalf("atomic media restore at %d: %v", end, p != nil)
		}
		if p != nil {
			p.close(false)
		}
	}
}
