package mlxrunner

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
	"github.com/ollama/ollama/mlxrunner/cache"
)

func evictionTestCache(old time.Time) *prefixCache {
	c := newPrefixCache(nil)
	c.ensureRoot()
	cold := c.root.appendChild([]trieKey{1, 2}, 2)
	cold.lastUsed = old
	frontier := c.root.appendChild([]trieKey{3, 4}, 2)
	c.activePath = []*trieNode{c.root, frontier}
	return c
}

func TestCacheEvictionReplayIgnoresLocalOrder(t *testing.T) {
	leader := []*prefixCache{evictionTestCache(time.Unix(1, 0)), nil, evictionTestCache(time.Unix(2, 0))}
	follower := []*prefixCache{evictionTestCache(time.Unix(2, 0)), nil, evictionTestCache(time.Unix(1, 0))}
	victim, _, ok := chooseColdHistory(leader, nil, nil)
	if !ok || victim.Slot != 0 || !slices.Equal(victim.Path, []int{0}) {
		t.Fatalf("selected %+v, want first slot's cold edge", victim)
	}
	other, _, ok := chooseColdHistory(follower, nil, nil)
	if !ok || other.Slot == victim.Slot {
		t.Fatal("fixture did not create different local eviction choices")
	}
	if len(leader[0].root.children) != 2 {
		t.Fatal("selection evicted history before application")
	}
	for _, slots := range [][]*prefixCache{leader, follower} {
		applyCacheEviction(slots, victim)
		if len(slots[0].root.children) != 1 || len(slots[2].root.children) != 2 {
			t.Fatal("replay chose a different slot or node")
		}
	}
}

func TestCacheEvictionKeepsSparseSlotIndex(t *testing.T) {
	c := evictionTestCache(time.Time{})
	slots := []*prefixCache{nil, nil, c, nil}
	victim, _, ok := chooseColdHistory(slots, nil, nil)
	if !ok || victim.Slot != 2 {
		t.Fatalf("single-cache selection changed its scheduler index: %+v", victim)
	}
	applyCacheEviction(slots, victim)
	if len(c.root.children) != 1 {
		t.Fatal("sparse victim was not applied to its original slot")
	}
}

func TestIdleReleaseDecisionDoesNotRecheckDisk(t *testing.T) {
	c := evictionTestCache(time.Time{})
	file := filepath.Join(t.TempDir(), "saved.kv")
	if err := os.WriteFile(file, nil, 0o600); err != nil {
		t.Fatal(err)
	}
	p := testRunnerCheckpoints(t.TempDir())
	p.idleAge = 0
	p.pending[c] = &pendingCheckpoint{cache: c, epoch: c.epoch, saved: file, store: &checkpointCache{}}
	slots := []*prefixCache{nil, c}
	selected := p.maintain(slots)
	if !slices.Equal(selected, []int{1}) || c.root == nil {
		t.Fatalf("selection = %v, resident root = %p", selected, c.root)
	}
	if err := os.Remove(file); err != nil {
		t.Fatal(err)
	}
	// Followers may have different local I/O state. Only the deciding rank
	// checks availability; replay must still release the chosen slot.
	p.release(slots, selected)
	if c.root != nil || p.pending[c] != nil {
		t.Fatal("application rechecked disk and retained the selected slot")
	}
}

func TestSavedVictimsPreferLessReusableHistory(t *testing.T) {
	for _, shorter := range []int{0, 4, 12} {
		t.Run(fmt.Sprintf("other_match=%d", shorter), func(t *testing.T) {
			incoming := evictionTestCache(time.Unix(1, 0))
			other := evictionTestCache(time.Unix(2, 0))
			file := filepath.Join(t.TempDir(), "saved.kv")
			if err := os.WriteFile(file, nil, 0o600); err != nil {
				t.Fatal(err)
			}
			p := testRunnerCheckpoints(t.TempDir())
			p.pending[incoming] = &pendingCheckpoint{cache: incoming, epoch: incoming.epoch, saved: file}
			preferred := make(map[*trieNode]int)
			for _, item := range []struct {
				cache *prefixCache
				match int
			}{{incoming, 12}, {other, shorter}} {
				walkNodes(item.cache.root, func(n *trieNode) bool { preferred[n] = item.match; return true })
			}
			caches := []*prefixCache{incoming, nil, other}
			victims := p.savedVictims(caches, []*prefixCache{incoming, other}, preferred)
			if shorter < 12 && len(victims) != 0 {
				t.Fatal("disk availability overrode a more reusable resident prefix")
			}
			if shorter == 12 && !slices.Equal(victims, []int{0}) {
				t.Fatal("equally useful history did not prefer the already saved row")
			}
			if incoming.root == nil || len(other.root.children) != 2 {
				t.Fatal("victim selection mutated resident state")
			}
		})
	}
}

func TestCacheLayoutTagSeparatesRanks(t *testing.T) {
	seen := make(map[string]bool)
	for _, layout := range []cacheLayout{"", "layer/0/2/layers0-15", "layer/1/2/layers16-31", "layer/0/4/layers0-7", "tensor/0/2/columns0-511", "layer/0/2/layers16-31"} {
		var b strings.Builder
		layout.writeTo(&b)
		if layout == "" && b.Len() != 0 {
			t.Fatal("unsplit identity changed")
		}
		if seen[b.String()] {
			t.Fatalf("layout collision for %+v", layout)
		}
		seen[b.String()] = true
	}
}

func TestCheckpointRestoreAllLayersAndReadyState(t *testing.T) {
	root := t.TempDir()
	ctx := t.Context()
	mlxtest.Run(t, func(t *mlxtest.T) {
		mlx.Scoped(func() {
			c := newPrefixCache(checkpointTestCaches())
			defer c.close()
			session := c.beginScore([]int32{1, 2, 3}, nil)
			checkpointTestFeed(c.caches, session.remaining)
			session.close()
			store, err := openCacheDiskStore(ctx, root, "restore-steps")
			if err != nil {
				t.Fatal(err)
			}
			d, err := openCheckpointCache(ctx, store, "restore-steps", c.caches, make(chan struct{}, 1))
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
			for _, all := range []bool{false, true} {
				c.close()
				resident := c.beginScore([]int32{1}, nil)
				checkpointTestFeed(c.caches, resident.remaining)
				resident.close()
				p := d.restore(c, []int32{1, 2, 3, 4}, nil, 3, 2)
				if p == nil {
					t.Fatal("saved checkpoint unavailable")
				}
				scheduler := &batchScheduler{r: &Runner{restoreAllLayers: all}}
				slot := &batchSlot{cache: c, request: &batchRequest{request: request{Ctx: ctx, Tokens: []int32{1, 2, 3, 4}}, restore: p}}
				scheduler.slots = []*batchSlot{slot}
				event, done := scheduler.restoreProgress(slot)
				if done != all || !event.Restored {
					t.Fatalf("all=%v: done=%v, event=%+v", all, done, event)
				}
				if !done {
					if _, err := p.stepAll(ctx); err != nil {
						t.Fatal(err)
					}
				}
				if ready, err := p.step(context.Background()); !ready || err != nil {
					t.Fatalf("waiting for shared completion changed ready state: %v %v", ready, err)
				}
				if got := c.caches[0].State()[0].AsType(mlx.DTypeInt32).Ints(); !slices.Equal(got, []int32{1, 2, 3}) {
					t.Fatalf("all=%v: restored %v", all, got)
				}
				_, matched := findBestMatch(c.root, c.key([]uint32{1, 2, 3, 4}))
				if matched != 1 || p.saved.pins != 1 {
					t.Fatalf("local readiness published or unpinned checkpoint: matched=%d, pins=%d", matched, p.saved.pins)
				}
				if err := scheduler.finishRestore(checkpointRestoreFinished{Slot: 0, Restored: !all}); err != nil {
					t.Fatal(err)
				}
				want := 3
				if all {
					want = 1 // this rank succeeded, but a peer rejected completion; keep original resident prefix
				}
				if got := slot.request.prefill.position; got != want {
					t.Fatalf("shared restored=%v: prefill starts at %d, want %d", !all, got, want)
				}
				if d.saved[0].pins != 0 {
					t.Fatal("completion retained the checkpoint pin")
				}
			}
		})
	})
}

func TestCheckpointRestoreSelectionDoesNotStart(t *testing.T) {
	c := newPrefixCache([]cache.Cache{nil})
	d := &checkpointCache{saved: []*savedCheckpoint{{record: checkpointRecord{Keys: []trieKey{1, 2, 3}}}}}
	saved := d.findRestore(c, []int32{1, 2, 3, 4}, nil, 3, 0)
	if saved == nil || saved.pins != 0 || c.root != nil {
		t.Fatal("selection changed cache ownership or state")
	}
	if p := d.restore(c, []int32{1, 2, 3, 4}, nil, 2, 1); p != nil {
		t.Fatal("missing exact checkpoint silently chose another prefix")
	}
}

func TestCheckpointCompletionRequiresLocalReadiness(t *testing.T) {
	for _, restore := range []*checkpointRestore{nil, {}, {done: true, err: context.Canceled}} {
		s := &batchScheduler{slots: []*batchSlot{{request: &batchRequest{restore: restore}}}}
		if err := s.finishRestore(checkpointRestoreFinished{Slot: 0, Restored: true}); err == nil {
			t.Fatal("shared completion accepted an unavailable local shard")
		}
		if s.slots[0].request.prefill != nil {
			t.Fatal("invalid completion started a forward")
		}
	}
}

func TestCheckpointLayoutSeparatesStores(t *testing.T) {
	root := t.TempDir()
	mlxtest.Run(t, func(t *mlxtest.T) {
		p := testRunnerCheckpoints(root)
		defer p.close()
		c := newPrefixCache([]cache.Cache{cache.NewKVCache()})
		defer c.close()
		seen := make(map[string]bool)
		var unsplit *checkpointCache
		for _, layout := range []cacheLayout{"", "layer/0/2/layers0-15", "layer/0/2/layers16-31"} {
			p.layout = layout
			d := p.open(c)
			if d == nil || seen[d.store.dir] {
				t.Fatalf("layout %q did not get an independent store", layout)
			}
			seen[d.store.dir] = true
			if layout == "" {
				unsplit = d
			}
		}
		p.layout = ""
		if p.open(c) != unsplit {
			t.Fatal("unsplit store key was not stable")
		}
	})
}
