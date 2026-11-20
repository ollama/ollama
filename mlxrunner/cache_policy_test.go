package mlxrunner

import (
	"context"
	"testing"
	"time"

	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
)

func TestCachePolicy(t *testing.T) {
	for _, tt := range []struct {
		name, policy, size string
		age                time.Duration
		limit              int64
		disabled, invalid  bool
	}{
		{name: "default", age: checkpointIdleAge},
		{name: "balanced", policy: "balanced", age: checkpointIdleAge},
		{name: "lean", policy: "lean"},
		{name: "warm", policy: "warm", age: -1},
		{name: "limit", size: "123456", age: checkpointIdleAge, limit: 123456},
		{name: "disabled", size: "0", disabled: true},
		{name: "negative", size: "-1", invalid: true},
		{name: "invalid size", size: "lots", invalid: true},
		{name: "overflow", size: "9223372036854775808", invalid: true},
		{name: "invalid policy", policy: "other", invalid: true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			t.Setenv("OLLAMA_KV_CACHE_DIR", dir)
			t.Setenv("OLLAMA_KV_CACHE_POLICY", tt.policy)
			t.Setenv("OLLAMA_KV_CACHE_SIZE", tt.size)
			p, enabled, err := loadCachePolicy()
			if (err != nil) != tt.invalid || enabled != (!tt.disabled && !tt.invalid) {
				t.Fatalf("enabled=%v, error=%v", enabled, err)
			}
			if enabled && (p.dir != dir || p.idleAge != tt.age || p.diskLimit != tt.limit) {
				t.Fatalf("unexpected policy: %+v", p)
			}
		})
	}
}

func TestCheckpointRetentionPolicies(t *testing.T) {
	for _, policy := range []string{"balanced", "lean", "warm"} {
		t.Run(policy, func(t *testing.T) {
			t.Setenv("OLLAMA_KV_CACHE_DIR", t.TempDir())
			t.Setenv("OLLAMA_KV_CACHE_POLICY", policy)
			t.Setenv("OLLAMA_KV_CACHE_SIZE", "16777216")
			ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
			defer cancel()
			mlxtest.Run(t, func(t *mlxtest.T) {
				mlx.Scoped(func() {
					c := newPrefixCache(checkpointTestCaches())
					defer c.close()
					s := c.beginScore([]int32{1, 2, 3}, nil)
					checkpointTestFeed(c.caches, s.remaining)
					s.close()
					r := &Runner{checkpoints: newRunnerCheckpoints(&manifest.Manifest{})}
					if r.checkpoints == nil {
						t.Fatal("policy disabled persistence")
					}
					defer r.checkpoints.close()
					r.checkpoints.completed(s)
					d := r.checkpoints.open(c)
					if d == nil || d.store.limit != 16777216 || d.store.reserve < cacheMinimumFree {
						t.Fatal("configured capacity or independent free-space reserve was lost")
					}
					if err := r.flushCheckpoints(ctx); err != nil {
						t.Fatal(err)
					}
					mlx.Scoped(func() { maintainTestCheckpoints(r.checkpoints) })
					if got := c.minCacheOffset(); (got == 0) != (policy == "lean") {
						t.Fatalf("policy %s retained %d tokens after saving", policy, got)
					}
					if policy != "lean" {
						r.checkpoints.pending[c].used = time.Now().Add(-2 * checkpointIdleAge)
						mlx.Scoped(func() { maintainTestCheckpoints(r.checkpoints) })
						if got := c.minCacheOffset(); (got == 0) != (policy == "balanced") {
							t.Fatalf("policy %s retained %d aged tokens", policy, got)
						}
					}
					if policy == "warm" && r.checkpoints.reclaimSaved([]*prefixCache{c}, []*prefixCache{c}, nil, 1) <= 0 {
						t.Fatal("warm policy prevented pressure reclamation")
					}
					if c.minCacheOffset() != 0 || len(d.saved) != 1 {
						t.Fatal("reclamation lost the disk checkpoint or retained idle RAM")
					}
				})
			})
		})
	}
}
