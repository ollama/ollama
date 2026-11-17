package mlxrunner

import (
	"slices"
	"testing"
	"time"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
)

type recordedSnapshotRetention struct {
	caches, idle       []*prefixCache
	workspace, working int64
	admitting          bool
	calls              int
	changes            chan struct{}
}

func (r *recordedSnapshotRetention) trim(caches, idle []*prefixCache, workspace, working int64, admitting bool) {
	r.caches, r.idle = caches, idle
	r.workspace, r.working, r.admitting = workspace, working, admitting
	r.calls++
}

func (r *recordedSnapshotRetention) Changes() <-chan struct{} { return r.changes }
func (r *recordedSnapshotRetention) Close()                   {}

func TestSnapshotRetentionScheduling(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		active, idle := nativeRetentionHistory(), nativeRetentionHistory()
		defer active.close()
		defer idle.close()
		policy := &recordedSnapshotRetention{changes: make(chan struct{}, 1)}
		s := &batchScheduler{
			r: &Runner{parallel: 2}, retention: policy, retentionInitialized: true,
			snapshotWorkspace: 123,
			slots: []*batchSlot{
				{cache: active, request: &batchRequest{}},
				{cache: idle},
			},
		}
		s.trimSnapshots(true)
		if policy.calls != 1 || !policy.admitting || policy.workspace != 123 {
			t.Fatalf("admission did not pass workspace to the policy: %+v", policy)
		}
		if !slices.Equal(policy.caches, []*prefixCache{active, idle}) || !slices.Equal(policy.idle, []*prefixCache{idle}) {
			t.Fatal("policy did not receive the active/idle cache inventory")
		}
		if want := int64(active.liveBytes() + idle.liveBytes()); policy.working != want || want == 0 {
			t.Fatalf("copy reserve = %d, want %d", policy.working, want)
		}
		if s.snapshotChanges != policy.changes {
			t.Fatal("scheduler did not subscribe to the initialized policy")
		}
		// A future check avoids relying on how quickly the test machine runs.
		s.snapshotCheck = time.Now().Add(time.Hour)
		s.trimSnapshots(false)
		if policy.calls != 1 {
			t.Fatal("periodic maintenance bypassed its interval")
		}
		s.trimSnapshots(true)
		if policy.calls != 2 {
			t.Fatal("urgent maintenance was throttled")
		}
		s.snapshotCheck = time.Now().Add(-time.Hour)
		s.trimSnapshots(false)
		if policy.calls != 3 || policy.admitting {
			t.Fatal("due periodic maintenance did not run")
		}
	})
}

func TestCachelessSchedulerDoesNotInitializeRetention(t *testing.T) {
	s := &batchScheduler{r: &Runner{}, slots: []*batchSlot{{}}}
	s.trimSnapshots(true)
	s.trimSnapshots(false)
	if s.retentionInitialized || s.retention != nil || s.snapshotChanges != nil || !s.snapshotCheck.IsZero() {
		t.Fatal("cacheless scheduler initialized retention")
	}
}

func TestMetalRetentionInitializesLazily(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		if !mlx.MetalIsAvailable() {
			t.Skip("requires Metal")
		}
		policy, err := newSnapshotRetention(2)
		if err != nil {
			t.Fatal(err)
		}
		defer policy.Close()
		metal, ok := policy.(*metalRetention)
		if !ok || metal.memory != nil || policy.Changes() != nil {
			t.Fatal("selecting Metal consumed a memory observation")
		}
		c := nativeRetentionHistory()
		defer c.close()
		policy.trim([]*prefixCache{c}, nil, 0, 0, true)
		if metal.memory == nil || policy.Changes() != nil {
			t.Fatal("Metal did not initialize its periodic monitor")
		}
	})
}
