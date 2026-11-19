package mlxrunner

import (
	"context"
	"math"
	"testing"
	"time"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
)

func TestSnapshotRetentionScheduling(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		if !mlx.MetalIsAvailable() {
			t.Skip("adaptive host-memory retention requires Metal")
		}
		c := nativeRetentionHistory()
		defer c.close()
		s := &batchScheduler{r: &Runner{parallel: 2}, slots: []*batchSlot{{cache: c}}}
		s.trimSnapshots(nil, -1)
		if s.memory == nil || s.snapshotTimer == nil || s.r.snapshotLimit() != math.MaxInt64 {
			t.Fatal("first observation did not establish maintenance and the new-slot allowance")
		}
		defer s.memory.Close()
		defer s.snapshotTimer.Stop()
		monitor, timer := s.memory, s.snapshotTimer
		for range 3 {
			if got := s.r.snapshotLimit(); got != math.MaxInt64 {
				t.Fatalf("new slot allowance = %d", got)
			}
		}
		// Expire the actual timer without sleeping for the observation cadence.
		timer.Reset(0)
		select {
		case <-timer.C:
		case <-time.After(5 * time.Second):
			t.Fatal("maintenance timer did not wake the scheduler")
		}
		s.trimSnapshots(nil, -1)
		if s.memory != monitor || s.snapshotTimer != timer {
			t.Fatal("maintenance replaced the monitor or timer")
		}
	})
}

func TestCachelessSchedulerDoesNotInitializeRetention(t *testing.T) {
	s := &batchScheduler{r: &Runner{}, slots: []*batchSlot{{}}}
	s.trimSnapshots(nil, -1)
	s.trimSnapshots(request{}, 0)
	if s.memory != nil || s.snapshotTimer != nil {
		t.Fatal("cacheless scheduler initialized retention")
	}
}

func TestSnapshotLimitBeforeObservation(t *testing.T) {
	for _, parallel := range []int{0, 1, 4} {
		r := &Runner{parallel: parallel}
		if got, want := r.snapshotLimit(), maxPagedOutBytes/int64(max(1, parallel)); got != want {
			t.Fatalf("parallel %d: initial allowance = %d, want %d", parallel, got, want)
		}
	}
}

func TestCancelledAdmissionDoesNotTrimHistory(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		c := nativeRetentionHistory()
		defer c.close()
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		s := &batchScheduler{r: &Runner{}, slots: []*batchSlot{{cache: c}}}
		before := c.pagedOutBytes
		for _, req := range []queuedRequest{
			request{Ctx: ctx, Responses: make(chan CompletionResponse, 1)},
			&evaluationRequest{ctx: ctx, done: make(chan error, 1)},
			&speechRequest{ctx: ctx, done: make(chan error, 1)},
		} {
			if s.admit(req) {
				t.Fatalf("cancelled %T suppressed periodic maintenance", req)
			}
			if s.memory != nil || s.snapshotTimer != nil || c.pagedOutBytes != before || s.slots[0].busy() {
				t.Fatalf("cancelled %T touched cache retention or acquired a slot", req)
			}
		}
		s.r.Requests = make(chan queuedRequest, 1)
		s.r.Requests <- request{Ctx: ctx, Responses: make(chan CompletionResponse, 1)}
		if s.admitPending() {
			t.Fatal("draining a cancelled request suppressed periodic maintenance")
		}
	})
}
