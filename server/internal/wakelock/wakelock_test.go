package wakelock

import (
	"sync"
	"sync/atomic"
	"testing"
)

// fakeAssertion is a test double that counts acquire/release calls and
// guards against double release.
type fakeAssertion struct {
	released atomic.Bool
	releases atomic.Int32
}

func (f *fakeAssertion) release() {
	f.released.Store(true)
	f.releases.Add(1)
}

func TestAcquireReleaseHoldsAssertionOnlyWhenPositive(t *testing.T) {
	prev := newAssertion
	var acquires atomic.Int32
	var current *fakeAssertion
	newAssertion = func(reason string) (assertion, error) {
		acquires.Add(1)
		current = &fakeAssertion{}
		return current, nil
	}
	t.Cleanup(func() { newAssertion = prev })

	w := New("test")
	if w.Held() {
		t.Fatal("expected wake lock to be released initially")
	}

	w.Acquire()
	if got := w.Count(); got != 1 {
		t.Fatalf("count = %d, want 1", got)
	}
	if !w.Held() {
		t.Fatal("expected wake lock to be held after first acquire")
	}
	if got := acquires.Load(); got != 1 {
		t.Fatalf("acquires = %d, want 1", got)
	}

	w.Acquire()
	if got := w.Count(); got != 2 {
		t.Fatalf("count = %d, want 2", got)
	}
	if got := acquires.Load(); got != 1 {
		t.Fatalf("acquires = %d, want 1 (re-entry should not take a new assertion)", got)
	}

	w.Release()
	if got := w.Count(); got != 1 {
		t.Fatalf("count = %d, want 1", got)
	}
	if current.released.Load() {
		t.Fatal("assertion released too early")
	}
	if !w.Held() {
		t.Fatal("expected wake lock still held while count > 0")
	}

	w.Release()
	if got := w.Count(); got != 0 {
		t.Fatalf("count = %d, want 0", got)
	}
	if !current.released.Load() {
		t.Fatal("expected assertion released when count reached zero")
	}
	if w.Held() {
		t.Fatal("expected wake lock released")
	}
}

func TestReleaseBelowZeroIsNoop(t *testing.T) {
	prev := newAssertion
	newAssertion = func(reason string) (assertion, error) {
		return &fakeAssertion{}, nil
	}
	t.Cleanup(func() { newAssertion = prev })

	w := New("test")
	w.Release()
	w.Release()
	if got := w.Count(); got != 0 {
		t.Fatalf("count = %d, want 0", got)
	}
	if w.Held() {
		t.Fatal("expected wake lock released")
	}
}

func TestCloseReleasesAndPreventsAcquire(t *testing.T) {
	prev := newAssertion
	var current *fakeAssertion
	newAssertion = func(reason string) (assertion, error) {
		current = &fakeAssertion{}
		return current, nil
	}
	t.Cleanup(func() { newAssertion = prev })

	w := New("test")
	w.Acquire()
	w.Acquire()
	w.Close()
	if !current.released.Load() {
		t.Fatal("expected assertion released on Close")
	}
	if w.Held() {
		t.Fatal("expected wake lock not held after Close")
	}

	// After close, Acquire should be a no-op.
	w.Acquire()
	if w.Held() {
		t.Fatal("expected Acquire after Close to be a no-op")
	}
	if got := w.Count(); got != 0 {
		t.Fatalf("count after Acquire-post-Close = %d, want 0", got)
	}

	// Close is idempotent.
	w.Close()
	if got := current.releases.Load(); got != 1 {
		t.Fatalf("releases = %d, want 1 (Close should not double-release)", got)
	}
}

func TestConcurrentAcquireRelease(t *testing.T) {
	prev := newAssertion
	var acquires atomic.Int32
	var releases atomic.Int32
	newAssertion = func(reason string) (assertion, error) {
		acquires.Add(1)
		return &fakeAssertionCounted{releases: &releases}, nil
	}
	t.Cleanup(func() { newAssertion = prev })

	w := New("test")
	const workers = 64
	const iterations = 1000
	var wg sync.WaitGroup
	for range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range iterations {
				w.Acquire()
				w.Release()
			}
		}()
	}
	wg.Wait()

	if got := w.Count(); got != 0 {
		t.Fatalf("count = %d, want 0", got)
	}
	if w.Held() {
		t.Fatal("expected wake lock released after all workers finished")
	}
	// Whatever the interleaving was, every acquire must be paired with a release.
	if acquires.Load() != releases.Load() {
		t.Fatalf("acquires = %d, releases = %d (expected equal)", acquires.Load(), releases.Load())
	}
}

func TestAssertionFailureDoesNotPanic(t *testing.T) {
	prev := newAssertion
	newAssertion = func(reason string) (assertion, error) {
		return nil, errFake
	}
	t.Cleanup(func() { newAssertion = prev })

	w := New("test")
	w.Acquire()
	if w.Held() {
		t.Fatal("expected wake lock not held when underlying assertion fails")
	}
	if got := w.Count(); got != 1 {
		t.Fatalf("count = %d, want 1 (refcount still tracked even when OS assertion failed)", got)
	}
	// Release path must not blow up when no assertion was taken.
	w.Release()
	if got := w.Count(); got != 0 {
		t.Fatalf("count = %d, want 0", got)
	}
}

type fakeAssertionCounted struct {
	releases *atomic.Int32
}

func (f *fakeAssertionCounted) release() { f.releases.Add(1) }

type fakeErr struct{}

func (fakeErr) Error() string { return "fake" }

var errFake = fakeErr{}
