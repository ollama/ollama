package server

import (
	"os/exec"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
)

// The refcount core must share one OS assertion across concurrent holders and
// release exactly once, even under concurrency or double release.
func TestInferencePowerAssertionRefcount(t *testing.T) {
	var acquires atomic.Int32
	var releases atomic.Int32

	old := platformAcquirePowerAssertion
	platformAcquirePowerAssertion = func() (func(), bool) {
		acquires.Add(1)
		return func() { releases.Add(1) }, true
	}
	defer func() { platformAcquirePowerAssertion = old }()

	var p inferencePowerAssertion

	const n = 16
	rels := make([]func(), n)
	var wg sync.WaitGroup
	for i := range n {
		wg.Add(1)
		go func() {
			defer wg.Done()
			rels[i] = p.acquire()
		}()
	}
	wg.Wait()

	if got := acquires.Load(); got != 1 {
		t.Fatalf("platform acquire called %d times, want 1", got)
	}

	for _, r := range rels {
		r()
	}
	if got := releases.Load(); got != 1 {
		t.Fatalf("platform release called %d times, want 1", got)
	}

	// Double release must be harmless.
	rels[0]()
	if got := releases.Load(); got != 1 {
		t.Fatalf("double release called platform release again (%d)", got)
	}

	// A new cycle acquires again.
	r := p.acquire()
	r()
	if got := acquires.Load(); got != 2 {
		t.Fatalf("second cycle acquire called %d times total, want 2", got)
	}
	if got := releases.Load(); got != 2 {
		t.Fatalf("second cycle release called %d times total, want 2", got)
	}
}

// A failed platform acquire must not break the acquire/release contract.
func TestInferencePowerAssertionAcquireFailure(t *testing.T) {
	old := platformAcquirePowerAssertion
	platformAcquirePowerAssertion = func() (func(), bool) { return nil, false }
	defer func() { platformAcquirePowerAssertion = old }()

	var p inferencePowerAssertion
	r1 := p.acquire()
	r2 := p.acquire()
	r1()
	r2() // must not panic or call anything
}

// On macOS, holding the shared assertion must surface a real IOPMAssertion
// visible to `pmset -g assertions`. Skipped where pmset is unavailable.
func TestInferencePowerAssertionVisibleInPmset(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("darwin-only integration check")
	}
	if _, err := exec.LookPath("pmset"); err != nil {
		t.Skip("pmset not available")
	}

	release := sharedInferencePower.acquire()
	defer release()

	out, err := exec.Command("pmset", "-g", "assertions").Output()
	if err != nil {
		t.Fatalf("pmset failed: %v", err)
	}
	if !strings.Contains(string(out), "ollama inference") {
		t.Fatalf("assertion not visible in pmset output:\n%s", out)
	}
}
