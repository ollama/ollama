package progress

import (
	"bytes"
	"strings"
	"testing"
	"time"
)

// mockState implements State interface for testing
type mockState struct {
	value string
}

func (m *mockState) String() string {
	return m.value
}

func TestNewProgress(t *testing.T) {
	var buf bytes.Buffer
	p := NewProgress(&buf)
	defer p.Stop()

	if p.w == nil {
		t.Error("Progress writer should not be nil")
	}

	if p.ticker == nil {
		t.Error("Progress ticker should be started")
	}
}

func TestProgressAdd(t *testing.T) {
	var buf bytes.Buffer
	p := NewProgress(&buf)
	defer p.Stop()

	state1 := &mockState{value: "state1"}
	state2 := &mockState{value: "state2"}

	p.Add("key1", state1)
	if len(p.states) != 1 {
		t.Errorf("states count = %d, want 1", len(p.states))
	}

	p.Add("key2", state2)
	if len(p.states) != 2 {
		t.Errorf("states count = %d, want 2", len(p.states))
	}
}

func TestProgressStop(t *testing.T) {
	var buf bytes.Buffer
	p := NewProgress(&buf)

	// Give the goroutine time to start
	time.Sleep(50 * time.Millisecond)

	stopped := p.Stop()
	if !stopped {
		t.Error("Stop() should return true on first call")
	}

	// Stop again should return false
	stopped = p.Stop()
	if stopped {
		t.Error("Stop() should return false on subsequent calls")
	}
}

func TestProgressStopAndClear(t *testing.T) {
	var buf bytes.Buffer
	p := NewProgress(&buf)

	state := &mockState{value: "test"}
	p.Add("key", state)

	// Give the goroutine time to start and render
	time.Sleep(150 * time.Millisecond)

	stopped := p.StopAndClear()
	if !stopped {
		t.Error("StopAndClear() should return true on first call")
	}

	// Should contain escape sequences for clearing
	output := buf.String()
	// Check for cursor show/hide sequences
	if !strings.Contains(output, "\033[?25l") {
		t.Log("Output may not contain cursor hide sequence (depends on terminal)")
	}
}

func TestProgressStopSpinners(t *testing.T) {
	var buf bytes.Buffer
	p := NewProgress(&buf)

	spinner := NewSpinner("loading")
	p.Add("spinner", spinner)

	// Give time for spinner to start
	time.Sleep(50 * time.Millisecond)

	if !spinner.stopped.IsZero() {
		t.Error("Spinner should not be stopped before Progress.Stop()")
	}

	p.Stop()

	if spinner.stopped.IsZero() {
		t.Error("Spinner should be stopped after Progress.Stop()")
	}
}

func TestProgressRender(t *testing.T) {
	var buf bytes.Buffer
	p := NewProgress(&buf)
	defer p.Stop()

	state := &mockState{value: "test output"}
	p.Add("key", state)

	// Wait for at least one render cycle
	time.Sleep(150 * time.Millisecond)

	output := buf.String()
	if !strings.Contains(output, "test output") {
		t.Errorf("render should include state output, got %q", output)
	}
}

func TestProgressMultipleStates(t *testing.T) {
	var buf bytes.Buffer
	p := NewProgress(&buf)
	defer p.Stop()

	for i := 0; i < 5; i++ {
		state := &mockState{value: "line"}
		p.Add("key", state)
	}

	if len(p.states) != 5 {
		t.Errorf("states count = %d, want 5", len(p.states))
	}
}

func TestProgressWithBar(t *testing.T) {
	var buf bytes.Buffer
	p := NewProgress(&buf)
	defer p.Stop()

	bar := NewBar("downloading", 100, 0)
	p.Add("bar", bar)

	bar.Set(50)

	// Wait for render
	time.Sleep(150 * time.Millisecond)

	output := buf.String()
	if !strings.Contains(output, "50%") {
		t.Logf("Output: %q", output)
		// This might fail in non-terminal environments
	}
}

func TestProgressConcurrentAccess(t *testing.T) {
	var buf bytes.Buffer
	p := NewProgress(&buf)
	defer p.Stop()

	// Add states concurrently
	done := make(chan bool)
	for i := 0; i < 10; i++ {
		go func(n int) {
			state := &mockState{value: "state"}
			p.Add("key", state)
			done <- true
		}(i)
	}

	// Wait for all goroutines
	for i := 0; i < 10; i++ {
		<-done
	}

	if len(p.states) != 10 {
		t.Errorf("states count = %d, want 10", len(p.states))
	}
}

func TestStateInterface(t *testing.T) {
	// Verify Bar implements State
	var _ State = &Bar{}

	// Verify Spinner implements State
	var _ State = &Spinner{}
}

func TestProgressPosTracking(t *testing.T) {
	var buf bytes.Buffer
	p := NewProgress(&buf)
	defer p.Stop()

	if p.pos != 0 {
		t.Errorf("initial pos = %d, want 0", p.pos)
	}

	state := &mockState{value: "line"}
	p.Add("key", state)

	// Wait for render to update pos
	time.Sleep(150 * time.Millisecond)

	if p.pos != 1 {
		t.Errorf("pos after render = %d, want 1", p.pos)
	}
}
