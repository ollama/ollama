package progress

import (
	"bytes"
	"sync"
	"testing"
)

type progressTestState string

func (s progressTestState) String() string {
	return string(s)
}

func TestProgressStopIsIdempotent(t *testing.T) {
	var buf bytes.Buffer
	p := NewProgress(&buf)
	p.Add("one", progressTestState("working"))

	if !p.Stop() {
		t.Fatal("first stop should report stopped")
	}
	if p.Stop() {
		t.Fatal("second stop should report already stopped")
	}
}

func TestProgressConcurrentStopAndClear(t *testing.T) {
	var buf bytes.Buffer
	p := NewProgress(&buf)
	p.Add("one", progressTestState("working"))

	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			p.StopAndClear()
		}()
	}
	wg.Wait()
}
