// Package mlxthreadtest runs tests on a persistent MLX worker thread.
package mlxthreadtest

import (
	"context"
	"testing"

	"github.com/ollama/ollama/x/internal/mlxthread"
)

// Thread is a pinned worker used by MLX tests.
type Thread struct {
	worker *mlxthread.Thread
	id     uint64
}

// Start creates a pinned test worker.
func Start(name string, init func() error) (*Thread, error) {
	t := &Thread{}
	thread, err := mlxthread.Start(name, func() error {
		t.id = currentThreadID()
		if init != nil {
			return init()
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	t.worker = thread
	return t, nil
}

// Do runs fn on the pinned worker.
func (t *Thread) Do(ctx context.Context, fn func() error) error {
	if t.onWorker() {
		panic("mlxthreadtest.Thread.Do called from its pinned worker")
	}
	return t.worker.Do(ctx, fn)
}

// Stop shuts down the pinned worker after running cleanup on it.
func (t *Thread) Stop(ctx context.Context, cleanup func()) error {
	if t.onWorker() {
		panic("mlxthreadtest.Thread.Stop called from its pinned worker")
	}
	return t.worker.Stop(ctx, cleanup)
}

func (t *Thread) onWorker() bool {
	id := currentThreadID()
	return id != 0 && id == t.id
}

// T is the subset of testing.T supported by MLX test bodies. Operations that
// end a test are replayed by Run on the test goroutine so the MLX worker remains
// alive.
type T struct {
	testReporter
	cleanups []func()
	skipped  bool
	aborted  bool
}

// abortPanic unwinds a test body without terminating the worker goroutine.
var abortPanic = new(struct{ marker byte })

type testReporter interface {
	Error(...any)
	Errorf(string, ...any)
	Fail()
	Failed() bool
	Helper()
	Log(...any)
	Logf(string, ...any)
}

// Run executes fn on thread. The callback must use its T argument; calling
// FailNow or SkipNow on a captured *testing.T terminates the pinned worker.
func Run(t *testing.T, thread *Thread, fn func(*T)) {
	t.Helper()
	if thread.onWorker() {
		panic("mlxthreadtest.Run called recursively from its pinned worker")
	}

	mt := &T{testReporter: t}
	result := make(chan runResult, 1)
	go func() {
		defer func() {
			if v := recover(); v != nil {
				result <- runResult{panicValue: v}
			}
		}()
		err := thread.Do(context.Background(), func() error {
			returned := false
			defer func() {
				if v := recover(); v != nil {
					panic(v)
				}
				if !returned {
					result <- runResult{goexit: true}
				}
			}()

			mt.run(fn)
			returned = true
			return nil
		})
		result <- runResult{err: err}
	}()

	res := <-result
	if res.goexit {
		panic("pinned test body called runtime.Goexit; use the test value passed to the callback")
	}
	if res.panicValue != nil {
		panic(res.panicValue)
	}
	if res.err != nil {
		t.Fatal(res.err)
	}
	if mt.skipped {
		t.SkipNow()
	}
	if mt.aborted {
		t.FailNow()
	}
}

type runResult struct {
	err        error
	panicValue any
	goexit     bool
}

// Cleanup registers fn to run on the MLX thread after the current body.
func (t *T) Cleanup(fn func()) {
	t.cleanups = append(t.cleanups, fn)
}

func (t *T) FailNow() {
	t.Fail()
	t.aborted = true
	panic(abortPanic)
}

func (t *T) Fatal(args ...any) {
	t.Helper()
	t.Error(args...)
	t.aborted = true
	panic(abortPanic)
}

func (t *T) Fatalf(format string, args ...any) {
	t.Helper()
	t.Errorf(format, args...)
	t.aborted = true
	panic(abortPanic)
}

func (t *T) Skip(args ...any) {
	t.Helper()
	t.Log(args...)
	t.SkipNow()
}

func (t *T) Skipf(format string, args ...any) {
	t.Helper()
	t.Logf(format, args...)
	t.SkipNow()
}

func (t *T) SkipNow() {
	t.skipped = true
	t.aborted = true
	panic(abortPanic)
}

func (t *T) Skipped() bool {
	return t.skipped
}

func (t *T) run(fn func(*T)) {
	defer func() {
		if v := recover(); v != nil && v != abortPanic {
			panic(v)
		}
	}()
	defer t.runCleanups()
	fn(t)
}

func (t *T) runCleanups() {
	var panicValue any
	for len(t.cleanups) > 0 {
		last := len(t.cleanups) - 1
		cleanup := t.cleanups[last]
		t.cleanups = t.cleanups[:last]
		func() {
			defer func() {
				if v := recover(); v != nil && v != abortPanic && panicValue == nil {
					panicValue = v
				}
			}()
			cleanup()
		}()
	}
	if panicValue != nil {
		panic(panicValue)
	}
}
