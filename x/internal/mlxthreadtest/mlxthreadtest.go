// Package mlxthreadtest runs tests on a persistent MLX worker thread.
package mlxthreadtest

import (
	"context"
	"testing"

	"github.com/ollama/ollama/x/internal/mlxthread"
)

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

// Run executes fn on thread.
func Run(t *testing.T, thread *mlxthread.Thread, fn func(*T)) {
	t.Helper()

	mt := &T{testReporter: t}
	if err := thread.Do(context.Background(), func() error {
		mt.run(fn)
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if mt.skipped {
		t.SkipNow()
	}
	if mt.aborted {
		t.FailNow()
	}
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
	var panicValue any
	func() {
		defer func() {
			if v := recover(); v != nil && v != abortPanic {
				panicValue = v
			}
		}()
		fn(t)
	}()

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
