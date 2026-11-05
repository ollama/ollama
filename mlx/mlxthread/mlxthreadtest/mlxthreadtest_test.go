package mlxthreadtest

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

type fakeReporter struct{ failed bool }

func (t *fakeReporter) Error(...any)          { t.failed = true }
func (t *fakeReporter) Errorf(string, ...any) { t.failed = true }
func (t *fakeReporter) Fail()                 { t.failed = true }
func (t *fakeReporter) Failed() bool          { return t.failed }
func (*fakeReporter) Helper()                 {}
func (*fakeReporter) Log(...any)              {}
func (*fakeReporter) Logf(string, ...any)     {}

func TestControlsDoNotStopWorker(t *testing.T) {
	thread, err := Start("test", nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := thread.Stop(context.Background(), nil); err != nil {
			t.Error(err)
		}
	})

	tests := []struct {
		name        string
		body        func(*T)
		wantFailed  bool
		wantSkipped bool
	}{
		{name: "fail now", body: func(t *T) { t.FailNow() }, wantFailed: true},
		{name: "fatal", body: func(t *T) { t.Fatal("stop") }, wantFailed: true},
		{name: "fatalf", body: func(t *T) { t.Fatalf("%s", "stop") }, wantFailed: true},
		{name: "skip now", body: func(t *T) { t.SkipNow() }, wantSkipped: true},
		{name: "skip", body: func(t *T) { t.Skip("stop") }, wantSkipped: true},
		{name: "skipf", body: func(t *T) { t.Skipf("%s", "stop") }, wantSkipped: true},
		{
			name: "failed skip",
			body: func(t *T) {
				t.Error("failed")
				t.Skip("stop")
			},
			wantFailed:  true,
			wantSkipped: true,
		},
		{
			name: "cleanup fail now",
			body: func(t *T) {
				t.Cleanup(func() { t.FailNow() })
			},
			wantFailed: true,
		},
		{
			name: "cleanup fatal",
			body: func(t *T) {
				t.Cleanup(func() { t.Fatal("stop") })
			},
			wantFailed: true,
		},
		{
			name: "cleanup skip now",
			body: func(t *T) {
				t.Cleanup(func() { t.SkipNow() })
			},
			wantSkipped: true,
		},
		{
			name: "cleanup skip",
			body: func(t *T) {
				t.Cleanup(func() { t.Skip("stop") })
			},
			wantSkipped: true,
		},
	}
	for _, tt := range tests {
		reporter := &fakeReporter{}
		mt := &T{testReporter: reporter}
		if err := thread.Do(context.Background(), func() error {
			mt.run(tt.body)
			return nil
		}); err != nil {
			t.Fatalf("%s: %v", tt.name, err)
		}
		if mt.Failed() != tt.wantFailed || mt.skipped != tt.wantSkipped {
			t.Fatalf("%s: failed:%v skipped:%v", tt.name, mt.Failed(), mt.skipped)
		}

		ran := false
		if err := thread.Do(context.Background(), func() error {
			ran = true
			return nil
		}); err != nil {
			t.Fatalf("%s follow-up: %v", tt.name, err)
		}
		if !ran {
			t.Fatalf("worker did not run after %s", tt.name)
		}
	}

	reached := false
	t.Run("replay skip", func(t *testing.T) {
		Run(t, thread, func(t *T) { t.Skip("stop") })
		reached = true
	})
	if reached {
		t.Fatal("Run returned after Skip")
	}
	if err := thread.Do(context.Background(), func() error { return nil }); err != nil {
		t.Fatalf("worker did not survive replayed Skip: %v", err)
	}
}

func TestRunReplaysFatalWithoutStoppingWorker(t *testing.T) {
	if marker := os.Getenv("OLLAMA_TEST_MLX_FATAL_MARKER"); marker != "" {
		thread, err := Start("test", nil)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := thread.Stop(context.Background(), nil); err != nil {
				t.Error(err)
			}
		})
		t.Cleanup(func() {
			if err := thread.Do(context.Background(), func() error { return nil }); err != nil {
				t.Errorf("worker did not survive replayed Fatal: %v", err)
				return
			}
			if err := os.WriteFile(marker, nil, 0o600); err != nil {
				t.Error(err)
			}
		})

		Run(t, thread, func(t *T) { t.Fatal("stop") })
		return
	}

	marker := filepath.Join(t.TempDir(), "worker-survived")
	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestRunReplaysFatalWithoutStoppingWorker$", "-test.timeout=2s")
	cmd.Env = append(os.Environ(), "OLLAMA_TEST_MLX_FATAL_MARKER="+marker)
	output, err := cmd.CombinedOutput()
	if err == nil {
		t.Fatal("subprocess succeeded, want Fatal failure")
	}
	if ctx.Err() != nil {
		t.Fatalf("subprocess hung: %v", ctx.Err())
	}
	if !strings.Contains(string(output), "stop") {
		t.Fatalf("missing Fatal diagnostic:\n%s", output)
	}
	if _, err := os.Stat(marker); err != nil {
		t.Fatalf("worker survival check did not complete: %v\n%s", err, output)
	}
}

func TestRunRejectsRecursiveDispatch(t *testing.T) {
	if currentThreadID() == 0 {
		t.Skip("OS thread IDs are not available on this platform")
	}

	thread, err := Start("test", nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := thread.Stop(context.Background(), nil); err != nil {
			t.Error(err)
		}
	})

	tests := []struct {
		name string
		want string
		body func(*testing.T)
	}{
		{
			name: "Run",
			want: "called recursively",
			body: func(t *testing.T) { Run(t, thread, func(*T) {}) },
		},
		{
			name: "Do",
			want: "Thread.Do called from its pinned worker",
			body: func(*testing.T) {
				_ = thread.Do(context.Background(), func() error { return nil })
			},
		},
		{
			name: "Stop",
			want: "Thread.Stop called from its pinned worker",
			body: func(*testing.T) { _ = thread.Stop(context.Background(), nil) },
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var got any
			func() {
				defer func() { got = recover() }()
				Run(t, thread, func(*T) { tt.body(t) })
			}()
			if !strings.Contains(fmt.Sprint(got), tt.want) {
				t.Fatalf("got panic %v, want %q", got, tt.want)
			}
			if err := thread.Do(context.Background(), func() error { return nil }); err != nil {
				t.Fatalf("worker did not survive recursive %s: %v", tt.name, err)
			}
		})
	}
}

func TestRunRejectsTestingTGoexit(t *testing.T) {
	if method := os.Getenv("OLLAMA_TEST_MLX_GOEXIT"); method != "" {
		thread, err := Start("test", nil)
		if err != nil {
			t.Fatal(err)
		}
		Run(t, thread, func(*T) {
			switch method {
			case "fail":
				t.FailNow()
			case "skip":
				t.SkipNow()
			default:
				panic("unknown Goexit test method")
			}
		})
		return
	}

	for _, method := range []string{"fail", "skip"} {
		t.Run(method, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
			defer cancel()
			cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestRunRejectsTestingTGoexit$", "-test.timeout=2s")
			cmd.Env = append(os.Environ(), "OLLAMA_TEST_MLX_GOEXIT="+method)
			output, err := cmd.CombinedOutput()
			if err == nil {
				t.Fatal("subprocess succeeded, want misuse failure")
			}
			if ctx.Err() != nil {
				t.Fatalf("subprocess hung: %v", ctx.Err())
			}
			if !strings.Contains(string(output), "pinned test body called runtime.Goexit") {
				t.Fatalf("missing Goexit diagnostic:\n%s", output)
			}
		})
	}
}

func TestCleanupOrder(t *testing.T) {
	reporter := &fakeReporter{}
	mt := &T{testReporter: reporter}
	var got []string

	mt.run(func(t *T) {
		t.Cleanup(func() { got = append(got, "cleanup 1") })
		t.Cleanup(func() {
			got = append(got, "cleanup 2")
			t.Cleanup(func() { got = append(got, "cleanup 3") })
		})
		got = append(got, "body")
	})

	want := []string{"body", "cleanup 2", "cleanup 3", "cleanup 1"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v, want %v", got, want)
	}
}

func TestCleanupFailureDoesNotHideBodyPanic(t *testing.T) {
	reporter := &fakeReporter{}
	mt := &T{testReporter: reporter}
	var got any
	func() {
		defer func() { got = recover() }()
		mt.run(func(t *T) {
			t.Cleanup(func() { t.Fatal("cleanup failure") })
			panic("body panic")
		})
	}()
	if got != "body panic" {
		t.Fatalf("got panic %v, want body panic", got)
	}
	if !reporter.failed {
		t.Fatal("cleanup Fatal did not fail the test")
	}
}

func TestCleanupPanicRunsRemainingCleanups(t *testing.T) {
	reporter := &fakeReporter{}
	mt := &T{testReporter: reporter}
	var cleanups []string
	var got any
	func() {
		defer func() { got = recover() }()
		mt.run(func(t *T) {
			t.Cleanup(func() { cleanups = append(cleanups, "first") })
			t.Cleanup(func() {
				cleanups = append(cleanups, "second")
				panic("cleanup panic")
			})
		})
	}()
	if got != "cleanup panic" {
		t.Fatalf("got panic %v, want cleanup panic", got)
	}
	if want := []string{"second", "first"}; !reflect.DeepEqual(cleanups, want) {
		t.Fatalf("got cleanups %v, want %v", cleanups, want)
	}
}

func TestCleanupRunsOnWorker(t *testing.T) {
	thread, err := Start("test", nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := thread.Stop(context.Background(), nil); err != nil {
			t.Error(err)
		}
	})

	var bodyID, cleanupID uint64
	Run(t, thread, func(t *T) {
		bodyID = currentThreadID()
		t.Cleanup(func() { cleanupID = currentThreadID() })
	})
	if bodyID != thread.id || cleanupID != thread.id {
		t.Fatalf("body thread %d, cleanup thread %d, want worker thread %d", bodyID, cleanupID, thread.id)
	}
}

func panicAtTestBody() {
	panic("test panic")
}

func TestPanicIncludesTestBodyStack(t *testing.T) {
	thread, err := Start("test", nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := thread.Stop(context.Background(), nil); err != nil {
			t.Error(err)
		}
	})

	var cleanupRan bool
	var got any
	func() {
		defer func() { got = recover() }()
		Run(t, thread, func(t *T) {
			t.Cleanup(func() { cleanupRan = true })
			panicAtTestBody()
		})
	}()
	if !cleanupRan {
		t.Fatal("cleanup did not run after panic")
	}
	if !strings.Contains(fmt.Sprint(got), "panicAtTestBody") {
		t.Fatalf("panic stack does not include test body:\n%v", got)
	}
	if err := thread.Do(context.Background(), func() error { return nil }); err != nil {
		t.Fatalf("worker did not survive panic: %v", err)
	}
}
