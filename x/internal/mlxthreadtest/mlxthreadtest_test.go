package mlxthreadtest

import (
	"context"
	"reflect"
	"testing"

	"github.com/ollama/ollama/x/internal/mlxthread"
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
	thread, err := mlxthread.Start("test", nil)
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
		{name: "fatal", body: func(t *T) { t.Fatal("stop") }, wantFailed: true},
		{name: "skip", body: func(t *T) { t.Skip("stop") }, wantSkipped: true},
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
			name: "cleanup fatal",
			body: func(t *T) {
				t.Cleanup(func() { t.Fatal("stop") })
			},
			wantFailed: true,
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
