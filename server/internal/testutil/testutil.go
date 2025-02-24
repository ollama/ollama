package testutil

import (
	"bytes"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// LogWriter returns an [io.Writer] that logs each Write using t.Log.
func LogWriter(t *testing.T) io.Writer {
	return testWriter{t}
}

type testWriter struct{ t *testing.T }

func (w testWriter) Write(b []byte) (int, error) {
	w.t.Logf("%s", b)
	return len(b), nil
}

// Slogger returns a [*slog.Logger] that writes each message
// using t.Log.
func Slogger(t *testing.T) *slog.Logger {
	return slog.New(slog.NewTextHandler(LogWriter(t), nil))
}

// SlogBuffer returns a [*slog.Logger] that writes each message to out.
func SlogBuffer() (lg *slog.Logger, out *bytes.Buffer) {
	var buf bytes.Buffer
	lg = slog.New(slog.NewTextHandler(&buf, nil))
	return lg, &buf
}

// Check calls t.Fatal(err) if err is not nil.
func Check(t *testing.T, err error) {
	if err != nil {
		t.Helper()
		t.Fatal(err)
	}
}

// CheckFunc exists so other packages do not need to invent their own type for
// taking a Check function.
type CheckFunc func(err error)

// Checker returns a check function that
// calls t.Fatal if err is not nil.
func Checker(t *testing.T) (check func(err error)) {
	return func(err error) {
		if err != nil {
			t.Helper()
			t.Fatal(err)
		}
	}
}

// StopPanic runs f but silently recovers from any panic f causes.
// The normal usage is:
//
//	testutil.StopPanic(func() {
//		callThatShouldPanic()
//		t.Errorf("callThatShouldPanic did not panic")
//	})
func StopPanic(f func()) {
	defer func() { recover() }()
	f()
}

// CheckTime calls t.Fatalf if got != want. Included in the error message is
// want.Sub(got) to help diagnose the difference, along with their values in
// UTC.
func CheckTime(t *testing.T, got, want time.Time) {
	t.Helper()
	if !got.Equal(want) {
		t.Fatalf("got %v, want %v (%v)", got.UTC(), want.UTC(), want.Sub(got))
	}
}

// WriteFile writes data to a file named name. It makes the directory if it
// doesn't exist and sets the file mode to perm.
//
// The name must be a relative path and must not contain .. or start with a /;
// otherwise WriteFile will panic.
func WriteFile[S []byte | string](t testing.TB, name string, data S) {
	t.Helper()

	if filepath.IsAbs(name) {
		t.Fatalf("WriteFile: name must be a relative path, got %q", name)
	}
	name = filepath.Clean(name)
	dir := filepath.Dir(name)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(name, []byte(data), 0o644); err != nil {
		t.Fatal(err)
	}
}
