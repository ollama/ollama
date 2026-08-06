package chat

import (
	"bytes"
	"io"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

type fakeTermFile struct {
	bytes.Buffer
}

func (f *fakeTermFile) Read(p []byte) (int, error) { return f.Buffer.Read(p) }
func (f *fakeTermFile) Close() error               { return nil }
func (f *fakeTermFile) Fd() uintptr                { return 0 }

func TestCursorSyncWriterPassthroughWhenNoCaret(t *testing.T) {
	f := &fakeTermFile{}
	caret := &termCaretState{}
	w := newCursorSyncWriter(f, caret)

	n, err := w.Write([]byte("abc"))
	if err != nil || n != 3 {
		t.Fatalf("Write = (%d, %v), want (3, nil)", n, err)
	}
	if got := f.String(); got != "abc" {
		t.Fatalf("output = %q, want %q", got, "abc")
	}
}

func TestCursorSyncWriterParksAndRestores(t *testing.T) {
	f := &fakeTermFile{}
	caret := &termCaretState{}
	caret.set(caretPos{rowsUp: 2, col: 4, ok: true})
	w := newCursorSyncWriter(f, caret)

	if _, err := w.Write([]byte("abc")); err != nil {
		t.Fatalf("first Write: %v", err)
	}
	want := "abc\x1b7\x1b[2A\x1b[5G"
	if got := f.String(); got != want {
		t.Fatalf("first write output = %q, want %q", got, want)
	}
	if !w.parked {
		t.Fatal("writer should be parked after a write with a valid caret")
	}

	f.Reset()
	if _, err := w.Write([]byte("def")); err != nil {
		t.Fatalf("second Write: %v", err)
	}
	want = "\x1b8def\x1b7\x1b[2A\x1b[5G"
	if got := f.String(); got != want {
		t.Fatalf("second write output = %q, want %q", got, want)
	}
}

func TestCursorSyncWriterOmitsCursorUpWhenRowsUpIsZero(t *testing.T) {
	f := &fakeTermFile{}
	caret := &termCaretState{}
	caret.set(caretPos{rowsUp: 0, col: 3, ok: true})
	w := newCursorSyncWriter(f, caret)

	if _, err := w.Write([]byte("x")); err != nil {
		t.Fatalf("Write: %v", err)
	}
	want := "x\x1b7\x1b[4G"
	if got := f.String(); got != want {
		t.Fatalf("output = %q, want %q", got, want)
	}
}

func TestCursorSyncWriterUnparksWhenCaretBecomesInvalid(t *testing.T) {
	f := &fakeTermFile{}
	caret := &termCaretState{}
	caret.set(caretPos{rowsUp: 1, col: 0, ok: true})
	w := newCursorSyncWriter(f, caret)

	if _, err := w.Write([]byte("a")); err != nil {
		t.Fatalf("first Write: %v", err)
	}
	if !w.parked {
		t.Fatal("expected writer to be parked")
	}

	caret.set(caretPos{})
	f.Reset()
	if _, err := w.Write([]byte("b")); err != nil {
		t.Fatalf("second Write: %v", err)
	}
	if got := f.String(); got != "\x1b8b" {
		t.Fatalf("output = %q, want %q", got, "\x1b8b")
	}
	if w.parked {
		t.Fatal("writer should no longer be parked")
	}
}

func TestCursorSyncWriterRestoreIfParked(t *testing.T) {
	f := &fakeTermFile{}
	caret := &termCaretState{}
	caret.set(caretPos{rowsUp: 1, col: 0, ok: true})
	w := newCursorSyncWriter(f, caret)

	if _, err := w.Write([]byte("a")); err != nil {
		t.Fatalf("Write: %v", err)
	}

	f.Reset()
	w.restoreIfParked()
	if got := f.String(); got != "\x1b8" {
		t.Fatalf("restoreIfParked output = %q, want %q", got, "\x1b8")
	}
	if w.parked {
		t.Fatal("writer should not be parked after restoreIfParked")
	}

	f.Reset()
	w.restoreIfParked()
	if got := f.String(); got != "" {
		t.Fatalf("second restoreIfParked should be a no-op, got %q", got)
	}
}

// concurrencyDetectingTermFile fails the test if two Write calls ever
// overlap, simulating bubbletea's real usage: its own renderer already
// serializes writes with an internal mutex (verified by reading
// standard_renderer.go), but cursorSyncWriter must not assume that and stay
// safe on its own, since it's called from both the event-loop goroutine
// (execute, e.g. hideCursor/showCursor) and the renderer's ticker goroutine
// (flush).
type concurrencyDetectingTermFile struct {
	fakeTermFile
	inFlight int32
	t        *testing.T
}

func (f *concurrencyDetectingTermFile) Write(p []byte) (int, error) {
	if !atomic.CompareAndSwapInt32(&f.inFlight, 0, 1) {
		f.t.Fatal("concurrent Write calls reached the underlying file")
	}
	// Give a concurrent Write a window to slip in if the caller's
	// synchronization is broken.
	time.Sleep(time.Millisecond)
	n, err := f.fakeTermFile.Write(p)
	atomic.StoreInt32(&f.inFlight, 0)
	return n, err
}

func TestCursorSyncWriterSerializesConcurrentWrites(t *testing.T) {
	f := &concurrencyDetectingTermFile{t: t}
	caret := &termCaretState{}
	w := newCursorSyncWriter(f, caret)

	const goroutines = 20
	var wg sync.WaitGroup
	wg.Add(goroutines * 2)
	for i := 0; i < goroutines; i++ {
		go func(i int) {
			defer wg.Done()
			_, _ = w.Write([]byte("frame"))
		}(i)
		go func(i int) {
			defer wg.Done()
			caret.set(caretPos{rowsUp: i % 3, col: i, ok: i%2 == 0})
		}(i)
	}
	wg.Wait()
}

type erroringTermFile struct {
	fakeTermFile
	failAfter int
}

// Write mimics a short write: it only accepts failAfter bytes of the
// payload and reports io.ErrShortWrite, per the io.Writer contract.
func (f *erroringTermFile) Write(p []byte) (int, error) {
	n := min(len(p), f.failAfter)
	written, _ := f.Buffer.Write(p[:n])
	if written < len(p) {
		return written, io.ErrShortWrite
	}
	return written, nil
}

func TestCursorSyncWriterReportsPayloadBytesOnPartialWrite(t *testing.T) {
	f := &erroringTermFile{failAfter: 2}
	caret := &termCaretState{}
	caret.set(caretPos{rowsUp: 1, col: 0, ok: true})
	w := newCursorSyncWriter(f, caret)

	n, err := w.Write([]byte("abc"))
	if err == nil {
		t.Fatal("expected an error from the underlying writer")
	}
	if n != 2 {
		t.Fatalf("n = %d, want 2 (payload bytes actually written)", n)
	}
}
