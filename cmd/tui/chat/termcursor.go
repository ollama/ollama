package chat

import (
	"bytes"
	"fmt"
	"sync"

	"github.com/charmbracelet/x/term"
)

// caretPos is the on-screen cell of the input caret, expressed relative to
// the last line of the frame bubbletea just rendered.
type caretPos struct {
	rowsUp int // rows above the frame's last line (>= 0)
	col    int // 0-based screen column
	ok     bool
}

// termCaretState is shared between the chatModel (event-loop goroutine,
// publishes one caret position per View call) and cursorSyncWriter (renderer
// goroutine, reads it on every flush).
type termCaretState struct {
	mu  sync.Mutex
	pos caretPos
}

func (s *termCaretState) set(p caretPos) {
	s.mu.Lock()
	s.pos = p
	s.mu.Unlock()
}

func (s *termCaretState) get() caretPos {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.pos
}

// cursorSyncWriter wraps the real terminal output. Bubble Tea hides the
// physical cursor for the whole TUI session and draws its own glyph, which
// leaves the physical cursor parked at column 0 of the last rendered line.
// Terminals anchor the IME composition (preedit) overlay at the physical
// cursor regardless of its visibility, so without this the preedit renders
// in the wrong place (see ollama/ollama#17521).
//
// After every write, cursorSyncWriter parks the (still hidden) physical
// cursor on the input caret's cell (DECSC then relative moves), and restores
// it (DECRC) before the next write so the renderer's own relative-cursor
// math is unaffected.
type cursorSyncWriter struct {
	f     term.File
	caret *termCaretState

	mu     sync.Mutex
	parked bool
}

func newCursorSyncWriter(f term.File, caret *termCaretState) *cursorSyncWriter {
	return &cursorSyncWriter{f: f, caret: caret}
}

func (w *cursorSyncWriter) Write(p []byte) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()

	var buf bytes.Buffer
	if w.parked {
		buf.WriteString("\x1b8") // DECRC
		w.parked = false
	}
	prefix := buf.Len()

	buf.Write(p)

	if c := w.caret.get(); c.ok {
		buf.WriteString("\x1b7") // DECSC
		if c.rowsUp > 0 {
			fmt.Fprintf(&buf, "\x1b[%dA", c.rowsUp)
		}
		fmt.Fprintf(&buf, "\x1b[%dG", c.col+1)
		w.parked = true
	}

	n, err := w.f.Write(buf.Bytes())
	if err != nil {
		// Report only how much of the caller's payload was written.
		written := n - prefix
		if written < 0 {
			written = 0
		}
		if written > len(p) {
			written = len(p)
		}
		return written, err
	}
	return len(p), nil
}

func (w *cursorSyncWriter) Read(p []byte) (int, error) { return w.f.Read(p) }
func (w *cursorSyncWriter) Close() error               { return w.f.Close() }
func (w *cursorSyncWriter) Fd() uintptr                { return w.f.Fd() }

// restoreIfParked un-parks the physical cursor if it was left on the caret
// cell. It covers the case where the program is killed without a final
// View/flush (the normal quit path already un-parks via the last frame).
func (w *cursorSyncWriter) restoreIfParked() {
	w.mu.Lock()
	defer w.mu.Unlock()
	if !w.parked {
		return
	}
	w.parked = false
	_, _ = w.f.Write([]byte("\x1b8"))
}
