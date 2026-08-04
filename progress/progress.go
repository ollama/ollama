package progress

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"sync"
	"time"

	"golang.org/x/term"
)

const (
	defaultTermWidth  = 80
	defaultTermHeight = 24
)

type State interface {
	String() string
}

type Progress struct {
	mu sync.Mutex
	// buffer output to minimize flickering on all terminals
	w *bufio.Writer

	pos int

	states []State

	stopOnce sync.Once
	// done is closed to tell the render loop to exit.
	done chan struct{}
}

func NewProgress(w io.Writer) *Progress {
	p := &Progress{w: bufio.NewWriter(w), done: make(chan struct{})}
	go p.start()
	return p
}

// stop halts the render loop, stopping any spinners first. It reports whether
// rendering was active and how many lines were last rendered.
func (p *Progress) stop() (bool, int) {
	var stopped bool
	p.stopOnce.Do(func() {
		close(p.done)
		stopped = true
	})

	p.mu.Lock()
	defer p.mu.Unlock()

	for _, state := range p.states {
		if spinner, ok := state.(*Spinner); ok {
			spinner.Stop()
		}
	}

	if stopped {
		p.renderLocked()
	}
	return stopped, p.pos
}

func (p *Progress) Stop() bool {
	stopped, _ := p.stop()
	if stopped {
		fmt.Fprint(p.w, "\n")
		p.w.Flush()
	}
	return stopped
}

func (p *Progress) StopAndClear() bool {
	defer p.w.Flush()

	fmt.Fprint(p.w, "\033[?25l")
	defer fmt.Fprint(p.w, "\033[?25h")

	stopped, pos := p.stop()
	if stopped {
		// clear all progress lines
		for i := range pos {
			if i > 0 {
				fmt.Fprint(p.w, "\033[A")
			}
			fmt.Fprint(p.w, "\033[2K\033[1G")
		}
	}

	return stopped
}

func (p *Progress) Add(key string, state State) {
	p.mu.Lock()
	defer p.mu.Unlock()

	p.states = append(p.states, state)
}

func (p *Progress) render() {
	p.mu.Lock()
	defer p.mu.Unlock()

	p.renderLocked()
}

// renderLocked renders with p.mu held.
func (p *Progress) renderLocked() {
	_, termHeight, err := term.GetSize(int(os.Stderr.Fd()))
	if err != nil {
		termHeight = defaultTermHeight
	}

	defer p.w.Flush()

	// eliminate flickering on terminals that support synchronized output
	fmt.Fprint(p.w, "\033[?2026h")
	defer fmt.Fprint(p.w, "\033[?2026l")

	fmt.Fprint(p.w, "\033[?25l")
	defer fmt.Fprint(p.w, "\033[?25h")

	// move the cursor back to the beginning
	for range p.pos - 1 {
		fmt.Fprint(p.w, "\033[A")
	}
	fmt.Fprint(p.w, "\033[1G")

	// render progress lines
	maxHeight := min(len(p.states), termHeight)
	for i := len(p.states) - maxHeight; i < len(p.states); i++ {
		fmt.Fprint(p.w, p.states[i].String(), "\033[K")
		if i < len(p.states)-1 {
			fmt.Fprint(p.w, "\n")
		}
	}

	p.pos = len(p.states)
}

func (p *Progress) start() {
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-p.done:
			return
		case <-ticker.C:
			p.render()
		}
	}
}
