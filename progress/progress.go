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

	ticker *time.Ticker
	states []State
}

func NewProgress(w io.Writer) *Progress {
	p := &Progress{w: bufio.NewWriter(w)}
	go p.start()
	return p
}

func (p *Progress) Stop() bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.stopLocked()
}

func (p *Progress) StopAndClear() bool {
	p.mu.Lock()
	defer p.mu.Unlock()

	fmt.Fprint(p.w, "\033[?25l")
	defer fmt.Fprint(p.w, "\033[?25h")

	stopped := p.stopLocked()
	if stopped {
		// clear all progress lines
		for i := range p.pos {
			if i > 0 {
				fmt.Fprint(p.w, "\033[A")
			}
			fmt.Fprint(p.w, "\033[2K\033[1G")
		}
	}

	p.w.Flush()
	return stopped
}

// stopLocked stops the progress ticker. Must be called with p.mu held.
func (p *Progress) stopLocked() bool {
	// Stop all spinners
	for _, state := range p.states {
		if spinner, ok := state.(*Spinner); ok {
			spinner.Stop()
		}
	}

	if p.ticker != nil {
		ticker := p.ticker
		p.ticker = nil

		// Stop ticker OUTSIDE lock to avoid deadlock
		p.mu.Unlock()
		ticker.Stop()
		p.mu.Lock()

		p.renderLocked()
		fmt.Fprint(p.w, "\n")
		p.w.Flush()
		return true
	}

	return false
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

// renderLocked renders the progress. Must be called with p.mu held.
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

	// Write ticker field while holding lock
	p.mu.Lock()
	p.ticker = ticker
	p.mu.Unlock()

	for range ticker.C {
		p.render()
	}
}
