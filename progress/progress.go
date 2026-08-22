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
	done   chan struct{}
	states []State
}

func NewProgress(w io.Writer) *Progress {
	ticker := time.NewTicker(100 * time.Millisecond)
	p := &Progress{
		w:      bufio.NewWriter(w),
		ticker: ticker,
		done:   make(chan struct{}),
	}
	go p.start(ticker)
	return p
}

// stop halts rendering, draws the final state, and writes the closing output
// in a single critical section so it cannot interleave with the render
// goroutine on the shared writer. clear selects the StopAndClear epilogue
// (erase the progress lines) over the Stop epilogue (trailing newline).
func (p *Progress) stop(clear bool) bool {
	p.mu.Lock()
	defer p.mu.Unlock()

	if clear {
		defer p.w.Flush()
		fmt.Fprint(p.w, "\033[?25l")
		defer fmt.Fprint(p.w, "\033[?25h")
	}

	if p.ticker != nil {
		p.ticker.Stop()
		p.ticker = nil
		close(p.done)

		for _, state := range p.states {
			if spinner, ok := state.(*Spinner); ok {
				spinner.Stop()
			}
		}

		p.renderLocked()

		if clear {
			// clear all progress lines
			for i := range p.pos {
				if i > 0 {
					fmt.Fprint(p.w, "\033[A")
				}
				fmt.Fprint(p.w, "\033[2K\033[1G")
			}
		} else {
			fmt.Fprint(p.w, "\n")
			p.w.Flush()
		}

		return true
	}

	return false
}

func (p *Progress) Stop() bool {
	return p.stop(false)
}

func (p *Progress) StopAndClear() bool {
	return p.stop(true)
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

func (p *Progress) start(ticker *time.Ticker) {
	for {
		select {
		case <-p.done:
			return
		case <-ticker.C:
			p.render()
		}
	}
}
