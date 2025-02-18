package progress

import (
	"bufio"
	"fmt"
	"io"
	"sync"
	"time"
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

func (p *Progress) stop() bool {
	for _, state := range p.states {
		if spinner, ok := state.(*Spinner); ok {
			spinner.Stop()
		}
	}

	if p.ticker != nil {
		p.ticker.Stop()
		p.ticker = nil
		p.render()
		return true
	}

	return false
}

func (p *Progress) Stop() bool {
	stopped := p.stop()
	if stopped {
		fmt.Fprintln(p.w)
	}

	// show cursor
	fmt.Fprint(p.w, "\033[?25h")
	p.w.Flush()
	return stopped
}

func (p *Progress) StopAndClear() bool {
	stopped := p.stop()
	if stopped {
		// clear all progress lines
		for range p.pos - 1 {
			fmt.Fprint(p.w, "\033[A")
		}

		fmt.Fprint(p.w, "\033[2K", "\033[1G")
	}

	// show cursor
	fmt.Fprint(p.w, "\033[?25h")
	p.w.Flush()
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

	fmt.Fprint(p.w, "\033[?2026h")
	defer fmt.Fprint(p.w, "\033[?2026l")

	for range p.pos - 1 {
		fmt.Fprint(p.w, "\033[A")
	}

	fmt.Fprint(p.w, "\033[1G")

	// render progress lines
	for i, state := range p.states {
		fmt.Fprint(p.w, state.String(), "\033[K")
		if i < len(p.states)-1 {
			fmt.Fprint(p.w, "\n")
		}
	}

	p.pos = len(p.states)
	p.w.Flush()
}

func (p *Progress) start() {
	p.ticker = time.NewTicker(100 * time.Millisecond)
	// hide cursor
	fmt.Fprint(p.w, "\033[?25l")
	for range p.ticker.C {
		p.render()
	}
}
