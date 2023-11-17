package progress

import (
	"fmt"
	"io"
	"sync"
	"time"
)

type State interface {
	String() string
}

type Progress struct {
	mu  sync.Mutex
	pos int
	w   io.Writer

	ticker *time.Ticker
	states []State
}

func NewProgress(w io.Writer) *Progress {
	p := &Progress{w: w}
	go p.start()
	return p
}

func (p *Progress) Stop() bool {
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

func (p *Progress) StopAndClear() bool {
	fmt.Fprint(p.w, "\033[?25l")
	defer fmt.Fprint(p.w, "\033[?25h")

	stopped := p.Stop()
	if stopped {
		// clear the progress bar by:
		// 1. for each line in the progress:
		//   a. move the cursor up one line
		//   b. clear the line
		for i := 0; i < p.pos; i++ {
			fmt.Fprint(p.w, "\033[A\033[2K")
		}
	}

	return stopped
}

func (p *Progress) Add(key string, state State) {
	p.mu.Lock()
	defer p.mu.Unlock()

	p.states = append(p.states, state)
}

func (p *Progress) render() error {
	p.mu.Lock()
	defer p.mu.Unlock()

	fmt.Fprint(p.w, "\033[?25l")
	defer fmt.Fprint(p.w, "\033[?25h")

	if p.pos > 0 {
		fmt.Fprintf(p.w, "\033[%dA", p.pos)
	}

	for _, state := range p.states {
		fmt.Fprintln(p.w, state.String())
	}

	if len(p.states) > 0 {
		p.pos = len(p.states)
	}

	return nil
}

func (p *Progress) start() {
	p.ticker = time.NewTicker(100 * time.Millisecond)
	for range p.ticker.C {
		p.render()
	}
}
