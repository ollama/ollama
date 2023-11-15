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
	p := &Progress{pos: -1, w: w}
	go p.start()
	return p
}

func (p *Progress) Stop() {
	if p.ticker != nil {
		p.ticker.Stop()
		p.ticker = nil
		p.render()
	}
}

func (p *Progress) Add(key string, state State) {
	p.mu.Lock()
	defer p.mu.Unlock()

	p.states = append(p.states, state)
}

func (p *Progress) render() error {
	p.mu.Lock()
	defer p.mu.Unlock()

	fmt.Fprintf(p.w, "\033[%dA", p.pos)
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
