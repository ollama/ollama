package progress

import (
	"fmt"
	"io"
	"os"
	"strings"
	"sync"
	"time"

	"golang.org/x/term"
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
	stopped := p.Stop()
	if stopped {
		termWidth, _, err := term.GetSize(int(os.Stderr.Fd()))
		if err != nil {
			panic(err)
		}

		// clear the progress bar by:
		// 1. reset to beginning of line
		// 2. move up to the first line of the progress bar
		// 3. fill the terminal width with spaces
		// 4. reset to beginning of line
		fmt.Fprintf(p.w, "\r\033[%dA%s\r", p.pos, strings.Repeat(" ", termWidth))
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
