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

	// done signals the background render goroutine to exit. It is created in
	// NewProgress (before the goroutine starts) and closed exactly once by
	// stopOnce, so stop is idempotent and free of a start/stop race.
	done     chan struct{}
	stopOnce sync.Once
	states   []State
}

func NewProgress(w io.Writer) *Progress {
	p := &Progress{w: bufio.NewWriter(w), done: make(chan struct{})}
	go p.start()
	return p
}

func (p *Progress) stop() bool {
	for _, state := range p.states {
		if spinner, ok := state.(*Spinner); ok {
			spinner.Stop()
		}
	}

	stopped := false
	p.stopOnce.Do(func() {
		// Closing done makes the start goroutine return; time.Ticker.Stop
		// alone would not, because it does not close the ticker channel.
		close(p.done)
		stopped = true
	})

	if stopped {
		p.render()
	}

	return stopped
}

func (p *Progress) Stop() bool {
	stopped := p.stop()
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

	stopped := p.stop()
	if stopped {
		// clear all progress lines
		for i := range p.pos {
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
	_, termHeight, err := term.GetSize(int(os.Stderr.Fd()))
	if err != nil {
		termHeight = defaultTermHeight
	}

	p.mu.Lock()
	defer p.mu.Unlock()

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
	// The ticker is owned solely by this goroutine, so there is no shared
	// mutable state to race on. defer Stop releases the ticker on exit.
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			p.render()
		case <-p.done:
			return
		}
	}
}
