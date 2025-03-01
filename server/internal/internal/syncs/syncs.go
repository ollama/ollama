package syncs

import (
	"sync"
	"sync/atomic"
)

// Group is a [sync.WaitGroup] with a Go method.
type Group struct {
	wg sync.WaitGroup
	n  atomic.Int64
}

func (g *Group) Go(f func()) {
	g.wg.Add(1)
	go func() {
		g.n.Add(1) // Now we are running
		defer func() {
			g.wg.Done()
			g.n.Add(-1) // Now we are done
		}()
		f()
	}()
}

// Running returns the number of goroutines that are currently running.
//
// If a call to [Running] returns zero, and a call to [Wait] is made without
// any calls to [Go], then [Wait] will return immediately. This is true even if
// a goroutine is started and finishes between the two calls.
//
// It is possible for [Running] to return non-zero and for [Wait] to return
// immediately. This can happen if the all running goroutines finish between
// the two calls.
func (g *Group) Running() int64 {
	return g.n.Load()
}

func (g *Group) Wait() {
	g.wg.Wait()
}
