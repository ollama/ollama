package main

import (
	"os"
	"os/exec"
	"sync"
	"time"
)

// child is a runner process bench started. Wait runs in its own goroutine from
// the start, so an early exit is visible while bench waits for readiness.
type child struct {
	cmd    *exec.Cmd
	exited chan struct{}
	once   sync.Once
}

func startChild(cmd *exec.Cmd) (*child, error) {
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	c := &child{cmd: cmd, exited: make(chan struct{})}
	go func() {
		_ = cmd.Wait()
		close(c.exited)
	}()
	return c, nil
}

// alive reports whether the process is still running. A nil child is a runner
// bench did not start, which it cannot observe and treats as alive.
func (c *child) alive() bool {
	if c == nil {
		return true
	}
	select {
	case <-c.exited:
		return false
	default:
		return true
	}
}

// stop interrupts the process and kills it if the interrupt cannot be sent or
// it has not exited within timeout. It is safe to call more than once.
func (c *child) stop(timeout time.Duration) {
	if c == nil {
		return
	}
	c.once.Do(func() {
		if err := c.cmd.Process.Signal(os.Interrupt); err != nil {
			_ = c.cmd.Process.Kill()
		}
		select {
		case <-c.exited:
		case <-time.After(timeout):
			_ = c.cmd.Process.Kill()
			<-c.exited
		}
	})
}
