package mlxthread

import (
	"context"
	"errors"
	"runtime"
	"sync/atomic"
)

var ErrStopped = errors.New("mlx thread stopped")

type Thread struct {
	name string

	jobs     chan job
	done     chan struct{}
	stopping atomic.Bool
}

type job struct {
	fn     func() error
	result chan result
	stop   bool
}

type result struct {
	err        error
	panicValue any
}

// Start creates a long-lived worker goroutine locked to one OS thread.
func Start(name string, init func() error) (*Thread, error) {
	t := &Thread{
		name: name,
		jobs: make(chan job),
		done: make(chan struct{}),
	}

	initResult := make(chan result, 1)
	go t.loop(init, initResult)

	res := <-initResult
	if res.panicValue != nil {
		panic(res.panicValue)
	}
	if res.err != nil {
		return nil, res.err
	}

	return t, nil
}

// Do runs fn on the locked OS thread.
//
// Context cancellation only applies while the work is queued. Once the worker
// accepts a job, the job runs until fn returns or reaches its own cancellation
// checks.
func (t *Thread) Do(ctx context.Context, fn func() error) error {
	res, err := t.enqueue(ctx, fn, false, false)
	if err != nil {
		return err
	}
	if res.panicValue != nil {
		panic(res.panicValue)
	}
	return res.err
}

func Call[T any](ctx context.Context, t *Thread, fn func() (T, error)) (T, error) {
	var value T
	err := t.Do(ctx, func() error {
		var err error
		value, err = fn()
		return err
	})
	return value, err
}

// Stop runs cleanup on the locked OS thread and then shuts the worker down.
func (t *Thread) Stop(ctx context.Context, cleanup func()) error {
	ctx = contextOrBackground(ctx)

	if !t.stopping.CompareAndSwap(false, true) {
		select {
		case <-t.done:
			return nil
		case <-ctx.Done():
			return ctx.Err()
		}
	}

	res, err := t.enqueue(ctx, func() error {
		if cleanup != nil {
			cleanup()
		}
		return nil
	}, true, true)
	if err != nil {
		if !errors.Is(err, ErrStopped) {
			t.stopping.Store(false)
		}
		return err
	}
	if res.panicValue != nil {
		panic(res.panicValue)
	}
	if res.err != nil {
		return res.err
	}

	select {
	case <-t.done:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (t *Thread) loop(init func() error, initResult chan<- result) {
	runtime.LockOSThread()
	// Deliberately do not unlock. MLX thread-local state belongs to this worker
	// until shutdown so it cannot leak back to arbitrary Go goroutines.

	res := run(init)
	initResult <- res
	if res.err != nil || res.panicValue != nil {
		close(t.done)
		return
	}

	for {
		j := <-t.jobs
		res := run(j.fn)
		j.result <- res
		if j.stop {
			close(t.done)
			return
		}
	}
}

func (t *Thread) enqueue(ctx context.Context, fn func() error, stop, allowStopping bool) (result, error) {
	ctx = contextOrBackground(ctx)
	if err := ctx.Err(); err != nil {
		return result{}, err
	}

	if !allowStopping && t.stopping.Load() {
		return result{}, ErrStopped
	}

	resultCh := make(chan result, 1)
	j := job{fn: fn, result: resultCh, stop: stop}

	select {
	case <-ctx.Done():
		return result{}, ctx.Err()
	case <-t.done:
		return result{}, ErrStopped
	case t.jobs <- j:
	}

	return <-resultCh, nil
}

func run(fn func() error) (res result) {
	defer func() {
		if v := recover(); v != nil {
			res.panicValue = v
		}
	}()
	if fn != nil {
		res.err = fn()
	}
	return res
}

func contextOrBackground(ctx context.Context) context.Context {
	if ctx != nil {
		return ctx
	}
	return context.Background()
}
