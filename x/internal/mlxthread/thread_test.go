package mlxthread

import (
	"context"
	"errors"
	"reflect"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestDoRunsInOrder(t *testing.T) {
	thread, err := Start("test", nil)
	if err != nil {
		t.Fatal(err)
	}
	defer thread.Stop(context.Background(), nil)

	var got []int
	for i := 0; i < 5; i++ {
		i := i
		if err := thread.Do(context.Background(), func() error {
			got = append(got, i)
			return nil
		}); err != nil {
			t.Fatal(err)
		}
	}

	if want := []int{0, 1, 2, 3, 4}; !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v, want %v", got, want)
	}
}

func TestDoPropagatesPanicToCaller(t *testing.T) {
	thread, err := Start("test", nil)
	if err != nil {
		t.Fatal(err)
	}
	defer thread.Stop(context.Background(), nil)

	defer func() {
		if got := recover(); got != "boom" {
			t.Fatalf("got panic %v, want boom", got)
		}
	}()

	_ = thread.Do(context.Background(), func() error {
		panic("boom")
	})
}

func TestDoCancelsBeforeJobStarts(t *testing.T) {
	thread, err := Start("test", nil)
	if err != nil {
		t.Fatal(err)
	}
	defer thread.Stop(context.Background(), nil)

	running := make(chan struct{})
	release := make(chan struct{})
	errCh := make(chan error, 1)
	go func() {
		errCh <- thread.Do(context.Background(), func() error {
			close(running)
			<-release
			return nil
		})
	}()
	<-running

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	err = thread.Do(ctx, func() error {
		t.Fatal("canceled job should not run")
		return nil
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("got %v, want %v", err, context.Canceled)
	}

	close(release)
	if err := <-errCh; err != nil {
		t.Fatal(err)
	}
}

func TestCallReturnsValue(t *testing.T) {
	thread, err := Start("test", nil)
	if err != nil {
		t.Fatal(err)
	}
	defer thread.Stop(context.Background(), nil)

	got, err := Call(context.Background(), thread, func() (int, error) {
		return 42, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if got != 42 {
		t.Fatalf("got %d, want 42", got)
	}
}

func TestDoRunsConcurrentlySubmittedWorkSerially(t *testing.T) {
	thread, err := Start("test", nil)
	if err != nil {
		t.Fatal(err)
	}
	defer thread.Stop(context.Background(), nil)

	oldProcs := runtime.GOMAXPROCS(8)
	defer runtime.GOMAXPROCS(oldProcs)

	const goroutines = 16
	const iterations = 64

	var active atomic.Int32
	var count atomic.Int64
	var wg sync.WaitGroup
	errCh := make(chan error, goroutines)

	for range goroutines {
		wg.Add(1)
		go func() {
			defer wg.Done()

			for range iterations {
				if err := thread.Do(context.Background(), func() error {
					if got := active.Add(1); got != 1 {
						return errors.New("thread executed jobs concurrently")
					}
					runtime.Gosched()
					count.Add(1)
					if got := active.Add(-1); got != 0 {
						return errors.New("thread active count did not return to zero")
					}
					return nil
				}); err != nil {
					errCh <- err
					return
				}
			}
		}()
	}

	wg.Wait()
	close(errCh)

	for err := range errCh {
		t.Fatal(err)
	}
	if got, want := count.Load(), int64(goroutines*iterations); got != want {
		t.Fatalf("got %d jobs, want %d", got, want)
	}
}

func TestStopRunsCleanupAndRejectsWork(t *testing.T) {
	thread, err := Start("test", nil)
	if err != nil {
		t.Fatal(err)
	}

	cleaned := 0
	if err := thread.Stop(context.Background(), func() {
		cleaned++
	}); err != nil {
		t.Fatal(err)
	}
	if cleaned != 1 {
		t.Fatalf("cleanup ran %d times, want 1", cleaned)
	}

	if err := thread.Stop(context.Background(), func() {
		cleaned++
	}); err != nil {
		t.Fatal(err)
	}
	if cleaned != 1 {
		t.Fatalf("cleanup ran %d times after second Stop, want 1", cleaned)
	}

	err = thread.Do(context.Background(), func() error {
		t.Fatal("job should not run after stop")
		return nil
	})
	if !errors.Is(err, ErrStopped) {
		t.Fatalf("got %v, want %v", err, ErrStopped)
	}
}

func TestStopCanceledBeforeEnqueueCanBeRetried(t *testing.T) {
	thread, err := Start("test", nil)
	if err != nil {
		t.Fatal(err)
	}
	defer thread.Stop(context.Background(), nil)

	running := make(chan struct{})
	release := make(chan struct{})
	errCh := make(chan error, 1)
	go func() {
		errCh <- thread.Do(context.Background(), func() error {
			close(running)
			<-release
			return nil
		})
	}()
	<-running

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()

	cleanupRan := false
	err = thread.Stop(ctx, func() {
		cleanupRan = true
	})
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("got %v, want %v", err, context.DeadlineExceeded)
	}
	if cleanupRan {
		t.Fatal("cleanup ran even though stop was not enqueued")
	}

	close(release)
	if err := <-errCh; err != nil {
		t.Fatal(err)
	}

	if err := thread.Do(context.Background(), func() error { return nil }); err != nil {
		t.Fatalf("thread did not accept work after canceled Stop: %v", err)
	}

	cleanupRan = false
	if err := thread.Stop(context.Background(), func() {
		cleanupRan = true
	}); err != nil {
		t.Fatal(err)
	}
	if !cleanupRan {
		t.Fatal("cleanup did not run on retried Stop")
	}
}

func TestStopWaitsForActiveWorkBeforeCleanup(t *testing.T) {
	thread, err := Start("test", nil)
	if err != nil {
		t.Fatal(err)
	}

	running := make(chan struct{})
	release := make(chan struct{})
	jobErr := make(chan error, 1)
	go func() {
		jobErr <- thread.Do(context.Background(), func() error {
			close(running)
			<-release
			return nil
		})
	}()
	<-running

	cleaned := make(chan struct{})
	stopErr := make(chan error, 1)
	go func() {
		stopErr <- thread.Stop(context.Background(), func() {
			close(cleaned)
		})
	}()

	select {
	case <-cleaned:
		t.Fatal("cleanup ran before active job completed")
	case <-time.After(10 * time.Millisecond):
	}

	err = thread.Do(context.Background(), func() error {
		return errors.New("work should be rejected once Stop starts")
	})
	if !errors.Is(err, ErrStopped) {
		t.Fatalf("got %v, want %v", err, ErrStopped)
	}

	close(release)
	if err := <-jobErr; err != nil {
		t.Fatal(err)
	}
	if err := <-stopErr; err != nil {
		t.Fatal(err)
	}

	select {
	case <-cleaned:
	default:
		t.Fatal("cleanup did not run")
	}
}
