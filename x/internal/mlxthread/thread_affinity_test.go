//go:build darwin || linux

package mlxthread

import (
	"context"
	"fmt"
	"testing"
)

func TestDoUsesSameOSThread(t *testing.T) {
	thread, err := Start("test", nil)
	if err != nil {
		t.Fatal(err)
	}
	defer thread.Stop(context.Background(), nil)

	var first uint64
	for i := 0; i < 32; i++ {
		if err := thread.Do(context.Background(), func() error {
			id := currentThreadID()
			if first == 0 {
				first = id
			} else if id != first {
				return fmt.Errorf("job ran on OS thread %d, want %d", id, first)
			}
			return nil
		}); err != nil {
			t.Fatal(err)
		}
	}
}
