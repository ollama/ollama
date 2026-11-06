//go:build darwin || windows

package cmd

import (
	"context"
	"errors"
	"time"

	"github.com/ollama/ollama/api"
)

func waitForServer(ctx context.Context, client *api.Client) error {
	// wait for the server to start
	timeout := time.NewTimer(5 * time.Second)
	defer timeout.Stop()
	tick := time.NewTicker(500 * time.Millisecond)
	defer tick.Stop()
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-timeout.C:
			return errors.New("timed out waiting for server to start")
		case <-tick.C:
			if err := client.Heartbeat(ctx); err == nil {
				return nil // server has started
			}
		}
	}
}
