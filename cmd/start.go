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
	timeout := time.After(5 * time.Second)
	tick := time.Tick(500 * time.Millisecond)
	for {
		select {
		case <-timeout:
			return errors.New("timed out waiting for server to start")
		case <-tick:
			if err := client.Heartbeat(ctx); err == nil {
				return nil // server has started
			}
		}
	}
}
