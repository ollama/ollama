//go:build darwin || windows

package cmd

import (
	"context"
	"errors"
	"time"

	"github.com/ollama/ollama/api"
)

func waitForServer(ctx context.Context, client *api.Client) error {
	waitCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	ticker := time.NewTicker(500 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-waitCtx.Done():
			if err := ctx.Err(); err != nil {
				return err
			}
			return errors.New("timed out waiting for server to start")
		case <-ticker.C:
			if err := client.Heartbeat(waitCtx); err == nil {
				return nil // server has started
			}
		}
	}
}
