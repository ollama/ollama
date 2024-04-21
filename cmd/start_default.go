//go:build !windows && !darwin

package cmd

import (
	"context"
	"fmt"

	"ollama.com/api"
)

func startApp(ctx context.Context, client *api.Client) error {
	return fmt.Errorf("could not connect to ollama server, run 'ollama serve' to start it")
}
