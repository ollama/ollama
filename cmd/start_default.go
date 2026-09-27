//go:build !windows && !darwin

package cmd

import (
	"context"
	"errors"
	"github.com/ollama/ollama/i18n"

	"github.com/ollama/ollama/api"
)

func startApp(ctx context.Context, client *api.Client) error {
	return errors.New(i18n.T("could not connect to ollama server, run 'ollama serve' to start it"))
}
