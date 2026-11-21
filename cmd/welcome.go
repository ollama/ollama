package cmd

import (
	"os"

	"github.com/ollama/ollama/cmd/config"
	"github.com/ollama/ollama/cmd/tui"
	"golang.org/x/term"
)

func runWelcome() error {
	if !term.IsTerminal(int(os.Stdin.Fd())) || !term.IsTerminal(int(os.Stdout.Fd())) {
		return nil
	}
	return ensureWelcome(func() error {
		return tui.RunWelcome(tui.WelcomeOptions{
			IsCompleted: func() bool {
				needed, err := config.NeedsWelcome()
				return err == nil && !needed
			},
		})
	})
}

func ensureWelcome(show func() error) error {
	needed, err := config.NeedsWelcome()
	if err != nil {
		return err
	}
	if !needed {
		return nil
	}
	if err := show(); err != nil {
		return err
	}
	return config.CompleteWelcome()
}
