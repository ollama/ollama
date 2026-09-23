package cmd

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/cmd/config"
	"github.com/ollama/ollama/cmd/launch"
	"github.com/ollama/ollama/cmd/tui"
	"github.com/spf13/cobra"
	"golang.org/x/term"
)

func runWelcome(ctx context.Context) error {
	if !term.IsTerminal(int(os.Stdin.Fd())) || !term.IsTerminal(int(os.Stdout.Fd())) {
		return nil
	}
	return ensureWelcome(func() error {
		return tui.RunWelcome(tui.WelcomeOptions{
			CheckAccount: func() tui.WelcomeAccount { return checkWelcomeAccount(ctx) },
			OpenBrowser:  launch.OpenBrowser,
			IsCompleted: func() bool {
				needed, err := config.NeedsWelcome()
				return err == nil && !needed
			},
		})
	})
}

func checkWelcomeAccount(ctx context.Context) tui.WelcomeAccount {
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	cmd := &cobra.Command{}
	cmd.SetContext(ctx)
	if err := checkServerHeartbeat(cmd, nil); err != nil {
		return tui.WelcomeAccount{Err: err}
	}
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return tui.WelcomeAccount{Err: err}
	}
	if status, err := client.CloudStatusExperimental(ctx); err == nil && status.Cloud.Disabled {
		return tui.WelcomeAccount{CloudDisabled: true}
	}
	user, err := client.Whoami(ctx)
	if err != nil {
		var authErr api.AuthorizationError
		if errors.As(err, &authErr) && authErr.StatusCode == http.StatusUnauthorized && authErr.SigninURL != "" {
			return tui.WelcomeAccount{SigninURL: authErr.SigninURL}
		}
		return tui.WelcomeAccount{Err: err}
	}
	if user != nil && strings.TrimSpace(user.Name) != "" {
		return tui.WelcomeAccount{SignedIn: true}
	}
	return tui.WelcomeAccount{Err: fmt.Errorf("could not verify the Ollama account")}
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
