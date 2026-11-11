package launch

import (
	"context"
	"errors"
	"fmt"
	"os"
	"time"

	"github.com/ollama/ollama/cmd/config"
)

// ChatGPTStarter is the desktop connection used by both the button and CLI.
type ChatGPTStarter interface {
	Installed() bool
	Running() bool
	OllamaConfigured() bool
	Onboard() error
	UseOllamaFromDesktop(string, []LaunchModel, bool) error
	RestoreFromDesktop(bool) error
}

// StartChatGPT reuses existing app choices or its current defaults. It does not
// save a model choice or a separate starting-model preference.
func StartChatGPT(ctx context.Context, app ChatGPTStarter, load func(context.Context, []string) (string, []LaunchModel, error), confirmed bool) error {
	if !app.Installed() {
		return errors.New("ChatGPT is not installed")
	}
	saved, err := config.LoadIntegration(chatGPTIntegrationName)
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	var selected []string
	if saved != nil {
		selected = saved.Models
	}
	primary, models, err := load(ctx, selected)
	if err != nil {
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if app.Running() && !confirmed {
		return ErrCodexAppRestartConfirmationRequired
	}
	if err := app.Onboard(); err != nil {
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	wasConfigured := app.OllamaConfigured()
	if err := app.UseOllamaFromDesktop(primary, models, confirmed); err != nil {
		if errors.Is(err, ErrCodexAppRestartConfirmationRequired) {
			return err
		}
		if !wasConfigured && app.OllamaConfigured() {
			return errors.Join(err, app.RestoreFromDesktop(true))
		}
		return err
	}
	return nil
}

var chatGPTModelSourceFactory = NewChatGPTModelSource

type chatGPTCLIStarter struct {
	*CodexApp
	start func(string, []LaunchModel, bool) error
}

func (c chatGPTCLIStarter) UseOllamaFromDesktop(primary string, models []LaunchModel, confirmed bool) error {
	return c.start(primary, models, confirmed)
}

func (c *launcherClient) launchChatGPT(ctx context.Context, app *CodexApp) error {
	source := chatGPTModelSourceFactory(c.apiClient)
	load := func(ctx context.Context, selected []string) (string, []LaunchModel, error) {
		loadCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
		defer cancel()
		return source.Resolve(loadCtx, selected, true)
	}
	starter := chatGPTCLIStarter{CodexApp: app, start: func(primary string, models []LaunchModel, confirmed bool) error {
		var cancelled <-chan struct{}
		if confirmed {
			spinner := StartSpinner(codexAppRestartMessage)
			defer spinner.Stop()
			cancelled = spinner.Cancelled()
		}
		processCtx, stop := codexAppExitContextFrom(ctx, codexAppCLIExitTimeout(), cancelled)
		defer stop()
		return app.updateOllamaModels(processCtx, primary, models, true, confirmed)
	}}
	err := StartChatGPT(ctx, starter, load, false)
	if errors.Is(err, ErrCodexAppRestartConfirmationRequired) {
		restorePolicy := withLaunchConfirmPolicy(c.policy.confirmPolicy())
		confirmed, confirmErr := ConfirmPromptWithOptions("Restart ChatGPT to add Ollama models?", ConfirmOptions{YesLabel: "Restart", NoLabel: "Cancel", Default: ConfirmDefaultNo})
		restorePolicy()
		if confirmErr != nil {
			return confirmErr
		}
		if !confirmed {
			return ErrCancelled
		}
		err = StartChatGPT(ctx, starter, load, true)
	}
	if err != nil {
		return fmt.Errorf("start ChatGPT: %w", err)
	}
	printConfigurationSuccess(app)
	return nil
}
