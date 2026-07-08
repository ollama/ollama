package launch

import (
	"context"
	"fmt"
	"os"
	"strconv"
)

type launchModelRunPreparer interface {
	prepareRunLaunchModels(context.Context, *launcherClient, string, []LaunchModel) ([]LaunchModel, error)
}

type launchModelConfigPreparer interface {
	prepareConfigLaunchModels(context.Context, *launcherClient, string, []LaunchModel) []LaunchModel
}

func (c *launcherClient) prepareLaunchModelsForRun(ctx context.Context, runner Runner, model string, models []LaunchModel) ([]LaunchModel, error) {
	if preparer, ok := runner.(launchModelRunPreparer); ok {
		return preparer.prepareRunLaunchModels(ctx, c, model, models)
	}
	return models, nil
}

func (c *launcherClient) prepareLaunchModelsForConfig(ctx context.Context, editor Editor, primary string, models []LaunchModel) []LaunchModel {
	if preparer, ok := editor.(launchModelConfigPreparer); ok {
		return preparer.prepareConfigLaunchModels(ctx, c, primary, models)
	}
	return models
}

func (c *launcherClient) localServerContextLength(ctx context.Context) (int, bool) {
	if c == nil || c.apiClient == nil {
		return 0, false
	}
	status, err := c.apiClient.CloudStatusExperimental(ctx)
	if err != nil || status.ContextLength <= 0 {
		return 0, false
	}
	return status.ContextLength, true
}

func confirmLocalContextWarning(integration string, current, recommended int) error {
	shortWarning := fmt.Sprintf(
		"Warning: %s works best with at least %s context; current local context is %s.",
		integration,
		formatContextLength(recommended),
		formatContextLength(current),
	)
	if currentLaunchConfirmPolicy.yes {
		fmt.Fprintf(os.Stderr, "%s\nContinuing because --yes was provided.\n", shortWarning)
		return nil
	}
	if currentLaunchConfirmPolicy.requireYesMessage {
		return fmt.Errorf("%s Re-run with --yes to continue", shortWarning)
	}

	prompt := fmt.Sprintf(
		"%s works best with at least %s context.\nCurrent local context: %s.\nAdjust Context length in Ollama Settings and restart to change this.\n\nContinue launching %s?",
		integration,
		formatContextLength(recommended),
		formatContextLength(current),
		integration,
	)
	ok, err := ConfirmPromptWithOptions(prompt, ConfirmOptions{
		YesLabel:  "Continue",
		NoLabel:   "Cancel",
		DefaultNo: true,
	})
	if err != nil {
		return err
	}
	if !ok {
		return ErrCancelled
	}
	return nil
}

func formatContextLength(tokens int) string {
	switch {
	case tokens <= 0:
		return strconv.Itoa(tokens)
	case tokens%1024 == 0:
		return strconv.Itoa(tokens/1024) + "k"
	case tokens%1000 == 0:
		return strconv.Itoa(tokens/1000) + "k"
	default:
		return strconv.Itoa(tokens)
	}
}
