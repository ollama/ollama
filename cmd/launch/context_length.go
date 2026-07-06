package launch

import (
	"context"
	"fmt"
	"os"
	"strconv"
)

func (c *launcherClient) prepareLaunchModelsForRun(ctx context.Context, runner Runner, model string, models []LaunchModel) ([]LaunchModel, error) {
	switch runner.(type) {
	case *Claude:
		return c.prepareClaudeLaunchModels(ctx, runner, model, models)
	case *OpenCode:
		return c.prepareOpenCodeLaunchModels(ctx, runner, model, models, true)
	default:
		return models, nil
	}
}

func (c *launcherClient) prepareLaunchModelsForConfig(ctx context.Context, editor Editor, primary string, models []LaunchModel) []LaunchModel {
	switch editor := editor.(type) {
	case *OpenCode:
		prepared, _ := c.prepareOpenCodeLaunchModels(ctx, editor, primary, models, false)
		return prepared
	default:
		return models
	}
}

func (c *launcherClient) prepareClaudeLaunchModels(ctx context.Context, runner Runner, model string, models []LaunchModel) ([]LaunchModel, error) {
	if model == "" || isCloudModelName(model) {
		return models, nil
	}

	contextLength, ok := c.localServerContextLength(ctx)
	if !ok {
		return models, nil
	}

	models = launchModelsWithContextLength(model, models, contextLength)
	if contextLength >= claudeCodeAutoCompactMinContext {
		return models, nil
	}

	if err := confirmLocalContextWarning(runner.String(), contextLength, claudeCodeAutoCompactMinContext); err != nil {
		return nil, err
	}
	return models, nil
}

const openCodeRecommendedContext = 64 * 1024

func (c *launcherClient) prepareOpenCodeLaunchModels(ctx context.Context, integration fmt.Stringer, primary string, models []LaunchModel, warn bool) ([]LaunchModel, error) {
	if primary == "" {
		return models, nil
	}

	if !hasLocalLaunchModel(primary, models) {
		return models, nil
	}

	contextLength, ok := c.localServerContextLength(ctx)
	if !ok {
		return models, nil
	}

	models = launchModelsWithOpenCodeLocalLimits(primary, models, contextLength)
	if warn && !isCloudModelName(primary) && contextLength < openCodeRecommendedContext {
		if err := confirmLocalContextWarning(integration.String(), contextLength, openCodeRecommendedContext); err != nil {
			return nil, err
		}
	}
	return models, nil
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

func launchModelsWithContextLength(primary string, models []LaunchModel, contextLength int) []LaunchModel {
	if contextLength <= 0 {
		return models
	}
	if len(models) == 0 && primary != "" {
		models = launchModelsFromNames([]string{primary})
	}

	out := cloneLaunchModels(models)
	for i := range out {
		if launchModelMatches(out[i].Name, primary) {
			out[i].ContextLength = contextLength
			return out
		}
	}

	if primary != "" {
		model := fallbackLaunchModel(primary)
		model.ContextLength = contextLength
		out = append([]LaunchModel{model}, out...)
	}
	return out
}

func launchModelsWithOpenCodeLocalLimits(primary string, models []LaunchModel, contextLength int) []LaunchModel {
	if contextLength <= 0 {
		return models
	}
	if len(models) == 0 && primary != "" {
		models = launchModelsFromNames([]string{primary})
	}

	out := cloneLaunchModels(models)
	for i := range out {
		if isCloudModelName(out[i].Name) {
			continue
		}
		out[i].ContextLength = contextLength
		out[i].MaxOutputTokens = openCodeLocalMaxOutputTokens(contextLength)
	}
	if primary != "" && !isCloudModelName(primary) && !hasLaunchModel(out, primary) {
		model := fallbackLaunchModel(primary)
		model.ContextLength = contextLength
		model.MaxOutputTokens = openCodeLocalMaxOutputTokens(contextLength)
		out = append([]LaunchModel{model}, out...)
	}
	return out
}

func openCodeLocalMaxOutputTokens(contextLength int) int {
	return min(8192, max(2048, contextLength/4))
}

func hasLocalLaunchModel(primary string, models []LaunchModel) bool {
	if primary != "" && !isCloudModelName(primary) {
		return true
	}
	for _, model := range models {
		if model.Name != "" && !isCloudModelName(model.Name) {
			return true
		}
	}
	return false
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
