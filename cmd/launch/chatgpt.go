package launch

import (
	"context"
	"errors"
	"fmt"
	"os"
	"slices"
	"time"

	"github.com/ollama/ollama/cmd/config"
)

func chatGPTSavedSelection() (*config.IntegrationConfig, error) {
	saved, err := config.LoadIntegration(chatGPTIntegrationName)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	return saved, err
}

func chatGPTSelectionEqual(a, b *config.IntegrationConfig) bool {
	if a == nil {
		a = &config.IntegrationConfig{}
	}
	if b == nil {
		b = &config.IntegrationConfig{}
	}
	return slices.Equal(a.Models, b.Models) && a.Primary == b.Primary
}

func chatGPTCLIProfileOperation(confirmed bool, operation func(context.Context) error) error {
	var cancelled <-chan struct{}
	if confirmed {
		sp := StartSpinner(codexAppRestartMessage)
		defer sp.Stop()
		cancelled = sp.Cancelled()
	}
	ctx, stop := codexAppExitContext(codexAppCLIExitTimeout(), cancelled)
	defer stop()
	return operation(ctx)
}

// Keep the app's selected list separate from the CLI's starting-model choice.
// Native picker choices are never written back into these preferences.
func (c *launcherClient) launchChatGPT(ctx context.Context, app *CodexApp, saved *config.IntegrationConfig, req IntegrationLaunchRequest) error {
	if len(req.ExtraArgs) > 0 {
		return errors.New("chatgpt does not accept extra arguments")
	}
	wasConfigured, current := app.OllamaConfigured(), app.CurrentModel()
	previousModels, _ := codexAppRoutingModels()
	selected, primary := []string(nil), ""
	if saved != nil {
		selected, primary = slices.Clone(saved.Models), saved.Primary
	}
	if len(selected) > 5 {
		return errors.New("choose up to five ChatGPT models in Ollama Settings")
	}
	if primary == "" && len(selected) > 0 {
		primary = selected[0]
		if slices.ContainsFunc(selected, func(name string) bool {
			return launchModelRecommendationKey(name) == launchModelRecommendationKey(current)
		}) {
			primary = current
		}
	}
	if primary == "" {
		primary = current
	}
	if req.ModelOverride != "" || req.ForceConfigure || primary == "" {
		var err error
		primary, _, err = c.resolveSingleIntegrationTarget(ctx, chatGPTIntegrationName, app, primary, req)
		if err != nil || primary == "" {
			return err
		}
	}
	models := slices.Clone(selected)
	if len(models) == 0 {
		models = slices.Clone(previousModels)
	}
	contains := slices.ContainsFunc(models, func(name string) bool {
		return launchModelRecommendationKey(name) == launchModelRecommendationKey(primary)
	})
	if !contains {
		if len(selected) > 0 && req.ModelOverride == "" && !req.ForceConfigure {
			return errors.New("ChatGPT's saved starting model is not in its model list; update the selection in Ollama Settings")
		}
		if len(models) >= 5 {
			return errors.New("ChatGPT already has five Ollama models; change the model list in Ollama Settings first")
		}
		models = append(models, primary)
		if len(selected) > 0 {
			selected = slices.Clone(models)
		}
	}
	if saved == nil || len(selected) == 0 && (req.ModelOverride != "" || req.ForceConfigure) {
		// An explicit CLI choice keeps the app's current default list too,
		// so disconnect/reconnect can restore it without rediscovery.
		selected = slices.Clone(models)
	}
	for i := range models {
		if launchModelRecommendationKey(models[i]) == launchModelRecommendationKey(primary) {
			// Preserve the saved alias on its single enriched catalog entry.
			models[i] = primary
		}
	}
	applied := &config.IntegrationConfig{Models: selected, Primary: primary}
	if len(selected) == 0 && req.ModelOverride == "" && !req.ForceConfigure {
		applied.Primary = ""
		if saved != nil {
			applied.Primary = saved.Primary
		}
	}
	// Use the existing catalog when its selection already matches. Ordinary
	// relaunch does not depend on rediscovering model metadata.
	unchanged := primary == current && len(models) == len(previousModels)
	for i := range models {
		unchanged = unchanged && i < len(previousModels) && launchModelRecommendationKey(models[i]) == launchModelRecommendationKey(previousModels[i])
	}
	var resolved []LaunchModel
	if !unchanged || req.ConfigureOnly {
		resolved = c.resolveRunModels(ctx, chatGPTIntegrationName, models)
		for i := range resolved {
			if launchModelRecommendationKey(resolved[i].Name) == launchModelRecommendationKey(primary) {
				resolved[i].Name = primary
			}
		}
	}
	confirmed := false
	if app.Running() {
		restorePolicy := withLaunchConfirmPolicy(c.policy.confirmPolicy())
		var err error
		confirmed, err = ConfirmPromptWithOptions("Restart ChatGPT to add Ollama models?", ConfirmOptions{YesLabel: "Restart", NoLabel: "Cancel", Default: ConfirmDefaultNo})
		restorePolicy()
		if err != nil {
			return err
		}
		if !confirmed {
			return ErrCancelled
		}
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	unlock, err := config.LockChatGPTConfiguration()
	if err != nil {
		return err
	}
	defer unlock()
	if err := ctx.Err(); err != nil {
		return err
	}
	latest, err := chatGPTSavedSelection()
	if err != nil {
		return err
	}
	if !chatGPTSelectionEqual(saved, latest) || app.OllamaConfigured() != wasConfigured {
		return errors.New("ChatGPT settings changed while choosing a model or confirming restart; try again")
	}
	if err := app.Onboard(); err != nil {
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := config.SwapPrimaryModelSelection(chatGPTIntegrationName, saved, applied); err != nil {
		return err
	}
	err = chatGPTCLIProfileOperation(confirmed, func(processCtx context.Context) error {
		if unchanged && !req.ConfigureOnly {
			return app.restart(processCtx, confirmed)
		}
		return app.updateOllamaModels(processCtx, primary, resolved, !req.ConfigureOnly, confirmed)
	})
	if err == nil {
		printConfigurationSuccess(app)
		return nil
	}
	if rollbackErr := config.SwapPrimaryModelSelection(chatGPTIntegrationName, applied, saved); rollbackErr != nil {
		return errors.Join(err, rollbackErr)
	}
	if unchanged && !req.ConfigureOnly || errors.Is(err, ErrCodexAppProfileUnchanged) {
		return err
	}
	if !wasConfigured {
		if app.OllamaConfigured() {
			return errors.Join(err, app.RestoreFromDesktop(true))
		}
		return err
	}
	// A failed native write uses the existing profile recovery path. Never
	// overwrite a newer saved selection while recovering this attempt.
	rollbackCtx, stop := context.WithTimeout(context.WithoutCancel(ctx), 10*time.Second)
	defer stop()
	rollbackModels := c.resolveRunModels(rollbackCtx, chatGPTIntegrationName, previousModels)
	if rollbackErr := app.UpdateOllamaModelsFromDesktop(current, rollbackModels, true); rollbackErr != nil {
		return errors.Join(err, fmt.Errorf("restore previous ChatGPT models: %w", rollbackErr), app.RestoreFromDesktop(true))
	}
	return err
}
