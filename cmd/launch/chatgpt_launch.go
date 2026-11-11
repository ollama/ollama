package launch

import (
	"context"
	"errors"
	"fmt"
	"time"
)

var chatGPTModelSourceFactory = NewChatGPTModelSource

// CLI operations keep the CLI's graceful quit wait and Ctrl+C handling while
// sharing configuration and profile writes with Settings.
type chatGPTCLIController struct{ *CodexApp }

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

func (c chatGPTCLIController) UseOllamaFromDesktop(primary string, models []LaunchModel, confirmed bool) error {
	return chatGPTCLIProfileOperation(confirmed, func(ctx context.Context) error {
		return c.updateOllamaModels(ctx, primary, models, true, confirmed)
	})
}

func (c chatGPTCLIController) UpdateOllamaModelsFromDesktop(primary string, models []LaunchModel, confirmed bool) error {
	return chatGPTCLIProfileOperation(confirmed, func(ctx context.Context) error {
		return c.updateOllamaModels(ctx, primary, models, false, confirmed)
	})
}

func (c chatGPTCLIController) RestartFromDesktop(confirmed bool) error {
	return chatGPTCLIProfileOperation(confirmed, func(ctx context.Context) error {
		return c.restart(ctx, confirmed)
	})
}

func (c *launcherClient) launchChatGPTIntegration(ctx context.Context, controller ChatGPTController, req IntegrationLaunchRequest) error {
	if len(req.ExtraArgs) > 0 {
		return errors.New("chatgpt does not accept extra arguments")
	}
	if app, ok := controller.(*CodexApp); ok {
		controller = chatGPTCLIController{app}
	}
	source := chatGPTModelSourceFactory(c.apiClient)
	request := ChatGPTApplyRequest{
		UseSavedSelection:         true,
		Primary:                   req.ModelOverride,
		OpenWhenStopped:           !req.ConfigureOnly,
		Onboard:                   true,
		RestartOnInventoryFailure: !req.ConfigureOnly && !req.ForceConfigure && req.ModelOverride == "",
	}
	if req.ForceConfigure && req.ModelOverride == "" {
		saved, err := LoadChatGPTConfiguration()
		if err != nil {
			return err
		}
		inventoryCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
		inventory, err := source.Inventory(inventoryCtx)
		cancel()
		if err != nil {
			return err
		}
		checked := ChatGPTModelNames(ChatGPTDefaultModels(inventory))
		if saved != nil && len(saved.Models) > 0 {
			checked = saved.Models
		}
		items := make([]SelectionItem, 0, len(inventory.Available))
		seen := make(map[string]bool)
		for _, model := range inventory.Catalog {
			name := ChatGPTSelectionModelName(model.Model.Name, checked)
			seen[ChatGPTModelKey(name)] = true
			items = append(items, SelectionItem{Name: name, Description: model.Description, Recommended: model.Recommended})
		}
		for _, name := range checked {
			if !seen[ChatGPTModelKey(name)] {
				items = append(items, SelectionItem{Name: name, Description: "No longer available; remove it to choose another model", AvailabilityBadge: "Unavailable"})
				seen[ChatGPTModelKey(name)] = true
			}
		}
		title := fmt.Sprintf("Select up to %d models for ChatGPT:", ChatGPTMaxModels)
		if DefaultModelListSelector != nil {
			request.Models, err = DefaultModelListSelector(title, items, checked, ChatGPTMaxModels)
		} else {
			request.Models, err = runMultiSelector(title, items, checked, nil)
		}
		if err != nil {
			return err
		}
		request.UseSavedSelection = false
	}
	// The shared flow requests consent after validation and before any writes.
	request.ConfirmRestart = func() (bool, error) {
		restorePolicy := withLaunchConfirmPolicy(c.policy.confirmPolicy())
		defer restorePolicy()
		confirmed, err := ConfirmPromptWithOptions("Restart ChatGPT to add Ollama models?", ConfirmOptions{YesLabel: "Restart", NoLabel: "Cancel", Default: ConfirmDefaultNo})
		return confirmed, err
	}
	reconnect := !controller.OllamaConfigured() && !req.ForceConfigure && req.ModelOverride == ""
	load := func(ctx context.Context, selected []string) (string, []LaunchModel, error) {
		loadCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
		defer cancel()
		return source.Resolve(loadCtx, selected, reconnect)
	}
	if err := ApplyChatGPTModels(ctx, controller, load, request); err != nil {
		return err
	}
	printConfigurationSuccess(controller)
	return nil
}
