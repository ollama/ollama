package launch

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"time"

	"github.com/ollama/ollama/cmd/config"
	"github.com/ollama/ollama/cmd/internal/fileutil"
	"github.com/ollama/ollama/internal/launchconfig"
)

// ChatGPTController applies the additive catalog to the regular ChatGPT profile.
// Settings and CLI provide the same controller; fixtures can replace it.
type ChatGPTController interface {
	Installed() bool
	OllamaConfigured() bool
	Running() bool
	UseOllamaFromDesktop(string, []LaunchModel, bool) error
	UpdateOllamaModelsFromDesktop(string, []LaunchModel, bool) error
	RestoreFromDesktop(bool) error
	RestartFromDesktop(bool) error
	Onboard() error
}

type ChatGPTModelLoader func(context.Context, []string) (string, []LaunchModel, error)

type ChatGPTApplyRequest struct {
	Models                    []string
	Primary                   string
	UseSavedSelection         bool
	OpenWhenStopped           bool
	SaveOnlyWhenDisconnected  bool
	RestartConfirmed          bool
	ConfirmRestart            func() (bool, error)
	RestartOnInventoryFailure bool
	Onboard                   bool
	ExpectedVersion           string
}

type chatGPTRestartConfirmationError struct{ version string }

func (e *chatGPTRestartConfirmationError) Error() string {
	return ErrCodexAppRestartConfirmationRequired.Error()
}
func (e *chatGPTRestartConfirmationError) Unwrap() error {
	return ErrCodexAppRestartConfirmationRequired
}
func (e *chatGPTRestartConfirmationError) ConfirmationVersion() string { return e.version }

func chatGPTConfigurationVersion(selection *config.IntegrationConfig, connected bool) string {
	models, primary := []string{}, ""
	if selection != nil {
		models, primary = selection.Models, selection.Primary
	}
	if len(models) == 0 {
		models = []string{}
	}
	data, _ := json.Marshal(struct {
		Models    []string
		Primary   string
		Connected bool
	}{models, primary, connected})
	return fmt.Sprintf("%x", sha256.Sum256(data))
}

func chatGPTModelsWithPrimaryAlias(primary string, models []LaunchModel) []LaunchModel {
	for i, model := range models {
		if model.Name != primary && ChatGPTModelKey(model.Name) == ChatGPTModelKey(primary) {
			models = cloneLaunchModels(models)
			models[i].Name = primary
			break
		}
	}
	return models
}

// LoadChatGPTConfiguration never replaces unreadable preferences with defaults.
func LoadChatGPTConfiguration() (*config.IntegrationConfig, error) {
	saved, err := launchconfig.LoadIntegration(chatGPTIntegrationName)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	return saved, err
}

func chatGPTConfigurationLock() (func(), error) {
	path, err := launchconfig.Path()
	if err != nil {
		return nil, err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, err
	}
	return fileutil.Lock(filepath.Join(filepath.Dir(path), "chatgpt-configuration.lock"))
}

// ResetChatGPTModels clears explicit preferences without connecting or opening
// ChatGPT. An empty selection continues to mean current account defaults.
func ResetChatGPTModels() error {
	unlock, err := chatGPTConfigurationLock()
	if err != nil {
		return err
	}
	defer unlock()
	previous, err := LoadChatGPTConfiguration()
	if err != nil || previous == nil {
		return err
	}
	return config.SwapPrimaryModelSelection(chatGPTIntegrationName, previous, &config.IntegrationConfig{})
}

// RestoreChatGPTModels serializes disconnect with shared configure/reset work.
// A stopped app stays stopped when disconnecting from Settings.
func RestoreChatGPTModels(controller ChatGPTController, restartConfirmed bool) error {
	unlock, err := chatGPTConfigurationLock()
	if err != nil {
		return err
	}
	defer unlock()
	return controller.RestoreFromDesktop(restartConfirmed)
}

func chatGPTCheckSelection(expected *config.IntegrationConfig) error {
	current, err := LoadChatGPTConfiguration()
	if err != nil {
		return err
	}
	if current == nil {
		current = &config.IntegrationConfig{}
	}
	if expected == nil {
		expected = &config.IntegrationConfig{}
	}
	if !slices.Equal(current.Models, expected.Models) || current.Primary != expected.Primary {
		return errors.New("ChatGPT model selection changed after this attempt; preserving the newer selection")
	}
	return nil
}

func chatGPTModelSelected(name string, models []LaunchModel) bool {
	key := ChatGPTModelKey(name)
	return key != "" && slices.ContainsFunc(models, func(model LaunchModel) bool { return ChatGPTModelKey(model.Name) == key })
}

// ApplyChatGPTModels validates before persisting or touching the app. Failed
// attempts restore only their own selection and leave later edits untouched.
func ApplyChatGPTModels(ctx context.Context, controller ChatGPTController, load ChatGPTModelLoader, request ChatGPTApplyRequest) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	unlock, err := chatGPTConfigurationLock()
	if err != nil {
		return err
	}
	defer func() {
		if unlock != nil {
			unlock()
		}
	}()
	previous, err := LoadChatGPTConfiguration()
	if err != nil {
		return err
	}
	selected := slices.Clone(request.Models)
	if request.UseSavedSelection && previous != nil {
		selected = slices.Clone(previous.Models)
	}
	previousModels, previousPrimary := []string(nil), ""
	if previous != nil {
		previousModels, previousPrimary = previous.Models, previous.Primary
	}
	if current, ok := controller.(interface{ CurrentModel() string }); ok && previousPrimary == "" {
		previousPrimary = current.CurrentModel()
	}
	wasConfigured := controller.OllamaConfigured()
	version := chatGPTConfigurationVersion(previous, wasConfigured)
	if request.ExpectedVersion != "" && request.ExpectedVersion != version {
		return errors.New("ChatGPT settings changed while awaiting restart confirmation; review the current models and try again")
	}
	restartRequired := func() error { return &chatGPTRestartConfirmationError{version: version} }
	selectionUnchanged := slices.Equal(selected, previousModels)
	confirmRestart := func() error {
		if !controller.Running() || request.RestartConfirmed {
			return nil
		}
		if request.ConfirmRestart == nil {
			return restartRequired()
		}
		// A CLI prompt can stay open indefinitely. Let Settings finish a later
		// operation, then reject this attempt if its saved state changed.
		unlock()
		unlock = nil
		confirmed, err := request.ConfirmRestart()
		if err != nil {
			return err
		}
		if !confirmed {
			return ErrCancelled
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		unlock, err = chatGPTConfigurationLock()
		if err != nil {
			return err
		}
		if err := chatGPTCheckSelection(previous); err != nil {
			return err
		}
		if controller.OllamaConfigured() != wasConfigured {
			return errors.New("ChatGPT connection changed while awaiting restart confirmation; try again")
		}
		request.RestartConfirmed = true
		return nil
	}
	if request.Primary != "" && len(selected) > 0 {
		found := false
		for _, name := range selected {
			found = found || ChatGPTModelKey(name) == ChatGPTModelKey(request.Primary)
		}
		if !found {
			selected = append(selected, request.Primary)
		}
	}
	primary, models, err := load(ctx, selected)
	if err != nil {
		if ctx.Err() == nil && request.RestartOnInventoryFailure && wasConfigured && selectionUnchanged && request.Primary == "" {
			if err := confirmRestart(); err != nil {
				return err
			}
			if err := ctx.Err(); err != nil {
				return err
			}
			if err := chatGPTCheckSelection(previous); err != nil {
				return err
			}
			if err := controller.RestartFromDesktop(request.RestartConfirmed); errors.Is(err, ErrCodexAppRestartConfirmationRequired) {
				return restartRequired()
			} else {
				return err
			}
		}
		return err
	}
	if request.Primary != "" && len(selected) == 0 && !chatGPTModelSelected(request.Primary, models) {
		// Keep defaults implicit when only their primary changes. A model
		// outside that list extends the resolved choices without dropping any.
		if len(models) >= ChatGPTMaxModels {
			return fmt.Errorf("ChatGPT already has %d default models; use 'ollama launch chatgpt --config' to choose up to %d models including %q", ChatGPTMaxModels, ChatGPTMaxModels, request.Primary)
		}
		selected = append(ChatGPTModelNames(models), request.Primary)
		primary, models, err = load(ctx, selected)
		if err != nil {
			return err
		}
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if len(models) == 0 || len(models) > ChatGPTMaxModels {
		return fmt.Errorf("choose between 1 and %d available Ollama models for ChatGPT", ChatGPTMaxModels)
	}
	if request.Primary == "" && selectionUnchanged && previous != nil && previous.Primary != "" && !chatGPTModelSelected(previous.Primary, models) {
		return fmt.Errorf("ChatGPT's saved starting model %q is no longer in the selected model list; update the model list or reset to defaults", previous.Primary)
	}
	preferred := request.Primary
	if preferred == "" && (len(previousModels) > 0 || previous != nil && previous.Primary != "") {
		preferred = previousPrimary
	}
	if preferred != "" {
		if !chatGPTModelSelected(preferred, models) && request.Primary != "" {
			return fmt.Errorf("ChatGPT primary model %q is not selected", preferred)
		}
		if chatGPTModelSelected(preferred, models) {
			primary = preferred
		}
	}
	// Keep the exact primary alias on its enriched entry. Otherwise the native
	// catalog writer can add a second fallback entry for the same cloud model.
	models = chatGPTModelsWithPrimaryAlias(primary, models)
	applied := &config.IntegrationConfig{Models: selected}
	if len(selected) > 0 || request.Primary != "" {
		applied.Primary = primary
	}
	if selectionUnchanged && request.Primary == "" && previous != nil {
		applied.Primary = previous.Primary
	}
	if request.SaveOnlyWhenDisconnected && !wasConfigured && !request.OpenWhenStopped {
		return config.SwapPrimaryModelSelection(chatGPTIntegrationName, previous, applied)
	}
	if err := confirmRestart(); err != nil {
		return err
	}
	if request.Onboard && !controller.Installed() {
		return errors.New("ChatGPT is not installed")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := config.SwapPrimaryModelSelection(chatGPTIntegrationName, previous, applied); err != nil {
		return fmt.Errorf("save ChatGPT models: %w", err)
	}
	rollbackSelection := func() error {
		return config.SwapPrimaryModelSelection(chatGPTIntegrationName, applied, previous)
	}
	if request.Onboard {
		if err := controller.Onboard(); err != nil {
			return errors.Join(fmt.Errorf("save ChatGPT integration state: %w", err), rollbackSelection())
		}
	}
	if err := ctx.Err(); err != nil {
		return errors.Join(err, rollbackSelection())
	}
	if err := chatGPTCheckSelection(applied); err != nil {
		return err
	}
	updateModels := controller.UpdateOllamaModelsFromDesktop
	if request.OpenWhenStopped {
		updateModels = controller.UseOllamaFromDesktop
	}
	applyErr := updateModels(primary, models, request.RestartConfirmed)
	if applyErr == nil {
		return nil
	}
	if err := rollbackSelection(); err != nil {
		// A newer selection may already own the live profile as well.
		return errors.Join(applyErr, fmt.Errorf("restore ChatGPT selection: %w", err))
	}
	if errors.Is(applyErr, ErrCodexAppRestartConfirmationRequired) {
		return restartRequired()
	}
	if errors.Is(applyErr, errCodexAppProfileUnchanged) {
		return applyErr
	}
	if !wasConfigured {
		if controller.OllamaConfigured() {
			if err := chatGPTCheckSelection(previous); err != nil {
				return errors.Join(applyErr, err)
			}
			if err := controller.RestoreFromDesktop(true); err != nil {
				return errors.Join(applyErr, fmt.Errorf("restore ChatGPT after failed update: %w", err))
			}
		}
		return fmt.Errorf("start ChatGPT with selected Ollama models: %w", applyErr)
	}
	rollbackCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 10*time.Second)
	defer cancel()
	rollbackPrimary, rollbackModels, rollbackErr := load(rollbackCtx, previousModels)
	if err := chatGPTCheckSelection(previous); err != nil {
		return errors.Join(applyErr, err)
	}
	if rollbackErr == nil {
		if previousPrimary != "" && !chatGPTModelSelected(previousPrimary, rollbackModels) {
			rollbackErr = fmt.Errorf("previous ChatGPT starting model %q is no longer available", previousPrimary)
		} else {
			if previousPrimary != "" {
				rollbackPrimary = previousPrimary
			}
			rollbackModels = chatGPTModelsWithPrimaryAlias(rollbackPrimary, rollbackModels)
			rollbackErr = updateModels(rollbackPrimary, rollbackModels, true)
		}
	}
	if rollbackErr != nil {
		if err := chatGPTCheckSelection(previous); err != nil {
			return errors.Join(applyErr, rollbackErr, err)
		}
		if restoreErr := controller.RestoreFromDesktop(true); restoreErr != nil {
			return errors.Join(applyErr, fmt.Errorf("restore previous Ollama profile: %w", rollbackErr), fmt.Errorf("restore normal ChatGPT profile: %w", restoreErr))
		}
		return fmt.Errorf("apply ChatGPT models: %v; restore previous Ollama profile: %v; restored the normal ChatGPT profile", applyErr, rollbackErr)
	}
	return fmt.Errorf("apply ChatGPT models: %w", applyErr)
}
