package launch

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/cmd/config"
	"github.com/spf13/cobra"
)

// IntegrationInfo re-exports config integration display metadata for the TUI.
type IntegrationInfo = config.IntegrationInfo

// LauncherState is the launch-owned snapshot used to render the root launcher menu.
type LauncherState struct {
	LastSelection  string
	RunModel       string
	RunModelUsable bool
	Integrations   map[string]LauncherIntegrationState
}

// LauncherIntegrationState is the launch-owned status for one launcher integration.
type LauncherIntegrationState struct {
	Name            string
	DisplayName     string
	Description     string
	Installed       bool
	AutoInstallable bool
	Selectable      bool
	Changeable      bool
	CurrentModel    string
	ModelUsable     bool
	InstallHint     string
	Editor          bool
}

// RunModelRequest controls how the root launcher resolves the chat model.
type RunModelRequest struct {
	ForcePicker bool
}

// IntegrationLaunchRequest controls the canonical integration launcher flow.
type IntegrationLaunchRequest struct {
	Name           string
	ModelOverride  string
	ForceConfigure bool
	ConfigureOnly  bool
	ExtraArgs      []string
}

// LauncherInvocation carries one-shot root launcher overrides derived from CLI flags.
type LauncherInvocation struct {
	ModelOverride string
	ExtraArgs     []string
}

// ListIntegrationInfos returns the registered integrations in launcher display order.
func ListIntegrationInfos() []IntegrationInfo {
	return config.ListIntegrationInfos()
}

// LaunchIntegrationByName launches the named integration using saved config or prompts for setup.
func LaunchIntegrationByName(name string) error {
	return LaunchIntegration(context.Background(), IntegrationLaunchRequest{Name: name})
}

// LaunchIntegrationWithModel launches the named integration with the specified model.
func LaunchIntegrationWithModel(name, modelName string) error {
	return LaunchIntegration(context.Background(), IntegrationLaunchRequest{
		Name:          name,
		ModelOverride: modelName,
	})
}

// SaveAndEditIntegration saves the models for an integration and, when supported,
// runs its Edit method to write any integration-managed config files.
func SaveAndEditIntegration(name string, models []string) error {
	key, runner, err := config.LookupIntegration(name)
	if err != nil {
		return err
	}

	editor, ok := runner.(config.Editor)
	if !ok {
		return config.SaveIntegration(key, models)
	}

	return config.PrepareEditorIntegration(key, runner, editor, models)
}

// ConfigureIntegrationWithSelectors allows the user to select/change the model for an integration using custom selectors.
func ConfigureIntegrationWithSelectors(ctx context.Context, name string, single config.SingleSelector, multi config.MultiSelector) error {
	oldSingle := config.DefaultSingleSelector
	oldMulti := config.DefaultMultiSelector
	if single != nil {
		config.DefaultSingleSelector = single
	}
	if multi != nil {
		config.DefaultMultiSelector = multi
	}
	defer func() {
		config.DefaultSingleSelector = oldSingle
		config.DefaultMultiSelector = oldMulti
	}()

	return LaunchIntegration(ctx, IntegrationLaunchRequest{
		Name:           name,
		ForceConfigure: true,
		ConfigureOnly:  true,
	})
}

// ConfigureIntegration allows the user to select/change the model for an integration.
func ConfigureIntegration(ctx context.Context, name string) error {
	return LaunchIntegration(ctx, IntegrationLaunchRequest{
		Name:           name,
		ForceConfigure: true,
		ConfigureOnly:  true,
	})
}

// LaunchCmd returns the cobra command for launching integrations.
// The runTUI callback is called when the root launcher UI should be shown.
func LaunchCmd(checkServerHeartbeat func(cmd *cobra.Command, args []string) error, runTUI func(cmd *cobra.Command, inv LauncherInvocation)) *cobra.Command {
	var modelFlag string
	var configFlag bool

	cmd := &cobra.Command{
		Use:   "launch [INTEGRATION] [-- [EXTRA_ARGS...]]",
		Short: "Launch the Ollama menu or an integration",
		Long: `Launch the Ollama interactive menu, or directly launch a specific integration.

Without arguments, this is equivalent to running 'ollama' directly.

Supported integrations:
  claude    Claude Code
  cline     Cline
  codex     Codex
  droid     Droid
  opencode  OpenCode
  openclaw  OpenClaw (aliases: clawdbot, moltbot)
  pi        Pi

Examples:
  ollama launch
  ollama launch claude
  ollama launch claude --model <model>
  ollama launch droid --config (does not auto-launch)
  ollama launch codex -- -p myprofile (pass extra args to integration)
  ollama launch codex -- --sandbox workspace-write`,
		Args:    cobra.ArbitraryArgs,
		PreRunE: checkServerHeartbeat,
		RunE: func(cmd *cobra.Command, args []string) error {
			if len(args) == 0 && modelFlag == "" && !configFlag {
				runTUI(cmd, LauncherInvocation{})
				return nil
			}

			var name string
			var passArgs []string
			dashIdx := cmd.ArgsLenAtDash()

			if dashIdx == -1 {
				if len(args) > 1 {
					return fmt.Errorf("unexpected arguments: %v\nUse '--' to pass extra arguments to the integration", args[1:])
				}
				if len(args) == 1 {
					name = args[0]
				}
			} else {
				if dashIdx > 1 {
					return fmt.Errorf("expected at most 1 integration name before '--', got %d", dashIdx)
				}
				if dashIdx == 1 {
					name = args[0]
				}
				passArgs = args[dashIdx:]
			}

			if name == "" && modelFlag != "" && !configFlag {
				runTUI(cmd, LauncherInvocation{
					ModelOverride: modelFlag,
					ExtraArgs:     append([]string(nil), passArgs...),
				})
				return nil
			}

			if name == "" {
				var err error
				name, err = SelectIntegration()
				if errors.Is(err, config.ErrCancelled) {
					return nil
				}
				if err != nil {
					return err
				}
			}

			err := LaunchIntegration(cmd.Context(), IntegrationLaunchRequest{
				Name:           name,
				ModelOverride:  modelFlag,
				ForceConfigure: configFlag || modelFlag == "",
				ConfigureOnly:  configFlag,
				ExtraArgs:      passArgs,
			})
			if errors.Is(err, config.ErrCancelled) {
				return nil
			}
			return err
		},
	}

	cmd.Flags().StringVar(&modelFlag, "model", "", "Model to use")
	cmd.Flags().BoolVar(&configFlag, "config", false, "Configure without launching")
	return cmd
}

type launcherClient struct {
	apiClient         *api.Client
	modelInventory    []config.ModelInfo
	cloudDisabled     bool
	cloudStatusLoaded bool
	inventoryLoaded   bool
}

func newLauncherClient() (*launcherClient, error) {
	apiClient, err := api.ClientFromEnvironment()
	if err != nil {
		return nil, err
	}

	return &launcherClient{
		apiClient: apiClient,
	}, nil
}

// BuildLauncherState returns the launch-owned root launcher menu snapshot.
func BuildLauncherState(ctx context.Context) (*LauncherState, error) {
	launchClient, err := newLauncherClient()
	if err != nil {
		return nil, err
	}
	return launchClient.buildLauncherState(ctx)
}

// ResolveRunModel returns the model that should be used for interactive chat.
func ResolveRunModel(ctx context.Context, req RunModelRequest) (string, error) {
	launchClient, err := newLauncherClient()
	if err != nil {
		return "", err
	}
	return launchClient.resolveRunModel(ctx, req)
}

// ResolveRequestedRunModel validates and persists an explicitly requested chat model.
func ResolveRequestedRunModel(ctx context.Context, model string) (string, error) {
	launchClient, err := newLauncherClient()
	if err != nil {
		return "", err
	}
	return launchClient.resolveRequestedRunModel(ctx, model)
}

// LaunchIntegration runs the canonical launcher flow for one integration.
func LaunchIntegration(ctx context.Context, req IntegrationLaunchRequest) error {
	name, runner, err := config.LookupIntegration(req.Name)
	if err != nil {
		return err
	}

	launchClient, err := newLauncherClient()
	if err != nil {
		return err
	}
	saved, _ := config.LoadIntegration(name)

	if aliasConfigurer, ok := runner.(config.AliasConfigurer); ok {
		return launchClient.launchAliasConfiguredIntegration(ctx, name, runner, aliasConfigurer, saved, req)
	}
	if editor, ok := runner.(config.Editor); ok {
		return launchClient.launchEditorIntegration(ctx, name, runner, editor, saved, req)
	}
	return launchClient.launchSingleIntegration(ctx, name, runner, saved, req)
}

// SelectIntegration lets the user choose which integration to launch.
func SelectIntegration() (string, error) {
	if config.DefaultSingleSelector == nil {
		return "", fmt.Errorf("no selector configured")
	}

	items, err := config.IntegrationSelectionItems()
	if err != nil {
		return "", err
	}

	return config.DefaultSingleSelector("Select integration:", items, "")
}

func (c *launcherClient) buildLauncherState(ctx context.Context) (*LauncherState, error) {
	if err := c.loadModelInventoryOnce(ctx); err != nil {
		return nil, err
	}

	state := &LauncherState{
		LastSelection: config.LastSelection(),
		RunModel:      config.LastModel(),
		Integrations:  make(map[string]LauncherIntegrationState),
	}
	runModelUsable, err := c.savedModelUsable(ctx, state.RunModel)
	if err != nil {
		return nil, err
	}
	state.RunModelUsable = runModelUsable

	for _, info := range config.ListIntegrationInfos() {
		integrationState, err := c.buildLauncherIntegrationState(ctx, info)
		if err != nil {
			return nil, err
		}
		state.Integrations[info.Name] = integrationState
	}

	return state, nil
}

func (c *launcherClient) buildLauncherIntegrationState(ctx context.Context, info config.IntegrationInfo) (LauncherIntegrationState, error) {
	installed := config.IsIntegrationInstalled(info.Name)
	autoInstallable := config.AutoInstallable(info.Name)
	isEditor := config.IsEditorIntegration(info.Name)
	currentModel, usable, err := c.launcherModelState(ctx, info.Name, isEditor)
	if err != nil {
		return LauncherIntegrationState{}, err
	}

	return LauncherIntegrationState{
		Name:            info.Name,
		DisplayName:     info.DisplayName,
		Description:     info.Description,
		Installed:       installed,
		AutoInstallable: autoInstallable,
		Selectable:      installed || autoInstallable,
		Changeable:      installed || autoInstallable,
		CurrentModel:    currentModel,
		ModelUsable:     usable,
		InstallHint:     config.IntegrationInstallHint(info.Name),
		Editor:          isEditor,
	}, nil
}

func (c *launcherClient) launcherModelState(ctx context.Context, name string, isEditor bool) (string, bool, error) {
	cfg, err := config.LoadIntegration(name)
	if err != nil || len(cfg.Models) == 0 {
		return "", false, nil
	}

	if isEditor {
		filtered := c.filterDisabledCloudModels(ctx, cfg.Models)
		if len(filtered) > 0 {
			return filtered[0], true, nil
		}
		return cfg.Models[0], false, nil
	}

	model := cfg.Models[0]
	usable, err := c.savedModelUsable(ctx, model)
	if err != nil {
		return "", false, err
	}
	return model, usable, nil
}

func (c *launcherClient) resolveRunModel(ctx context.Context, req RunModelRequest) (string, error) {
	current := config.LastModel()
	if !req.ForcePicker {
		usable, err := c.savedModelUsable(ctx, current)
		if err != nil {
			return "", err
		}
		if usable {
			if err := c.ensureModelsReady(ctx, []string{current}); err != nil {
				return "", err
			}
			if err := config.SetLastModel(current); err != nil {
				return "", err
			}
			return current, nil
		}
	}

	model, err := c.selectSingleModelWithSelector(ctx, "Select model to run:", current, config.DefaultSingleSelector)
	if err != nil {
		return "", err
	}
	if err := config.SetLastModel(model); err != nil {
		return "", err
	}
	return model, nil
}

func (c *launcherClient) resolveRequestedRunModel(ctx context.Context, model string) (string, error) {
	if err := c.ensureModelsReady(ctx, []string{model}); err != nil {
		return "", err
	}
	if err := config.SetLastModel(model); err != nil {
		return "", err
	}
	return model, nil
}

func (c *launcherClient) launchSingleIntegration(ctx context.Context, name string, runner config.Runner, saved *config.IntegrationConfig, req IntegrationLaunchRequest) error {
	current := primaryModelFromConfig(saved)
	target := req.ModelOverride
	needsConfigure := req.ForceConfigure

	if target == "" {
		target = current
		usable, err := c.savedModelUsable(ctx, target)
		if err != nil {
			return err
		}
		if !usable {
			needsConfigure = true
		}
	}

	if needsConfigure {
		selected, err := c.selectSingleModelWithSelector(ctx, fmt.Sprintf("Select model for %s:", runner), target, config.DefaultSingleSelector)
		if err != nil {
			return err
		}
		target = selected
	} else if err := c.ensureModelsReady(ctx, []string{target}); err != nil {
		return err
	}

	if target == "" {
		return nil
	}

	if err := config.SaveIntegration(name, []string{target}); err != nil {
		return fmt.Errorf("failed to save: %w", err)
	}

	return launchAfterConfiguration(name, runner, target, req)
}

func (c *launcherClient) launchEditorIntegration(ctx context.Context, name string, runner config.Runner, editor config.Editor, saved *config.IntegrationConfig, req IntegrationLaunchRequest) error {
	models, needsConfigure := c.resolveEditorLaunchModels(ctx, saved, req)

	if needsConfigure {
		selected, err := c.selectMultiModelsForIntegration(ctx, runner, models)
		if err != nil {
			return err
		}
		models = selected
	} else if err := c.ensureModelsReady(ctx, models); err != nil {
		return err
	}

	if len(models) == 0 {
		return nil
	}

	if needsConfigure || req.ModelOverride != "" {
		if err := config.PrepareEditorIntegration(name, runner, editor, models); err != nil {
			return err
		}
	}

	return launchAfterConfiguration(name, runner, models[0], req)
}

func (c *launcherClient) launchAliasConfiguredIntegration(ctx context.Context, name string, runner config.Runner, aliases config.AliasConfigurer, saved *config.IntegrationConfig, req IntegrationLaunchRequest) error {
	primary := req.ModelOverride
	var existingAliases map[string]string
	if saved != nil {
		existingAliases = saved.Aliases
		if primary == "" {
			primary = primaryModelFromConfig(saved)
		}
	}

	forceConfigure := req.ForceConfigure
	if primary == "" {
		forceConfigure = true
	} else {
		usable, err := c.savedModelUsable(ctx, primary)
		if err != nil {
			return err
		}
		if !usable {
			forceConfigure = true
		}
	}

	resolvedAliases := cloneAliases(existingAliases)
	if forceConfigure || primary != "" {
		var changed bool
		var err error
		resolvedAliases, changed, err = aliases.ConfigureAliases(ctx, primary, existingAliases, forceConfigure)
		if err != nil {
			return err
		}
		if changed || primary == "" {
			primary = resolvedAliases["primary"]
		}
	}

	if primary == "" {
		return nil
	}

	if err := c.ensureModelsReady(ctx, []string{primary}); err != nil {
		return err
	}

	if err := syncAliases(ctx, c.apiClient, aliases, name, primary, resolvedAliases); err != nil {
		fmt.Fprintf(os.Stderr, "Warning: could not sync aliases: %v\n", err)
	}
	if err := config.SaveAliases(name, normalizedAliases(primary, resolvedAliases)); err != nil {
		return fmt.Errorf("failed to save aliases: %w", err)
	}
	if err := config.SaveIntegration(name, []string{primary}); err != nil {
		return fmt.Errorf("failed to save: %w", err)
	}

	return launchAfterConfiguration(name, runner, primary, req)
}

func (c *launcherClient) selectSingleModelWithSelector(ctx context.Context, title, current string, selector config.SingleSelector) (string, error) {
	if selector == nil {
		return "", fmt.Errorf("no selector configured")
	}

	items, _, err := c.loadSelectableModels(ctx, singleModelPrechecked(current), current, "no models available, run 'ollama pull <model>' first")
	if err != nil {
		return "", err
	}

	selected, err := selector(title, items, current)
	if err != nil {
		return "", err
	}
	if err := c.ensureModelsReady(ctx, []string{selected}); err != nil {
		return "", err
	}
	return selected, nil
}

func (c *launcherClient) selectMultiModelsForIntegration(ctx context.Context, runner config.Runner, preChecked []string) ([]string, error) {
	if config.DefaultMultiSelector == nil {
		return nil, fmt.Errorf("no selector configured")
	}

	items, orderedChecked, err := c.loadSelectableModels(ctx, preChecked, firstModel(preChecked), "no models available")
	if err != nil {
		return nil, err
	}

	selected, err := config.DefaultMultiSelector(fmt.Sprintf("Select models for %s:", runner), items, orderedChecked)
	if err != nil {
		return nil, err
	}
	if err := c.ensureModelsReady(ctx, selected); err != nil {
		return nil, err
	}
	return selected, nil
}

func (c *launcherClient) loadSelectableModels(ctx context.Context, preChecked []string, current, emptyMessage string) ([]config.ModelItem, []string, error) {
	if err := c.loadModelInventoryOnce(ctx); err != nil {
		return nil, nil, err
	}

	items, orderedChecked, _, _ := config.BuildModelList(c.modelInventory, preChecked, current)
	if c.cloudDisabled {
		items = config.FilterCloudItems(items)
		orderedChecked = c.filterDisabledCloudModels(ctx, orderedChecked)
	}
	if len(items) == 0 {
		return nil, nil, errors.New(emptyMessage)
	}
	return items, orderedChecked, nil
}

func (c *launcherClient) ensureModelsReady(ctx context.Context, models []string) error {
	var deduped []string
	seen := make(map[string]bool, len(models))
	for _, model := range models {
		if model == "" || seen[model] {
			continue
		}
		seen[model] = true
		deduped = append(deduped, model)
	}
	models = deduped
	if len(models) == 0 {
		return nil
	}

	cloudModels := make(map[string]bool, len(models))
	for _, model := range models {
		if err := config.ShowOrPull(ctx, c.apiClient, model); err != nil {
			return err
		}
		if config.IsCloudModelName(model) {
			cloudModels[model] = true
		}
	}

	return config.EnsureAuth(ctx, c.apiClient, cloudModels, models)
}

func (c *launcherClient) resolveEditorLaunchModels(ctx context.Context, saved *config.IntegrationConfig, req IntegrationLaunchRequest) ([]string, bool) {
	if req.ForceConfigure {
		return editorPreCheckedModels(saved, req.ModelOverride), true
	}

	if req.ModelOverride != "" {
		models := append([]string{req.ModelOverride}, additionalSavedModels(saved, req.ModelOverride)...)
		models = c.filterDisabledCloudModels(ctx, models)
		return models, len(models) == 0
	}

	if saved == nil || len(saved.Models) == 0 {
		return nil, true
	}

	models := c.filterDisabledCloudModels(ctx, saved.Models)
	return models, len(models) == 0
}

func (c *launcherClient) filterDisabledCloudModels(ctx context.Context, models []string) []string {
	c.ensureCloudStatus(ctx)
	if !c.cloudDisabled {
		return append([]string(nil), models...)
	}

	filtered := make([]string, 0, len(models))
	for _, model := range models {
		if !config.IsCloudModelName(model) {
			filtered = append(filtered, model)
		}
	}
	return filtered
}

func (c *launcherClient) savedModelUsable(ctx context.Context, name string) (bool, error) {
	if err := c.loadModelInventoryOnce(ctx); err != nil {
		return false, err
	}
	return c.singleModelUsable(name), nil
}

func (c *launcherClient) singleModelUsable(name string) bool {
	if name == "" {
		return false
	}
	if config.IsCloudModelName(name) {
		return !c.cloudDisabled
	}
	return c.hasLocalModel(name)
}

func (c *launcherClient) hasLocalModel(name string) bool {
	for _, model := range c.modelInventory {
		if model.Remote {
			continue
		}
		if model.Name == name || strings.HasPrefix(model.Name, name+":") {
			return true
		}
	}
	return false
}

func (c *launcherClient) ensureCloudStatus(ctx context.Context) {
	if c.cloudStatusLoaded {
		return
	}
	c.cloudDisabled, _ = config.CloudStatusDisabled(ctx, c.apiClient)
	c.cloudStatusLoaded = true
}

func (c *launcherClient) loadModelInventoryOnce(ctx context.Context) error {
	if c.inventoryLoaded {
		return nil
	}

	resp, err := c.apiClient.List(ctx)
	if err != nil {
		return err
	}

	c.ensureCloudStatus(ctx)
	c.modelInventory = c.modelInventory[:0]
	for _, model := range resp.Models {
		c.modelInventory = append(c.modelInventory, config.ModelInfo{
			Name:   model.Name,
			Remote: model.RemoteModel != "",
		})
	}
	if c.cloudDisabled {
		c.modelInventory = config.FilterCloudModels(c.modelInventory)
	}
	c.inventoryLoaded = true
	return nil
}

func runIntegration(runner config.Runner, modelName string, args []string) error {
	fmt.Fprintf(os.Stderr, "\nLaunching %s with %s...\n", runner, modelName)
	return runner.Run(modelName, args)
}

func syncAliases(ctx context.Context, client *api.Client, aliasConfigurer config.AliasConfigurer, name, model string, existing map[string]string) error {
	aliases := cloneAliases(existing)
	aliases["primary"] = model

	if config.IsCloudModelName(model) {
		aliases["fast"] = model
	} else {
		delete(aliases, "fast")
	}

	if err := aliasConfigurer.SetAliases(ctx, aliases); err != nil {
		return err
	}
	return config.SaveAliases(name, aliases)
}

func launchAfterConfiguration(name string, runner config.Runner, model string, req IntegrationLaunchRequest) error {
	if req.ConfigureOnly {
		launch, err := config.ConfirmPrompt(fmt.Sprintf("Launch %s now?", runner))
		if err != nil {
			return err
		}
		if !launch {
			return nil
		}
	}
	if err := config.EnsureIntegrationInstalled(name, runner); err != nil {
		return err
	}
	return runIntegration(runner, model, req.ExtraArgs)
}

func primaryModelFromConfig(cfg *config.IntegrationConfig) string {
	if cfg == nil || len(cfg.Models) == 0 {
		return ""
	}
	return cfg.Models[0]
}

func cloneAliases(aliases map[string]string) map[string]string {
	if len(aliases) == 0 {
		return make(map[string]string)
	}

	cloned := make(map[string]string, len(aliases))
	for key, value := range aliases {
		cloned[key] = value
	}
	return cloned
}

func normalizedAliases(primary string, aliases map[string]string) map[string]string {
	normalized := cloneAliases(aliases)
	normalized["primary"] = primary
	if config.IsCloudModelName(primary) {
		normalized["fast"] = primary
	} else {
		delete(normalized, "fast")
	}
	return normalized
}

func singleModelPrechecked(current string) []string {
	if current == "" {
		return nil
	}
	return []string{current}
}

func firstModel(models []string) string {
	if len(models) == 0 {
		return ""
	}
	return models[0]
}

func editorPreCheckedModels(saved *config.IntegrationConfig, override string) []string {
	if override == "" {
		if saved == nil {
			return nil
		}
		return append([]string(nil), saved.Models...)
	}
	return append([]string{override}, additionalSavedModels(saved, override)...)
}

func additionalSavedModels(saved *config.IntegrationConfig, exclude string) []string {
	if saved == nil {
		return nil
	}

	var models []string
	for _, model := range saved.Models {
		if model != exclude {
			models = append(models, model)
		}
	}
	return models
}
