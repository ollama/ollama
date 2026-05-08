package launch

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"slices"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/cmd/config"
	"github.com/spf13/cobra"
	"golang.org/x/term"
)

// LauncherState is the launch-owned snapshot used to render the root launcher menu.
type LauncherState struct {
	LastSelection  string
	RunModel       string
	RunModelUsable bool
	Integrations   map[string]LauncherIntegrationState
	AccountState   *AccountState
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
	ForcePicker          bool
	Policy               *LaunchPolicy
	AccountState         *AccountState
	AccountStateProvider func() *AccountState
	AccountStateUpdates  func(context.Context) <-chan *AccountState
}

// LaunchConfirmMode controls confirmation behavior across launch flows.
type LaunchConfirmMode int

const (
	// LaunchConfirmPrompt prompts the user for confirmation.
	LaunchConfirmPrompt LaunchConfirmMode = iota
	// LaunchConfirmAutoApprove skips prompts and treats confirmation as accepted.
	LaunchConfirmAutoApprove
	// LaunchConfirmRequireYes rejects confirmation requests with a --yes hint.
	LaunchConfirmRequireYes
)

// LaunchMissingModelMode controls local missing-model handling in launch flows.
type LaunchMissingModelMode int

const (
	// LaunchMissingModelPromptToPull prompts to pull a missing local model.
	LaunchMissingModelPromptToPull LaunchMissingModelMode = iota
	// LaunchMissingModelAutoPull pulls a missing local model without prompting.
	LaunchMissingModelAutoPull
	// LaunchMissingModelFail fails immediately when a local model is missing.
	LaunchMissingModelFail
)

// LaunchPolicy controls launch behavior that may vary by caller context.
type LaunchPolicy struct {
	Confirm      LaunchConfirmMode
	MissingModel LaunchMissingModelMode
}

func defaultLaunchPolicy(interactive bool, yes bool) LaunchPolicy {
	policy := LaunchPolicy{
		Confirm:      LaunchConfirmPrompt,
		MissingModel: LaunchMissingModelPromptToPull,
	}
	switch {
	case yes:
		// if yes flag is set, auto approve and auto pull
		policy.Confirm = LaunchConfirmAutoApprove
		policy.MissingModel = LaunchMissingModelAutoPull
	case !interactive:
		// otherwise make sure to stop when needed
		policy.Confirm = LaunchConfirmRequireYes
		policy.MissingModel = LaunchMissingModelFail
	}
	return policy
}

func (p LaunchPolicy) confirmPolicy() launchConfirmPolicy {
	switch p.Confirm {
	case LaunchConfirmAutoApprove:
		return launchConfirmPolicy{yes: true}
	case LaunchConfirmRequireYes:
		return launchConfirmPolicy{requireYesMessage: true}
	default:
		return launchConfirmPolicy{}
	}
}

func (p LaunchPolicy) missingModelPolicy() missingModelPolicy {
	switch p.MissingModel {
	case LaunchMissingModelAutoPull:
		return missingModelAutoPull
	case LaunchMissingModelFail:
		return missingModelFail
	default:
		return missingModelPromptPull
	}
}

// IntegrationLaunchRequest controls the canonical integration launcher flow.
type IntegrationLaunchRequest struct {
	Name                 string
	ModelOverride        string
	ForceConfigure       bool
	ConfigureOnly        bool
	Restore              bool
	ExtraArgs            []string
	Policy               *LaunchPolicy
	AccountState         *AccountState
	AccountStateProvider func() *AccountState
	AccountStateUpdates  func(context.Context) <-chan *AccountState
}

var isInteractiveSession = func() bool {
	return term.IsTerminal(int(os.Stdin.Fd())) && term.IsTerminal(int(os.Stdout.Fd()))
}

// Runner executes a model with an integration.
type Runner interface {
	Run(model string, args []string) error
	String() string
}

// Editor can edit config files for integrations that support model configuration.
type Editor interface {
	Paths() []string
	Edit(models []string) error
	Models() []string
}

// ManagedSingleModel is the narrow launch-owned config path for integrations
// like Hermes that have one primary model selected by launcher, need launcher
// to persist minimal config, and still keep their own model discovery and
// onboarding UX. This stays separate from Runner-only integrations and the
// multi-model Editor flow so Hermes-specific behavior stays scoped to one path.
type ManagedSingleModel interface {
	Paths() []string
	Configure(model string) error
	CurrentModel() string
	Onboard() error
}

// ManagedModelListConfigurer lets managed single-model integrations receive
// the launcher's model list while still preserving one primary selected model.
type ManagedModelListConfigurer interface {
	ConfigureWithModels(primary string, models []string) error
}

// ManagedAutodiscoveryIntegration is for managed integrations that do not need
// a launcher-selected model because the app discovers available models itself.
type ManagedAutodiscoveryIntegration interface {
	Paths() []string
	AutodiscoveredModel() string
	AutodiscoveryConfigured() bool
	ConfigureAutodiscovery() error
	Onboard() error
}

// ManagedAutodiscoveryCloudIntegration marks an autodiscovery integration whose
// discovered model catalog depends on the user's local Ollama Cloud auth state.
type ManagedAutodiscoveryCloudIntegration interface {
	UsesOllamaCloud() bool
}

// RestoreHintIntegration can provide a short restore command after launch
// switches an app into a launch-managed mode.
type RestoreHintIntegration interface {
	RestoreHint() string
}

// ConfigurationSuccessIntegration can print a short message after launcher
// successfully switches an app into a launch-managed mode.
type ConfigurationSuccessIntegration interface {
	ConfigurationSuccessMessage() string
}

// RestoreSuccessIntegration can print a short message after launcher restores
// an app back to its default mode.
type RestoreSuccessIntegration interface {
	RestoreSuccessMessage() string
}

// ManagedRuntimeRefresher lets managed integrations refresh any long-lived
// background runtime after launch rewrites their config.
type ManagedRuntimeRefresher interface {
	RefreshRuntimeAfterConfigure() error
}

// ManagedOnboardingValidator lets managed integrations re-check saved
// onboarding state when launcher needs a stronger live readiness signal.
type ManagedOnboardingValidator interface {
	OnboardingComplete() bool
}

// ManagedInteractiveOnboarding lets a managed integration declare whether its
// onboarding step really requires an interactive terminal. Hermes does not.
type ManagedInteractiveOnboarding interface {
	RequiresInteractiveOnboarding() bool
}

// ManagedModelReadinessSkipper lets managed integrations opt out of local
// Ollama model readiness checks when the configured runtime is not the local
// daemon.
type ManagedModelReadinessSkipper interface {
	SkipModelReadiness() bool
}

// RestorableIntegration lets integrations switch back from a launch-managed
// mode to the application's normal/default mode.
type RestorableIntegration interface {
	Restore() error
}

// SupportedIntegration lets an integration report platform support separately
// from whether the underlying app binary is installed.
type SupportedIntegration interface {
	Supported() error
}

type modelInfo struct {
	Name        string
	Remote      bool
	ToolCapable bool
}

// ModelInfo re-exports launcher model inventory details for callers.
type ModelInfo = modelInfo

// ModelItem represents model metadata before selector-only UI state is derived.
type ModelItem struct {
	Name            string
	Description     string
	Recommended     bool
	VRAMBytes       int64
	ContextLength   int
	MaxOutputTokens int
	RequiredPlan    string
}

// SelectionItem represents a model row after launch has derived selector-only UI state.
type SelectionItem struct {
	Name              string
	Description       string
	Recommended       bool
	AvailabilityBadge string
}

// LaunchCmd returns the cobra command for launching integrations.
// The runTUI callback is called when the root launcher UI should be shown.
func LaunchCmd(checkServerHeartbeat func(cmd *cobra.Command, args []string) error, runTUI func(cmd *cobra.Command)) *cobra.Command {
	var modelFlag string
	var configFlag bool
	var yesFlag bool
	var restoreFlag bool

	cmd := &cobra.Command{
		Use:   "launch [INTEGRATION] [-- [EXTRA_ARGS...]]",
		Short: "Launch the Ollama menu or an integration",
		Long: `Launch the Ollama interactive menu, or directly launch a specific integration.

Without arguments, this is equivalent to running 'ollama' directly.
Flags and extra arguments require an integration name.

Supported integrations:
  claude          Claude Code
  cline           Cline
  codex           Codex
  copilot         Copilot CLI (aliases: copilot-cli)
  droid           Droid
  hermes          Hermes Agent
  kimi            Kimi Code CLI
  opencode        OpenCode
  openclaw        OpenClaw (aliases: clawdbot, moltbot)
  pi              Pi
  pool            Pool
  vscode          VS Code (aliases: code)

Examples:
  ollama launch
  ollama launch claude
  ollama launch claude --model <model>
  ollama launch hermes
  ollama launch droid --config (does not auto-launch)
  ollama launch codex -- -p myprofile (pass extra args to integration)
  ollama launch codex -- --sandbox workspace-write`,
		Args: cobra.ArbitraryArgs,
		PreRunE: func(cmd *cobra.Command, args []string) error {
			if restoreFlag || launchCommandCanSkipHeartbeat(args) {
				return nil
			}
			return checkServerHeartbeat(cmd, args)
		},
		RunE: func(cmd *cobra.Command, args []string) error {
			policy := defaultLaunchPolicy(isInteractiveSession(), yesFlag)
			// reset when done to make sure state doens't leak between launches
			restoreConfirmPolicy := withLaunchConfirmPolicy(policy.confirmPolicy())
			defer restoreConfirmPolicy()

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

			if name == "" {
				if cmd.Flags().Changed("model") || cmd.Flags().Changed("config") || cmd.Flags().Changed("yes") || cmd.Flags().Changed("restore") || len(passArgs) > 0 {
					return fmt.Errorf("flags and extra args require an integration name, for example: 'ollama launch claude --model qwen3.5'")
				}
				runTUI(cmd)
				return nil
			}

			if !restoreFlag && launchCommandIsClaudeDesktop(name) {
				return errClaudeDesktopUnsupported()
			}

			if modelFlag != "" && isCloudModelName(modelFlag) {
				if client, err := api.ClientFromEnvironment(); err == nil {
					if disabled, _ := cloudStatusDisabled(cmd.Context(), client); disabled {
						fmt.Fprintf(os.Stderr, "Warning: ignoring --model %s because cloud is disabled\n", modelFlag)
						modelFlag = ""
					}
				}
			}

			headlessYes := yesFlag && !isInteractiveSession()
			forceConfigure := configFlag || (modelFlag == "" && !headlessYes)
			if forceConfigure && !configFlag && modelFlag == "" {
				if _, runner, err := LookupIntegration(name); err == nil {
					if _, ok := runner.(ManagedAutodiscoveryIntegration); ok {
						forceConfigure = false
					}
				}
			}
			err := LaunchIntegration(cmd.Context(), IntegrationLaunchRequest{
				Name:           name,
				ModelOverride:  modelFlag,
				ForceConfigure: forceConfigure,
				ConfigureOnly:  configFlag,
				Restore:        restoreFlag,
				ExtraArgs:      passArgs,
				Policy:         &policy,
			})
			if errors.Is(err, ErrCancelled) {
				return nil
			}
			return err
		},
	}

	cmd.Flags().StringVar(&modelFlag, "model", "", "Model to use")
	cmd.Flags().BoolVar(&configFlag, "config", false, "Configure without launching")
	cmd.Flags().BoolVar(&restoreFlag, "restore", false, "Restore an integration to its default profile")
	cmd.Flags().BoolVarP(&yesFlag, "yes", "y", false, "Automatically answer yes to confirmation prompts")
	return cmd
}

func launchCommandCanSkipHeartbeat(args []string) bool {
	if len(args) == 0 {
		return false
	}
	return launchCommandIsClaudeDesktop(args[0])
}

func launchCommandIsClaudeDesktop(name string) bool {
	canonical, _, err := LookupIntegration(name)
	return err == nil && canonical == claudeDesktopIntegrationName
}

type launcherClient struct {
	apiClient             *api.Client
	modelInventory        []ModelInfo
	inventoryLoaded       bool
	recommendationsLoaded bool
	recommendationItems   []ModelItem
	accountState          *AccountState
	accountStateProvider  func() *AccountState
	accountStateUpdates   func(context.Context) <-chan *AccountState
	policy                LaunchPolicy
}

func newLauncherClient(policy LaunchPolicy) (*launcherClient, error) {
	apiClient, err := api.ClientFromEnvironment()
	if err != nil {
		return nil, err
	}

	return &launcherClient{
		apiClient: apiClient,
		policy:    policy,
	}, nil
}

// BuildLauncherState returns the launch-owned root launcher menu snapshot.
func BuildLauncherState(ctx context.Context) (*LauncherState, error) {
	launchClient, err := newLauncherClient(defaultLaunchPolicy(isInteractiveSession(), false))
	if err != nil {
		return nil, err
	}
	return launchClient.buildLauncherState(ctx)
}

// ResolveRunModel returns the model that should be used for interactive chat.
func ResolveRunModel(ctx context.Context, req RunModelRequest) (string, error) {
	// Called by the launcher TUI "Run a model" action (cmd/runLauncherAction),
	// which resolves models separately from LaunchIntegration. Callers can pass
	// Policy directly; otherwise we fall back to ambient --yes/session defaults.
	policy := defaultLaunchPolicy(isInteractiveSession(), currentLaunchConfirmPolicy.yes)
	if req.Policy != nil {
		policy = *req.Policy
	}

	launchClient, err := newLauncherClient(policy)
	if err != nil {
		return "", err
	}
	launchClient.accountState = req.AccountState
	launchClient.accountStateProvider = req.AccountStateProvider
	launchClient.accountStateUpdates = req.AccountStateUpdates
	return launchClient.resolveRunModel(ctx, req)
}

// LaunchIntegration runs the canonical launcher flow for one integration.
func LaunchIntegration(ctx context.Context, req IntegrationLaunchRequest) error {
	name, runner, err := LookupIntegration(req.Name)
	if err != nil {
		return err
	}

	if name == claudeDesktopIntegrationName && !req.Restore {
		return errClaudeDesktopUnsupported()
	}

	policy := launchIntegrationPolicy(req)
	if req.Restore {
		return restoreIntegration(name, runner, req)
	}
	if policy.Confirm == LaunchConfirmAutoApprove && !isInteractiveSession() && req.ModelOverride == "" {
		if _, ok := runner.(ManagedAutodiscoveryIntegration); !ok {
			return fmt.Errorf("headless --yes launch for %s requires --model <model>", name)
		}
	}

	launchClient, saved, err := prepareIntegrationLaunch(name, policy)
	if err != nil {
		return err
	}
	launchClient.accountState = req.AccountState
	launchClient.accountStateProvider = req.AccountStateProvider
	launchClient.accountStateUpdates = req.AccountStateUpdates

	if autodiscovery, ok := runner.(ManagedAutodiscoveryIntegration); ok {
		if err := EnsureIntegrationInstalled(name, runner); err != nil {
			return err
		}
		return launchClient.launchManagedAutodiscoveryIntegration(ctx, name, runner, autodiscovery, saved, req)
	}

	if managed, ok := runner.(ManagedSingleModel); ok {
		if err := EnsureIntegrationInstalled(name, runner); err != nil {
			return err
		}
		return launchClient.launchManagedSingleIntegration(ctx, name, runner, managed, saved, req)
	}

	if !req.ConfigureOnly {
		if err := EnsureIntegrationInstalled(name, runner); err != nil {
			return err
		}
	}

	if editor, ok := runner.(Editor); ok {
		return launchClient.launchEditorIntegration(ctx, name, runner, editor, saved, req)
	}
	return launchClient.launchSingleIntegration(ctx, name, runner, saved, req)
}

func restoreIntegration(name string, runner Runner, req IntegrationLaunchRequest) error {
	if req.ModelOverride != "" || req.ConfigureOnly || len(req.ExtraArgs) > 0 {
		return fmt.Errorf("--restore cannot be combined with --model, --config, or extra args")
	}
	restorable, ok := runner.(RestorableIntegration)
	if !ok {
		return fmt.Errorf("%s does not support --restore", name)
	}
	if err := EnsureIntegrationInstalled(name, runner); err != nil {
		return err
	}
	if err := restorable.Restore(); err != nil {
		return err
	}
	printRestoreSuccess(restorable)
	return nil
}

func launchIntegrationPolicy(req IntegrationLaunchRequest) LaunchPolicy {
	// TUI does not set a policy, whereas ollama launch <app> does as it can
	// have flags which change the behavior.
	if req.Policy != nil {
		return *req.Policy
	}
	return defaultLaunchPolicy(isInteractiveSession(), false)
}

func prepareIntegrationLaunch(name string, policy LaunchPolicy) (*launcherClient, *config.IntegrationConfig, error) {
	launchClient, err := newLauncherClient(policy)
	if err != nil {
		return nil, nil, err
	}
	saved, _ := loadStoredIntegrationConfig(name)
	return launchClient, saved, nil
}

func (c *launcherClient) buildLauncherState(ctx context.Context) (*LauncherState, error) {
	_ = c.loadModelInventoryOnce(ctx)

	state := &LauncherState{
		LastSelection: config.LastSelection(),
		RunModel:      config.LastModel(),
		Integrations:  make(map[string]LauncherIntegrationState),
	}
	runModelUsable, err := c.savedModelUsable(ctx, state.RunModel)
	if err != nil {
		runModelUsable = false
	}
	state.RunModelUsable = runModelUsable

	for _, info := range ListIntegrationInfos() {
		integrationState, err := c.buildLauncherIntegrationState(ctx, info)
		if err != nil {
			return nil, err
		}
		state.Integrations[info.Name] = integrationState
	}

	return state, nil
}

func (c *launcherClient) buildLauncherIntegrationState(ctx context.Context, info IntegrationInfo) (LauncherIntegrationState, error) {
	integration, err := integrationFor(info.Name)
	if err != nil {
		return LauncherIntegrationState{}, err
	}
	var currentModel string
	var usable bool
	if autodiscovery, ok := integration.spec.Runner.(ManagedAutodiscoveryIntegration); ok {
		currentModel, usable, err = c.launcherManagedAutodiscoveryState(ctx, info.Name, autodiscovery)
		if err != nil {
			return LauncherIntegrationState{}, err
		}
	} else if managed, ok := integration.spec.Runner.(ManagedSingleModel); ok {
		currentModel, usable, err = c.launcherManagedModelState(ctx, info.Name, managed)
		if err != nil {
			return LauncherIntegrationState{}, err
		}
	} else {
		currentModel, usable, err = c.launcherModelState(ctx, info.Name, integration.editor)
		if err != nil {
			return LauncherIntegrationState{}, err
		}
	}

	return LauncherIntegrationState{
		Name:            info.Name,
		DisplayName:     info.DisplayName,
		Description:     info.Description,
		Installed:       integration.installed,
		AutoInstallable: integration.autoInstallable,
		Selectable:      integration.installed || integration.autoInstallable,
		Changeable:      integration.installed || integration.autoInstallable,
		CurrentModel:    currentModel,
		ModelUsable:     usable,
		InstallHint:     integration.installHint,
		Editor:          integration.editor,
	}, nil
}

func (c *launcherClient) launcherModelState(ctx context.Context, name string, isEditor bool) (string, bool, error) {
	cfg, loadErr := loadStoredIntegrationConfig(name)
	hasModels := loadErr == nil && len(cfg.Models) > 0
	if !hasModels {
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
	usable, usableErr := c.savedModelUsable(ctx, model)
	return model, usableErr == nil && usable, nil
}

func (c *launcherClient) launcherManagedModelState(ctx context.Context, name string, managed ManagedSingleModel) (string, bool, error) {
	current := managed.CurrentModel()
	if current == "" {
		cfg, loadErr := loadStoredIntegrationConfig(name)
		if loadErr == nil {
			current = primaryModelFromConfig(cfg)
		}
		if current != "" {
			return current, false, nil
		}
	}
	if current == "" {
		return "", false, nil
	}

	if skips, ok := managed.(ManagedModelReadinessSkipper); ok && skips.SkipModelReadiness() {
		return current, true, nil
	}

	usable, err := c.savedModelUsable(ctx, current)
	if err != nil {
		return current, false, err
	}
	return current, usable, nil
}

func (c *launcherClient) launcherManagedAutodiscoveryState(ctx context.Context, name string, autodiscovery ManagedAutodiscoveryIntegration) (string, bool, error) {
	if autodiscovery.AutodiscoveryConfigured() {
		return autodiscovery.AutodiscoveredModel(), c.managedAutodiscoveryUsable(ctx, autodiscovery), nil
	}

	cfg, loadErr := loadStoredIntegrationConfig(name)
	if loadErr == nil {
		if current := primaryModelFromConfig(cfg); current != "" {
			return current, false, nil
		}
	}
	return "", false, nil
}

func (c *launcherClient) resolveRunModel(ctx context.Context, req RunModelRequest) (string, error) {
	current := config.LastModel()
	if !req.ForcePicker && current != "" && c.policy.Confirm == LaunchConfirmAutoApprove && !isInteractiveSession() {
		if err := c.ensureModelsReady(ctx, []string{current}); err != nil {
			return "", err
		}
		fmt.Fprintf(os.Stderr, "Headless mode: auto-selected last used model %q\n", current)
		return current, nil
	}

	if !req.ForcePicker {
		usable, err := c.savedModelUsable(ctx, current)
		if err != nil {
			return "", err
		}
		if usable {
			if err := c.ensureModelsReady(ctx, []string{current}); err != nil {
				return "", err
			}
			return current, nil
		}
	}

	model, err := c.selectSingleModelWithSelector(ctx, "Select model to run:", current, DefaultSingleSelector)
	if err != nil {
		return "", err
	}
	if model != current {
		if err := config.SetLastModel(model); err != nil {
			return "", err
		}
	}
	return model, nil
}

func (c *launcherClient) launchSingleIntegration(ctx context.Context, name string, runner Runner, saved *config.IntegrationConfig, req IntegrationLaunchRequest) error {
	target, _, err := c.resolveSingleIntegrationTarget(ctx, runner, primaryModelFromConfig(saved), req)
	if err != nil {
		return err
	}
	if target == "" {
		return nil
	}

	current := primaryModelFromConfig(saved)
	if target != current {
		if err := config.SaveIntegration(name, []string{target}); err != nil {
			return fmt.Errorf("failed to save: %w", err)
		}
	}

	return launchAfterConfiguration(name, runner, target, req)
}

func (c *launcherClient) launchEditorIntegration(ctx context.Context, name string, runner Runner, editor Editor, saved *config.IntegrationConfig, req IntegrationLaunchRequest) error {
	models, needsConfigure := c.resolveEditorLaunchModels(ctx, saved, req)

	if needsConfigure {
		selected, err := c.selectMultiModelsForIntegration(ctx, runner, models)
		if err != nil {
			return err
		}
		models = selected
	} else if len(models) > 0 {
		if err := c.ensureModelsReady(ctx, models[:1]); err != nil {
			return err
		}
	}

	if len(models) == 0 {
		return nil
	}

	if (needsConfigure || req.ModelOverride != "") && !savedMatchesModels(saved, models) {
		if err := prepareEditorIntegration(name, editor, models); err != nil {
			return err
		}
	}

	return launchAfterConfiguration(name, runner, models[0], req)
}

func (c *launcherClient) launchManagedSingleIntegration(ctx context.Context, name string, runner Runner, managed ManagedSingleModel, saved *config.IntegrationConfig, req IntegrationLaunchRequest) error {
	current := managed.CurrentModel()
	selectionCurrent := current
	if selectionCurrent == "" {
		selectionCurrent = primaryModelFromConfig(saved)
	}

	target, needsConfigure, err := c.resolveSingleIntegrationTarget(ctx, runner, selectionCurrent, req)
	if err != nil {
		return err
	}
	if target == "" {
		return nil
	}

	if needsConfigure || req.ModelOverride != "" || (current != "" && target != current) || !savedMatchesModels(saved, []string{target}) {
		configureModels, err := c.managedSingleConfigureModels(ctx, managed, target)
		if err != nil {
			return err
		}
		if err := prepareManagedSingleIntegration(name, managed, target, configureModels); err != nil {
			return err
		}
		if refresher, ok := managed.(ManagedRuntimeRefresher); ok {
			if err := refresher.RefreshRuntimeAfterConfigure(); err != nil {
				return err
			}
		}
	}

	if !managedIntegrationOnboarded(saved, managed) {
		if !isInteractiveSession() && managedRequiresInteractiveOnboarding(managed) {
			return fmt.Errorf("%s still needs interactive gateway setup; run 'ollama launch %s' in a terminal to finish onboarding", runner, name)
		}
		if err := managed.Onboard(); err != nil {
			return err
		}
	}

	if req.ConfigureOnly {
		return nil
	}

	return runIntegration(runner, target, req.ExtraArgs)
}

func (c *launcherClient) launchManagedAutodiscoveryIntegration(ctx context.Context, name string, runner Runner, autodiscovery ManagedAutodiscoveryIntegration, saved *config.IntegrationConfig, req IntegrationLaunchRequest) error {
	if req.ModelOverride != "" {
		return fmt.Errorf("%s discovers models automatically; omit --model", runner)
	}

	target := autodiscovery.AutodiscoveredModel()
	if err := c.ensureManagedAutodiscoveryUsable(ctx, autodiscovery, target); err != nil {
		return err
	}

	needsConfigure := req.ForceConfigure || req.ConfigureOnly || !autodiscovery.AutodiscoveryConfigured() || !savedMatchesModels(saved, []string{target})

	if needsConfigure {
		if err := prepareManagedAutodiscoveryIntegration(name, autodiscovery, target); err != nil {
			return err
		}
		if refresher, ok := autodiscovery.(ManagedRuntimeRefresher); ok {
			if err := refresher.RefreshRuntimeAfterConfigure(); err != nil {
				return err
			}
		}
	}

	if !managedIntegrationOnboarded(saved, autodiscovery) {
		if !isInteractiveSession() && managedRequiresInteractiveOnboarding(autodiscovery) {
			return fmt.Errorf("%s still needs interactive gateway setup; run 'ollama launch %s' in a terminal to finish onboarding", runner, name)
		}
		if err := autodiscovery.Onboard(); err != nil {
			return err
		}
	}

	if !printConfigurationSuccess(autodiscovery) {
		printRestoreHint(autodiscovery)
	}

	if req.ConfigureOnly {
		return nil
	}

	return runIntegration(runner, target, req.ExtraArgs)
}

func (c *launcherClient) managedAutodiscoveryUsable(ctx context.Context, autodiscovery ManagedAutodiscoveryIntegration) bool {
	if !managedAutodiscoveryUsesOllamaCloud(autodiscovery) {
		return true
	}
	if disabled, known := cloudStatusDisabled(ctx, c.apiClient); known && disabled {
		return false
	}
	return true
}

func (c *launcherClient) ensureManagedAutodiscoveryUsable(ctx context.Context, autodiscovery ManagedAutodiscoveryIntegration, label string) error {
	if !managedAutodiscoveryUsesOllamaCloud(autodiscovery) {
		return nil
	}
	return ensureCloudAuth(ctx, c.apiClient, label)
}

func managedAutodiscoveryUsesOllamaCloud(autodiscovery ManagedAutodiscoveryIntegration) bool {
	cloud, ok := autodiscovery.(ManagedAutodiscoveryCloudIntegration)
	return ok && cloud.UsesOllamaCloud()
}

func printRestoreHint(integration any) {
	hint, ok := integration.(RestoreHintIntegration)
	if !ok {
		return
	}
	if msg := strings.TrimSpace(hint.RestoreHint()); msg != "" {
		fmt.Fprintln(os.Stderr, msg)
	}
}

func printConfigurationSuccess(integration any) bool {
	success, ok := integration.(ConfigurationSuccessIntegration)
	if !ok {
		return false
	}
	if msg := strings.TrimSpace(success.ConfigurationSuccessMessage()); msg != "" {
		fmt.Fprintln(os.Stderr, msg)
		return true
	}
	return false
}

func printRestoreSuccess(integration any) {
	success, ok := integration.(RestoreSuccessIntegration)
	if !ok {
		return
	}
	if msg := strings.TrimSpace(success.RestoreSuccessMessage()); msg != "" {
		fmt.Fprintln(os.Stderr, msg)
	}
}

func (c *launcherClient) managedSingleConfigureModels(ctx context.Context, managed ManagedSingleModel, target string) ([]string, error) {
	models := []string{target}
	if _, ok := managed.(ManagedModelListConfigurer); !ok {
		return models, nil
	}

	items, _, err := c.loadSelectableModels(ctx, []string{target}, target, "no models available")
	if err != nil {
		// Managed integrations that can use a model catalog should still be
		// configurable with an explicit target even if the broader inventory
		// cannot be loaded in the moment.
		//nolint:nilerr
		return models, nil
	}

	for _, item := range items {
		models = append(models, item.Name)
	}
	return dedupeModelList(models), nil
}

func (c *launcherClient) resolveSingleIntegrationTarget(ctx context.Context, runner Runner, current string, req IntegrationLaunchRequest) (string, bool, error) {
	target := req.ModelOverride
	needsConfigure := req.ForceConfigure
	skipReadiness := false
	if skipper, ok := runner.(ManagedModelReadinessSkipper); ok {
		skipReadiness = skipper.SkipModelReadiness()
	}

	if target == "" {
		target = current
		usable := skipReadiness && target != ""
		if !skipReadiness {
			var err error
			usable, err = c.savedModelUsable(ctx, target)
			if err != nil {
				return "", false, err
			}
		}
		if !usable {
			needsConfigure = true
		}
	}

	if needsConfigure {
		selected, err := c.selectSingleModelWithSelectorReady(ctx, fmt.Sprintf("Select model for %s:", runner), target, DefaultSingleSelector, !skipReadiness)
		if err != nil {
			return "", false, err
		}
		target = selected
	} else if !skipReadiness {
		if err := c.ensureModelsReady(ctx, []string{target}); err != nil {
			return "", false, err
		}
	}

	if target == "" {
		return "", false, nil
	}

	return target, needsConfigure, nil
}

func savedIntegrationOnboarded(saved *config.IntegrationConfig) bool {
	return saved != nil && saved.Onboarded
}

func managedIntegrationOnboarded(saved *config.IntegrationConfig, managed any) bool {
	if !savedIntegrationOnboarded(saved) {
		return false
	}
	validator, ok := managed.(ManagedOnboardingValidator)
	if !ok {
		return true
	}
	return validator.OnboardingComplete()
}

// Most managed integrations treat onboarding as an interactive terminal step.
// Hermes opts out because its launch-owned onboarding is just bookkeeping, so
// headless launches should not be blocked once config is already prepared.
func managedRequiresInteractiveOnboarding(managed any) bool {
	onboarding, ok := managed.(ManagedInteractiveOnboarding)
	if !ok {
		return true
	}
	return onboarding.RequiresInteractiveOnboarding()
}

func (c *launcherClient) selectSingleModelWithSelector(ctx context.Context, title, current string, selector SingleSelector) (string, error) {
	return c.selectSingleModelWithSelectorReady(ctx, title, current, selector, true)
}

func (c *launcherClient) latestAccountState() *AccountState {
	if c.accountStateProvider != nil {
		return c.accountStateProvider()
	}
	return c.accountState
}

func (c *launcherClient) selectSingleModelWithSelectorReady(ctx context.Context, title, current string, selector SingleSelector, ensureReady bool) (string, error) {
	if selector == nil && DefaultSingleSelectorWithUpdates == nil {
		return "", fmt.Errorf("no selector configured")
	}

	items, _, err := c.loadSelectableModels(ctx, nil, current, "no models available, run 'ollama pull <model>' first")
	if err != nil {
		return "", err
	}

	for {
		accountState := c.latestAccountState()
		selectionItems := SelectionItemsWithAccountState(items, accountState)
		var updates <-chan []SelectionItem
		if DefaultSingleSelectorWithUpdates != nil {
			updates = c.selectionItemUpdates(ctx, items, accountState)
		}
		selected, err := runSingleSelector(title, selectionItems, current, updates, selector)
		if err != nil {
			return "", err
		}
		if selected == "" {
			return "", ErrCancelled
		}
		if ensureReady {
			if err := c.ensureModelsReady(ctx, []string{selected}); err != nil {
				if errors.Is(err, errUpgradeCancelled) {
					current = selected
					continue
				}
				return "", err
			}
		}
		return selected, nil
	}
}

func (c *launcherClient) selectMultiModelsForIntegration(ctx context.Context, runner Runner, preChecked []string) ([]string, error) {
	if DefaultMultiSelector == nil && DefaultMultiSelectorWithUpdates == nil {
		return nil, fmt.Errorf("no selector configured")
	}

	current := firstModel(preChecked)
	items, orderedChecked, err := c.loadSelectableModels(ctx, preChecked, current, "no models available")
	if err != nil {
		return nil, err
	}

	for {
		accountState := c.latestAccountState()
		selectionItems := SelectionItemsWithAccountState(items, accountState)
		var updates <-chan []SelectionItem
		if DefaultMultiSelectorWithUpdates != nil {
			updates = c.selectionItemUpdates(ctx, items, accountState)
		}
		selected, err := runMultiSelector(fmt.Sprintf("Select models for %s:", runner), selectionItems, orderedChecked, updates)
		if err != nil {
			return nil, err
		}
		accepted, skipped, err := c.selectReadyModelsForSave(ctx, selected)
		if err != nil {
			if errors.Is(err, errUpgradeCancelled) {
				orderedChecked = append([]string(nil), selected...)
				continue
			}
			return nil, err
		}
		for _, skip := range skipped {
			fmt.Fprintf(os.Stderr, "Skipped %s: %s\n", skip.model, skip.reason)
		}
		return accepted, nil
	}
}

func runSingleSelector(title string, items []SelectionItem, current string, updates <-chan []SelectionItem, fallback SingleSelector) (string, error) {
	if DefaultSingleSelectorWithUpdates != nil {
		return DefaultSingleSelectorWithUpdates(title, items, current, updates)
	}
	if fallback == nil {
		return "", fmt.Errorf("no selector configured")
	}
	return fallback(title, items, current)
}

func runMultiSelector(title string, items []SelectionItem, preChecked []string, updates <-chan []SelectionItem) ([]string, error) {
	if DefaultMultiSelectorWithUpdates != nil {
		return DefaultMultiSelectorWithUpdates(title, items, preChecked, updates)
	}
	if DefaultMultiSelector == nil {
		return nil, fmt.Errorf("no selector configured")
	}
	return DefaultMultiSelector(title, items, preChecked)
}

func (c *launcherClient) loadSelectableModels(ctx context.Context, preChecked []string, current, emptyMessage string) ([]ModelItem, []string, error) {
	if err := c.loadModelInventoryOnce(ctx); err != nil {
		return nil, nil, err
	}
	recommendations := c.recommendations(ctx)

	cloudDisabled, _ := cloudStatusDisabled(ctx, c.apiClient)
	items, orderedChecked, _, _ := buildModelListWithRecommendations(c.modelInventory, recommendations, preChecked, current)
	if cloudDisabled {
		items = filterCloudItems(items)
		orderedChecked = c.filterDisabledCloudModels(ctx, orderedChecked)
	}
	if len(items) == 0 {
		return nil, nil, errors.New(emptyMessage)
	}
	return items, orderedChecked, nil
}

func (c *launcherClient) recommendations(ctx context.Context) []ModelItem {
	if c.recommendationsLoaded {
		return append([]ModelItem(nil), c.recommendationItems...)
	}

	recommendations, err := c.requestRecommendations(ctx)
	if err != nil || len(recommendations) == 0 {
		// Fail open: recommendation issues should not block launch flows.
		// Fall back to built-in recommendations until server data is available.
		fallback := append([]ModelItem(nil), recommendedModels...)
		setDynamicCloudModelLimits(cloudModelLimitsFromRecommendations(fallback))
		c.recommendationItems = fallback
		c.recommendationsLoaded = true
		return append([]ModelItem(nil), fallback...)
	}
	setDynamicCloudModelLimits(cloudModelLimitsFromRecommendations(recommendations))
	c.recommendationItems = recommendations
	c.recommendationsLoaded = true
	return append([]ModelItem(nil), recommendations...)
}

func (c *launcherClient) requestRecommendations(ctx context.Context) ([]ModelItem, error) {
	resp, err := c.apiClient.ModelRecommendationsExperimental(ctx)
	if err != nil {
		return nil, err
	}
	items := make([]ModelItem, 0, len(resp.Recommendations))
	seen := make(map[string]struct{}, len(resp.Recommendations))

	for _, rec := range resp.Recommendations {
		name := strings.TrimSpace(rec.Model)
		if name == "" {
			continue
		}
		if _, ok := seen[name]; ok {
			continue
		}
		seen[name] = struct{}{}

		if isCloudModelName(name) && (rec.ContextLength <= 0 || rec.MaxOutputTokens <= 0) {
			slog.Warn("skipping cloud recommendation with missing limits", "model", name)
			continue
		}

		description := strings.TrimSpace(rec.Description)
		if description == "" {
			description = "Recommended model"
		}

		items = append(items, ModelItem{
			Name:            name,
			Description:     description,
			Recommended:     true,
			VRAMBytes:       rec.VRAMBytes,
			ContextLength:   rec.ContextLength,
			MaxOutputTokens: rec.MaxOutputTokens,
			RequiredPlan:    strings.TrimSpace(rec.RequiredPlan),
		})
	}

	return items, nil
}

func (c *launcherClient) ensureModelsReady(ctx context.Context, models []string) error {
	models = dedupeModelList(models)
	if len(models) == 0 {
		return nil
	}

	cloudModels := make(map[string]bool, len(models))
	for _, model := range models {
		isCloudModel := isCloudModelName(model)
		if isCloudModel {
			cloudModels[model] = true
			if err := c.ensureCloudModelAccess(ctx, model); err != nil {
				return err
			}
		}
		if err := showOrPullWithPolicy(ctx, c.apiClient, model, c.policy.missingModelPolicy(), isCloudModel); err != nil {
			return err
		}
	}
	return ensureAuth(ctx, c.apiClient, cloudModels, models)
}

func dedupeModelList(models []string) []string {
	deduped := make([]string, 0, len(models))
	seen := make(map[string]bool, len(models))
	for _, model := range models {
		if model == "" || seen[model] {
			continue
		}
		seen[model] = true
		deduped = append(deduped, model)
	}
	return deduped
}

type skippedModel struct {
	model  string
	reason string
}

func (c *launcherClient) selectReadyModelsForSave(ctx context.Context, selected []string) ([]string, []skippedModel, error) {
	selected = dedupeModelList(selected)
	accepted := make([]string, 0, len(selected))
	skipped := make([]skippedModel, 0, len(selected))

	for _, model := range selected {
		if err := c.ensureModelsReady(ctx, []string{model}); err != nil {
			if errors.Is(err, errUpgradeCancelled) {
				return nil, nil, err
			}
			if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
				return nil, nil, err
			}
			skipped = append(skipped, skippedModel{
				model:  model,
				reason: skippedModelReason(model, err),
			})
			continue
		}
		accepted = append(accepted, model)
	}

	return accepted, skipped, nil
}

func skippedModelReason(model string, err error) string {
	if errors.Is(err, errUpgradeCancelled) {
		return "upgrade was cancelled"
	}
	if errors.Is(err, ErrCancelled) {
		if isCloudModelName(model) {
			return "sign in was cancelled"
		}
		return "download was cancelled"
	}
	return err.Error()
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
	// if connection cannot be established or there is a 404, cloud models will continue to be displayed
	cloudDisabled, _ := cloudStatusDisabled(ctx, c.apiClient)
	if !cloudDisabled {
		return append([]string(nil), models...)
	}

	filtered := make([]string, 0, len(models))
	for _, model := range models {
		if !isCloudModelName(model) {
			filtered = append(filtered, model)
		}
	}
	return filtered
}

func (c *launcherClient) savedModelUsable(ctx context.Context, name string) (bool, error) {
	if err := c.loadModelInventoryOnce(ctx); err != nil {
		return c.showBasedModelUsable(ctx, name)
	}
	return c.singleModelUsable(ctx, name), nil
}

func (c *launcherClient) showBasedModelUsable(ctx context.Context, name string) (bool, error) {
	if name == "" {
		return false, nil
	}

	info, err := c.apiClient.Show(ctx, &api.ShowRequest{Model: name})
	if err != nil {
		var statusErr api.StatusError
		if errors.As(err, &statusErr) && statusErr.StatusCode == http.StatusNotFound {
			return false, nil
		}
		return false, err
	}

	if isCloudModelName(name) || info.RemoteModel != "" {
		cloudDisabled, _ := cloudStatusDisabled(ctx, c.apiClient)

		return !cloudDisabled, nil
	}

	return true, nil
}

func (c *launcherClient) singleModelUsable(ctx context.Context, name string) bool {
	if name == "" {
		return false
	}
	if isCloudModelName(name) {
		cloudDisabled, _ := cloudStatusDisabled(ctx, c.apiClient)
		return !cloudDisabled
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

func (c *launcherClient) loadModelInventoryOnce(ctx context.Context) error {
	if c.inventoryLoaded {
		return nil
	}

	resp, err := c.apiClient.List(ctx)
	if err != nil {
		return err
	}

	c.modelInventory = c.modelInventory[:0]
	for _, model := range resp.Models {
		c.modelInventory = append(c.modelInventory, ModelInfo{
			Name:   model.Name,
			Remote: model.RemoteModel != "",
		})
	}

	cloudDisabled, _ := cloudStatusDisabled(ctx, c.apiClient)
	if cloudDisabled {
		c.modelInventory = filterCloudModels(c.modelInventory)
	}
	c.inventoryLoaded = true
	return nil
}

func runIntegration(runner Runner, modelName string, args []string) error {
	return runner.Run(modelName, args)
}

func launchAfterConfiguration(name string, runner Runner, model string, req IntegrationLaunchRequest) error {
	if req.ConfigureOnly {
		launch, err := ConfirmPrompt(fmt.Sprintf("Launch %s now?", runner))
		if err != nil {
			return err
		}
		if !launch {
			return nil
		}
	}
	if err := EnsureIntegrationInstalled(name, runner); err != nil {
		return err
	}
	return runIntegration(runner, model, req.ExtraArgs)
}

func loadStoredIntegrationConfig(name string) (*config.IntegrationConfig, error) {
	cfg, err := config.LoadIntegration(name)
	if err == nil {
		return cfg, nil
	}
	if !errors.Is(err, os.ErrNotExist) {
		return nil, err
	}

	spec, specErr := LookupIntegrationSpec(name)
	if specErr != nil {
		return nil, err
	}

	for _, alias := range spec.Aliases {
		legacy, legacyErr := config.LoadIntegration(alias)
		if legacyErr == nil {
			migrateLegacyIntegrationConfig(spec.Name, legacy)
			if migrated, migratedErr := config.LoadIntegration(spec.Name); migratedErr == nil {
				return migrated, nil
			}
			return legacy, nil
		}
		if legacyErr != nil && !errors.Is(legacyErr, os.ErrNotExist) {
			return nil, legacyErr
		}
	}

	return nil, err
}

func migrateLegacyIntegrationConfig(canonical string, legacy *config.IntegrationConfig) {
	if legacy == nil {
		return
	}

	_ = config.SaveIntegration(canonical, append([]string(nil), legacy.Models...))
	if len(legacy.Aliases) > 0 {
		_ = config.SaveAliases(canonical, cloneAliases(legacy.Aliases))
	}
	if legacy.Onboarded {
		_ = config.MarkIntegrationOnboarded(canonical)
	}
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

func firstModel(models []string) string {
	if len(models) == 0 {
		return ""
	}
	return models[0]
}

func savedMatchesModels(saved *config.IntegrationConfig, models []string) bool {
	if saved == nil {
		return false
	}
	return slices.Equal(saved.Models, models)
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
