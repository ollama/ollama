package config

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"runtime"
	"slices"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
	internalcloud "github.com/ollama/ollama/internal/cloud"
	"github.com/ollama/ollama/progress"
	"github.com/spf13/cobra"
)

// Runners execute the launching of a model with the integration - claude, codex
// Editors can edit config files (supports multi-model selection) - opencode, droid
// They are composable interfaces where in some cases an editor is also a runner - opencode, droid
// Runner can run an integration with a model.

type Runner interface {
	Run(model string, args []string) error
	// String returns the human-readable name of the integration
	String() string
}

// Editor can edit config files (supports multi-model selection)
type Editor interface {
	// Paths returns the paths to the config files for the integration
	Paths() []string
	// Edit updates the config files for the integration with the given models
	Edit(models []string) error
	// Models returns the models currently configured for the integration
	// TODO(parthsareen): add error return to Models()
	Models() []string
}

// AliasConfigurer can configure model aliases (e.g., for subagent routing).
// Integrations like Claude and Codex use this to route model requests to local models.
type AliasConfigurer interface {
	// ConfigureAliases prompts the user to configure aliases and returns the updated map.
	ConfigureAliases(ctx context.Context, primaryModel string, existing map[string]string, force bool) (map[string]string, bool, error)
	// SetAliases syncs the configured aliases to the server
	SetAliases(ctx context.Context, aliases map[string]string) error
}

// integrations is the registry of available integrations.
var integrations = map[string]Runner{
	"claude":   &Claude{},
	"clawdbot": &Openclaw{},
	"cline":    &Cline{},
	"codex":    &Codex{},
	"moltbot":  &Openclaw{},
	"droid":    &Droid{},
	"opencode": &OpenCode{},
	"openclaw": &Openclaw{},
	"pi":       &Pi{},
}

// recommendedModels are shown when the user has no models or as suggestions.
// Order matters: local models first, then cloud models.
var recommendedModels = []ModelItem{
	{Name: "minimax-m2.5:cloud", Description: "Fast, efficient coding and real-world productivity", Recommended: true},
	{Name: "glm-5:cloud", Description: "Reasoning and code generation", Recommended: true},
	{Name: "kimi-k2.5:cloud", Description: "Multimodal reasoning with subagents", Recommended: true},
	{Name: "glm-4.7-flash", Description: "Reasoning and code generation locally", Recommended: true},
	{Name: "qwen3:8b", Description: "Efficient all-purpose assistant", Recommended: true},
}

// cloudModelLimits maps cloud model base names to their token limits.
// TODO(parthsareen): grab context/output limits from model info instead of hardcoding
var cloudModelLimits = map[string]cloudModelLimit{
	"minimax-m2.5":        {Context: 204_800, Output: 128_000},
	"cogito-2.1:671b":     {Context: 163_840, Output: 65_536},
	"deepseek-v3.1:671b":  {Context: 163_840, Output: 163_840},
	"deepseek-v3.2":       {Context: 163_840, Output: 65_536},
	"glm-4.6":             {Context: 202_752, Output: 131_072},
	"glm-4.7":             {Context: 202_752, Output: 131_072},
	"gpt-oss:120b":        {Context: 131_072, Output: 131_072},
	"gpt-oss:20b":         {Context: 131_072, Output: 131_072},
	"kimi-k2:1t":          {Context: 262_144, Output: 262_144},
	"kimi-k2.5":           {Context: 262_144, Output: 262_144},
	"kimi-k2-thinking":    {Context: 262_144, Output: 262_144},
	"nemotron-3-nano:30b": {Context: 1_048_576, Output: 131_072},
	"qwen3-coder:480b":    {Context: 262_144, Output: 65_536},
	"qwen3-coder-next":    {Context: 262_144, Output: 32_768},
	"qwen3-next:80b":      {Context: 262_144, Output: 32_768},
}

// recommendedVRAM maps local recommended models to their approximate VRAM requirement.
var recommendedVRAM = map[string]string{
	"glm-4.7-flash": "~25GB",
	"qwen3:8b":      "~11GB",
}

// integrationAliases are hidden from the interactive selector but work as CLI arguments.
var integrationAliases = map[string]bool{
	"clawdbot": true,
	"moltbot":  true,
}

// integrationInstallHints maps integration names to install URLs.
var integrationInstallHints = map[string]string{
	"claude":   "https://code.claude.com/docs/en/quickstart",
	"cline":    "https://cline.bot/cli",
	"openclaw": "https://docs.openclaw.ai",
	"codex":    "https://developers.openai.com/codex/cli/",
	"droid":    "https://docs.factory.ai/cli/getting-started/quickstart",
	"opencode": "https://opencode.ai",
	"pi":       "https://github.com/badlogic/pi-mono",
}

// hyperlink wraps text in an OSC 8 terminal hyperlink so it is cmd+clickable.
func hyperlink(url, text string) string {
	return fmt.Sprintf("\033]8;;%s\033\\%s\033]8;;\033\\", url, text)
}

// IntegrationInfo contains display information about a registered integration.
type IntegrationInfo struct {
	Name        string // registry key, e.g. "claude"
	DisplayName string // human-readable, e.g. "Claude Code"
	Description string // short description, e.g. "Anthropic's agentic coding tool"
}

// integrationDescriptions maps integration names to short descriptions.
var integrationDescriptions = map[string]string{
	"claude":   "Anthropic's coding tool with subagents",
	"cline":    "Autonomous coding agent with parallel execution",
	"codex":    "OpenAI's open-source coding agent",
	"openclaw": "Personal AI with 100+ skills",
	"droid":    "Factory's coding agent across terminal and IDEs",
	"opencode": "Anomaly's open-source coding agent",
	"pi":       "Minimal AI agent toolkit with plugin support",
}

// integrationOrder defines a custom display order for integrations.
// Integrations listed here are placed at the end in the given order;
// all others appear first, sorted alphabetically.
var integrationOrder = []string{"opencode", "droid", "pi", "cline"}

// ListIntegrationInfos returns all non-alias registered integrations, sorted by name
// with integrationOrder entries placed at the end.
func ListIntegrationInfos() []IntegrationInfo {
	var result []IntegrationInfo
	for name, r := range integrations {
		if integrationAliases[name] {
			continue
		}
		result = append(result, IntegrationInfo{
			Name:        name,
			DisplayName: r.String(),
			Description: integrationDescriptions[name],
		})
	}

	orderRank := make(map[string]int, len(integrationOrder))
	for i, name := range integrationOrder {
		orderRank[name] = i + 1 // 1-indexed so 0 means "not in the list"
	}

	slices.SortFunc(result, func(a, b IntegrationInfo) int {
		aRank, bRank := orderRank[a.Name], orderRank[b.Name]
		// Both have custom order: sort by their rank
		if aRank > 0 && bRank > 0 {
			return aRank - bRank
		}
		// Only one has custom order: it goes last
		if aRank > 0 {
			return 1
		}
		if bRank > 0 {
			return -1
		}
		// Neither has custom order: alphabetical
		return strings.Compare(a.Name, b.Name)
	})
	return result
}

// IntegrationInstallHint returns a user-friendly install hint for the given integration,
// or an empty string if none is available. The URL is wrapped in an OSC 8 hyperlink
// so it is cmd+clickable in supported terminals.
func IntegrationInstallHint(name string) string {
	url := integrationInstallHints[name]
	if url == "" {
		return ""
	}
	return "Install from " + hyperlink(url, url)
}

// IsIntegrationInstalled checks if an integration binary is installed.
func IsIntegrationInstalled(name string) bool {
	switch name {
	case "claude":
		c := &Claude{}
		_, err := c.findPath()
		return err == nil
	case "openclaw":
		if _, err := exec.LookPath("openclaw"); err == nil {
			return true
		}
		if _, err := exec.LookPath("clawdbot"); err == nil {
			return true
		}
		return false
	case "codex":
		_, err := exec.LookPath("codex")
		return err == nil
	case "droid":
		_, err := exec.LookPath("droid")
		return err == nil
	case "cline":
		_, err := exec.LookPath("cline")
		return err == nil
	case "opencode":
		_, err := exec.LookPath("opencode")
		return err == nil
	case "pi":
		_, err := exec.LookPath("pi")
		return err == nil
	default:
		return true // Assume installed for unknown integrations
	}
}

// IsEditorIntegration returns true if the named integration uses multi-model
// selection (implements the Editor interface).
func IsEditorIntegration(name string) bool {
	r, ok := integrations[strings.ToLower(name)]
	if !ok {
		return false
	}
	_, isEditor := r.(Editor)
	return isEditor
}

// SelectModel lets the user select a model to run.
// ModelItem represents a model for selection.
type ModelItem struct {
	Name        string
	Description string
	Recommended bool
}

// SingleSelector is a function type for single item selection.
type SingleSelector func(title string, items []ModelItem) (string, error)

// MultiSelector is a function type for multi item selection.
type MultiSelector func(title string, items []ModelItem, preChecked []string) ([]string, error)

// SelectModelWithSelector prompts the user to select a model using the provided selector.
func SelectModelWithSelector(ctx context.Context, selector SingleSelector) (string, error) {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return "", err
	}

	models, err := client.List(ctx)
	if err != nil {
		return "", err
	}

	var existing []modelInfo
	for _, m := range models.Models {
		existing = append(existing, modelInfo{Name: m.Name, Remote: m.RemoteModel != ""})
	}

	cloudDisabled, _ := cloudStatusDisabled(ctx, client)
	if cloudDisabled {
		existing = filterCloudModels(existing)
	}

	lastModel := LastModel()
	var preChecked []string
	if lastModel != "" {
		preChecked = []string{lastModel}
	}

	items, _, existingModels, cloudModels := buildModelList(existing, preChecked, lastModel)

	if cloudDisabled {
		items = filterCloudItems(items)
	}

	if len(items) == 0 {
		return "", fmt.Errorf("no models available, run 'ollama pull <model>' first")
	}

	selected, err := selector("Select model to run:", items)
	if err != nil {
		return "", err
	}

	// If the selected model isn't installed, pull it first
	if !existingModels[selected] {
		if cloudModels[selected] {
			// Cloud models only pull a small manifest; no confirmation needed
			if err := pullModel(ctx, client, selected); err != nil {
				return "", fmt.Errorf("failed to pull %s: %w", selected, err)
			}
		} else {
			msg := fmt.Sprintf("Download %s?", selected)
			if ok, err := confirmPrompt(msg); err != nil {
				return "", err
			} else if !ok {
				return "", errCancelled
			}
			fmt.Fprintf(os.Stderr, "\n")
			if err := pullModel(ctx, client, selected); err != nil {
				return "", fmt.Errorf("failed to pull %s: %w", selected, err)
			}
		}
	}

	// If it's a cloud model, ensure user is signed in
	if cloudModels[selected] {
		user, err := client.Whoami(ctx)
		if err == nil && user != nil && user.Name != "" {
			return selected, nil
		}

		var aErr api.AuthorizationError
		if !errors.As(err, &aErr) || aErr.SigninURL == "" {
			return "", err
		}

		yes, err := confirmPrompt(fmt.Sprintf("sign in to use %s?", selected))
		if err != nil || !yes {
			return "", fmt.Errorf("%s requires sign in", selected)
		}

		fmt.Fprintf(os.Stderr, "\nTo sign in, navigate to:\n    %s\n\n", aErr.SigninURL)

		// Auto-open browser (best effort, fail silently)
		switch runtime.GOOS {
		case "darwin":
			_ = exec.Command("open", aErr.SigninURL).Start()
		case "linux":
			_ = exec.Command("xdg-open", aErr.SigninURL).Start()
		case "windows":
			_ = exec.Command("rundll32", "url.dll,FileProtocolHandler", aErr.SigninURL).Start()
		}

		spinnerFrames := []string{"|", "/", "-", "\\"}
		frame := 0

		fmt.Fprintf(os.Stderr, "\033[90mwaiting for sign in to complete... %s\033[0m", spinnerFrames[0])

		ticker := time.NewTicker(200 * time.Millisecond)
		defer ticker.Stop()

		for {
			select {
			case <-ctx.Done():
				fmt.Fprintf(os.Stderr, "\r\033[K")
				return "", ctx.Err()
			case <-ticker.C:
				frame++
				fmt.Fprintf(os.Stderr, "\r\033[90mwaiting for sign in to complete... %s\033[0m", spinnerFrames[frame%len(spinnerFrames)])

				// poll every 10th frame (~2 seconds)
				if frame%10 == 0 {
					u, err := client.Whoami(ctx)
					if err == nil && u != nil && u.Name != "" {
						fmt.Fprintf(os.Stderr, "\r\033[K\033[A\r\033[K\033[1msigned in:\033[0m %s\n", u.Name)
						return selected, nil
					}
				}
			}
		}
	}

	return selected, nil
}

func SelectModel(ctx context.Context) (string, error) {
	return SelectModelWithSelector(ctx, DefaultSingleSelector)
}

// DefaultSingleSelector is the default single-select implementation.
var DefaultSingleSelector SingleSelector

// DefaultMultiSelector is the default multi-select implementation.
var DefaultMultiSelector MultiSelector

// DefaultSignIn provides a TUI-based sign-in flow.
// When set, ensureAuth uses it instead of plain text prompts.
// Returns the signed-in username or an error.
var DefaultSignIn func(modelName, signInURL string) (string, error)

func selectIntegration() (string, error) {
	if DefaultSingleSelector == nil {
		return "", fmt.Errorf("no selector configured")
	}
	if len(integrations) == 0 {
		return "", fmt.Errorf("no integrations available")
	}

	var items []ModelItem
	for name, r := range integrations {
		if integrationAliases[name] {
			continue
		}
		description := r.String()
		if conn, err := loadIntegration(name); err == nil && len(conn.Models) > 0 {
			description = fmt.Sprintf("%s (%s)", r.String(), conn.Models[0])
		}
		items = append(items, ModelItem{Name: name, Description: description})
	}

	orderRank := make(map[string]int, len(integrationOrder))
	for i, name := range integrationOrder {
		orderRank[name] = i + 1
	}
	slices.SortFunc(items, func(a, b ModelItem) int {
		aRank, bRank := orderRank[a.Name], orderRank[b.Name]
		if aRank > 0 && bRank > 0 {
			return aRank - bRank
		}
		if aRank > 0 {
			return 1
		}
		if bRank > 0 {
			return -1
		}
		return strings.Compare(a.Name, b.Name)
	})

	return DefaultSingleSelector("Select integration:", items)
}

// selectModelsWithSelectors lets the user select models for an integration using provided selectors.
func selectModelsWithSelectors(ctx context.Context, name, current string, single SingleSelector, multi MultiSelector) ([]string, error) {
	r, ok := integrations[name]
	if !ok {
		return nil, fmt.Errorf("unknown integration: %s", name)
	}

	client, err := api.ClientFromEnvironment()
	if err != nil {
		return nil, err
	}

	models, err := client.List(ctx)
	if err != nil {
		return nil, err
	}

	var existing []modelInfo
	for _, m := range models.Models {
		existing = append(existing, modelInfo{Name: m.Name, Remote: m.RemoteModel != ""})
	}

	cloudDisabled, _ := cloudStatusDisabled(ctx, client)
	if cloudDisabled {
		existing = filterCloudModels(existing)
	}

	var preChecked []string
	if saved, err := loadIntegration(name); err == nil {
		preChecked = saved.Models
	} else if editor, ok := r.(Editor); ok {
		preChecked = editor.Models()
	}

	items, preChecked, existingModels, cloudModels := buildModelList(existing, preChecked, current)

	if cloudDisabled {
		items = filterCloudItems(items)
	}

	if len(items) == 0 {
		return nil, fmt.Errorf("no models available")
	}

	var selected []string
	if _, ok := r.(Editor); ok {
		selected, err = multi(fmt.Sprintf("Select models for %s:", r), items, preChecked)
		if err != nil {
			return nil, err
		}
	} else {
		prompt := fmt.Sprintf("Select model for %s:", r)
		if _, ok := r.(AliasConfigurer); ok {
			prompt = fmt.Sprintf("Select Primary model for %s:", r)
		}
		model, err := single(prompt, items)
		if err != nil {
			return nil, err
		}
		selected = []string{model}
	}

	var toPull []string
	for _, m := range selected {
		if !existingModels[m] {
			toPull = append(toPull, m)
		}
	}
	if len(toPull) > 0 {
		msg := fmt.Sprintf("Download %s?", strings.Join(toPull, ", "))
		if ok, err := confirmPrompt(msg); err != nil {
			return nil, err
		} else if !ok {
			return nil, errCancelled
		}
		for _, m := range toPull {
			fmt.Fprintf(os.Stderr, "\n")
			if err := pullModel(ctx, client, m); err != nil {
				return nil, fmt.Errorf("failed to pull %s: %w", m, err)
			}
		}
	}

	if err := ensureAuth(ctx, client, cloudModels, selected); err != nil {
		return nil, err
	}

	return selected, nil
}

func pullIfNeeded(ctx context.Context, client *api.Client, existingModels map[string]bool, model string) error {
	if existingModels[model] {
		return nil
	}
	msg := fmt.Sprintf("Download %s?", model)
	if ok, err := confirmPrompt(msg); err != nil {
		return err
	} else if !ok {
		return errCancelled
	}
	fmt.Fprintf(os.Stderr, "\n")
	if err := pullModel(ctx, client, model); err != nil {
		return fmt.Errorf("failed to pull %s: %w", model, err)
	}
	return nil
}

// TODO(parthsareen): pull this out to tui package
// ShowOrPull checks if a model exists via client.Show and offers to pull it if not found.
func ShowOrPull(ctx context.Context, client *api.Client, model string) error {
	if _, err := client.Show(ctx, &api.ShowRequest{Model: model}); err == nil {
		return nil
	}
	// Cloud models only pull a small manifest; skip the download confirmation
	// TODO(parthsareen): consolidate with cloud config changes
	if strings.HasSuffix(model, "cloud") {
		return pullModel(ctx, client, model)
	}
	if ok, err := confirmPrompt(fmt.Sprintf("Download %s?", model)); err != nil {
		return err
	} else if !ok {
		return errCancelled
	}
	fmt.Fprintf(os.Stderr, "\n")
	return pullModel(ctx, client, model)
}

func listModels(ctx context.Context) ([]ModelItem, map[string]bool, map[string]bool, *api.Client, error) {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return nil, nil, nil, nil, err
	}

	models, err := client.List(ctx)
	if err != nil {
		return nil, nil, nil, nil, err
	}

	var existing []modelInfo
	for _, m := range models.Models {
		existing = append(existing, modelInfo{
			Name:   m.Name,
			Remote: m.RemoteModel != "",
		})
	}

	cloudDisabled, _ := cloudStatusDisabled(ctx, client)
	if cloudDisabled {
		existing = filterCloudModels(existing)
	}

	items, _, existingModels, cloudModels := buildModelList(existing, nil, "")

	if cloudDisabled {
		items = filterCloudItems(items)
	}

	if len(items) == 0 {
		return nil, nil, nil, nil, fmt.Errorf("no models available, run 'ollama pull <model>' first")
	}

	return items, existingModels, cloudModels, client, nil
}

func OpenBrowser(url string) {
	switch runtime.GOOS {
	case "darwin":
		_ = exec.Command("open", url).Start()
	case "linux":
		_ = exec.Command("xdg-open", url).Start()
	case "windows":
		_ = exec.Command("rundll32", "url.dll,FileProtocolHandler", url).Start()
	}
}

func ensureAuth(ctx context.Context, client *api.Client, cloudModels map[string]bool, selected []string) error {
	var selectedCloudModels []string
	for _, m := range selected {
		if cloudModels[m] {
			selectedCloudModels = append(selectedCloudModels, m)
		}
	}
	if len(selectedCloudModels) == 0 {
		return nil
	}
	if disabled, known := cloudStatusDisabled(ctx, client); known && disabled {
		return errors.New(internalcloud.DisabledError("remote inference is unavailable"))
	}

	user, err := client.Whoami(ctx)
	if err == nil && user != nil && user.Name != "" {
		return nil
	}

	var aErr api.AuthorizationError
	if !errors.As(err, &aErr) || aErr.SigninURL == "" {
		return err
	}

	modelList := strings.Join(selectedCloudModels, ", ")

	if DefaultSignIn != nil {
		_, err := DefaultSignIn(modelList, aErr.SigninURL)
		if err != nil {
			return fmt.Errorf("%s requires sign in", modelList)
		}
		return nil
	}

	// Fallback: plain text sign-in flow
	yes, err := confirmPrompt(fmt.Sprintf("sign in to use %s?", modelList))
	if err != nil || !yes {
		return fmt.Errorf("%s requires sign in", modelList)
	}

	fmt.Fprintf(os.Stderr, "\nTo sign in, navigate to:\n    %s\n\n", aErr.SigninURL)

	OpenBrowser(aErr.SigninURL)

	spinnerFrames := []string{"|", "/", "-", "\\"}
	frame := 0

	fmt.Fprintf(os.Stderr, "\033[90mwaiting for sign in to complete... %s\033[0m", spinnerFrames[0])

	ticker := time.NewTicker(200 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			fmt.Fprintf(os.Stderr, "\r\033[K")
			return ctx.Err()
		case <-ticker.C:
			frame++
			fmt.Fprintf(os.Stderr, "\r\033[90mwaiting for sign in to complete... %s\033[0m", spinnerFrames[frame%len(spinnerFrames)])

			// poll every 10th frame (~2 seconds)
			if frame%10 == 0 {
				u, err := client.Whoami(ctx)
				if err == nil && u != nil && u.Name != "" {
					fmt.Fprintf(os.Stderr, "\r\033[K\033[A\r\033[K\033[1msigned in:\033[0m %s\n", u.Name)
					return nil
				}
			}
		}
	}
}

// selectModels lets the user select models for an integration using default selectors.
func selectModels(ctx context.Context, name, current string) ([]string, error) {
	return selectModelsWithSelectors(ctx, name, current, DefaultSingleSelector, DefaultMultiSelector)
}

func runIntegration(name, modelName string, args []string) error {
	r, ok := integrations[name]
	if !ok {
		return fmt.Errorf("unknown integration: %s", name)
	}

	fmt.Fprintf(os.Stderr, "\nLaunching %s with %s...\n", r, modelName)
	return r.Run(modelName, args)
}

// syncAliases syncs aliases to server and saves locally for an AliasConfigurer.
func syncAliases(ctx context.Context, client *api.Client, ac AliasConfigurer, name, model string, existing map[string]string) error {
	aliases := make(map[string]string)
	for k, v := range existing {
		aliases[k] = v
	}
	aliases["primary"] = model

	if isCloudModel(ctx, client, model) {
		if aliases["fast"] == "" || !isCloudModel(ctx, client, aliases["fast"]) {
			aliases["fast"] = model
		}
	} else {
		delete(aliases, "fast")
	}

	if err := ac.SetAliases(ctx, aliases); err != nil {
		return err
	}
	return saveAliases(name, aliases)
}

// LaunchIntegration launches the named integration using saved config or prompts for setup.
func LaunchIntegration(name string) error {
	r, ok := integrations[name]
	if !ok {
		return fmt.Errorf("unknown integration: %s", name)
	}

	// Try to use saved config
	if ic, err := loadIntegration(name); err == nil && len(ic.Models) > 0 {
		client, err := api.ClientFromEnvironment()
		if err != nil {
			return err
		}
		if err := ShowOrPull(context.Background(), client, ic.Models[0]); err != nil {
			return err
		}
		return runIntegration(name, ic.Models[0], nil)
	}

	// No saved config - prompt user to run setup
	return fmt.Errorf("%s is not configured. Run 'ollama launch %s' to set it up", r, name)
}

// LaunchIntegrationWithModel launches the named integration with the specified model.
func LaunchIntegrationWithModel(name, modelName string) error {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}
	if err := ShowOrPull(context.Background(), client, modelName); err != nil {
		return err
	}
	return runIntegration(name, modelName, nil)
}

// SaveAndEditIntegration saves the models for an Editor integration and runs its Edit method
// to write the integration's config files.
func SaveAndEditIntegration(name string, models []string) error {
	r, ok := integrations[strings.ToLower(name)]
	if !ok {
		return fmt.Errorf("unknown integration: %s", name)
	}
	if err := SaveIntegration(name, models); err != nil {
		return fmt.Errorf("failed to save: %w", err)
	}
	if editor, isEditor := r.(Editor); isEditor {
		if err := editor.Edit(models); err != nil {
			return fmt.Errorf("setup failed: %w", err)
		}
	}
	return nil
}

// resolveEditorModels filters out cloud-disabled models before editor launch.
// If no models remain, it invokes picker to collect a valid replacement list.
func resolveEditorModels(name string, models []string, picker func() ([]string, error)) ([]string, error) {
	filtered := filterDisabledCloudModels(models)
	if len(filtered) != len(models) {
		if err := SaveIntegration(name, filtered); err != nil {
			return nil, fmt.Errorf("failed to save: %w", err)
		}
	}
	if len(filtered) > 0 {
		return filtered, nil
	}

	selected, err := picker()
	if err != nil {
		return nil, err
	}
	if err := SaveIntegration(name, selected); err != nil {
		return nil, fmt.Errorf("failed to save: %w", err)
	}
	return selected, nil
}

// ConfigureIntegrationWithSelectors allows the user to select/change the model for an integration using custom selectors.
func ConfigureIntegrationWithSelectors(ctx context.Context, name string, single SingleSelector, multi MultiSelector) error {
	r, ok := integrations[name]
	if !ok {
		return fmt.Errorf("unknown integration: %s", name)
	}

	models, err := selectModelsWithSelectors(ctx, name, "", single, multi)
	if errors.Is(err, errCancelled) {
		return errCancelled
	}
	if err != nil {
		return err
	}

	if editor, isEditor := r.(Editor); isEditor {
		paths := editor.Paths()
		if len(paths) > 0 {
			fmt.Fprintf(os.Stderr, "This will modify your %s configuration:\n", r)
			for _, p := range paths {
				fmt.Fprintf(os.Stderr, "  %s\n", p)
			}
			fmt.Fprintf(os.Stderr, "Backups will be saved to %s/\n\n", backupDir())

			if ok, _ := confirmPrompt("Proceed?"); !ok {
				return nil
			}
		}

		if err := editor.Edit(models); err != nil {
			return fmt.Errorf("setup failed: %w", err)
		}
	}

	if err := SaveIntegration(name, models); err != nil {
		return fmt.Errorf("failed to save: %w", err)
	}

	if len(models) == 1 {
		fmt.Fprintf(os.Stderr, "Configured %s with %s\n", r, models[0])
	} else {
		fmt.Fprintf(os.Stderr, "Configured %s with %d models (default: %s)\n", r, len(models), models[0])
	}

	return nil
}

// ConfigureIntegration allows the user to select/change the model for an integration.
func ConfigureIntegration(ctx context.Context, name string) error {
	return ConfigureIntegrationWithSelectors(ctx, name, DefaultSingleSelector, DefaultMultiSelector)
}

// LaunchCmd returns the cobra command for launching integrations.
// The runTUI callback is called when no arguments are provided (alias for main TUI).
func LaunchCmd(checkServerHeartbeat func(cmd *cobra.Command, args []string) error, runTUI func(cmd *cobra.Command)) *cobra.Command {
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
			// No args and no flags - show the full TUI (same as bare 'ollama')
			if len(args) == 0 && modelFlag == "" && !configFlag {
				runTUI(cmd)
				return nil
			}

			// Extract integration name and args to pass through using -- separator
			var name string
			var passArgs []string
			dashIdx := cmd.ArgsLenAtDash()

			if dashIdx == -1 {
				// No "--" separator: only allow 0 or 1 args (integration name)
				if len(args) > 1 {
					return fmt.Errorf("unexpected arguments: %v\nUse '--' to pass extra arguments to the integration", args[1:])
				}
				if len(args) == 1 {
					name = args[0]
				}
			} else {
				// "--" was used: args before it = integration name, args after = passthrough
				if dashIdx > 1 {
					return fmt.Errorf("expected at most 1 integration name before '--', got %d", dashIdx)
				}
				if dashIdx == 1 {
					name = args[0]
				}
				passArgs = args[dashIdx:]
			}

			if name == "" {
				var err error
				name, err = selectIntegration()
				if errors.Is(err, errCancelled) {
					return nil
				}
				if err != nil {
					return err
				}
			}

			r, ok := integrations[strings.ToLower(name)]
			if !ok {
				return fmt.Errorf("unknown integration: %s", name)
			}

			if modelFlag != "" && IsCloudModelDisabled(cmd.Context(), modelFlag) {
				modelFlag = ""
			}

			// Handle AliasConfigurer integrations (claude, codex)
			if ac, ok := r.(AliasConfigurer); ok {
				client, err := api.ClientFromEnvironment()
				if err != nil {
					return err
				}

				// Validate --model flag if provided
				if modelFlag != "" {
					if err := ShowOrPull(cmd.Context(), client, modelFlag); err != nil {
						if errors.Is(err, errCancelled) {
							return nil
						}
						return err
					}
				}

				var model string
				var existingAliases map[string]string

				// Load saved config
				if cfg, err := loadIntegration(name); err == nil {
					existingAliases = cfg.Aliases
					if len(cfg.Models) > 0 {
						model = cfg.Models[0]
						// AliasConfigurer integrations use single model; sanitize if multiple
						if len(cfg.Models) > 1 {
							_ = SaveIntegration(name, []string{model})
						}
					}
				}

				// --model flag overrides saved model
				if modelFlag != "" {
					model = modelFlag
				}

				// Validate saved model still exists
				cloudCleared := false
				if model != "" && modelFlag == "" {
					if disabled, _ := cloudStatusDisabled(cmd.Context(), client); disabled && isCloudModelName(model) {
						model = ""
						cloudCleared = true
					} else if _, err := client.Show(cmd.Context(), &api.ShowRequest{Model: model}); err != nil {
						fmt.Fprintf(os.Stderr, "%sConfigured model %q not found%s\n\n", ansiGray, model, ansiReset)
						if err := ShowOrPull(cmd.Context(), client, model); err != nil {
							model = ""
						}
					}
				}

				// If no valid model or --config flag, show picker
				if model == "" || configFlag {
					aliases, _, err := ac.ConfigureAliases(cmd.Context(), model, existingAliases, configFlag || cloudCleared)
					if errors.Is(err, errCancelled) {
						return nil
					}
					if err != nil {
						return err
					}
					model = aliases["primary"]
					existingAliases = aliases
				}

				// Ensure cloud models are authenticated
				if isCloudModel(cmd.Context(), client, model) {
					if err := ensureAuth(cmd.Context(), client, map[string]bool{model: true}, []string{model}); err != nil {
						return err
					}
				}

				// Sync aliases and save
				if err := syncAliases(cmd.Context(), client, ac, name, model, existingAliases); err != nil {
					fmt.Fprintf(os.Stderr, "%sWarning: Could not sync aliases: %v%s\n", ansiGray, err, ansiReset)
				}
				if err := SaveIntegration(name, []string{model}); err != nil {
					return fmt.Errorf("failed to save: %w", err)
				}

				// Launch (unless --config without confirmation)
				if configFlag {
					if launch, _ := confirmPrompt(fmt.Sprintf("Launch %s now?", r)); launch {
						return runIntegration(name, model, passArgs)
					}
					return nil
				}
				return runIntegration(name, model, passArgs)
			}

			// Validate --model flag for non-AliasConfigurer integrations
			if modelFlag != "" {
				client, err := api.ClientFromEnvironment()
				if err != nil {
					return err
				}
				if err := ShowOrPull(cmd.Context(), client, modelFlag); err != nil {
					if errors.Is(err, errCancelled) {
						return nil
					}
					return err
				}
			}

			var models []string
			if modelFlag != "" {
				models = []string{modelFlag}
				if existing, err := loadIntegration(name); err == nil && len(existing.Models) > 0 {
					for _, m := range existing.Models {
						if m != modelFlag {
							models = append(models, m)
						}
					}
				}
				models = filterDisabledCloudModels(models)
				if len(models) == 0 {
					var err error
					models, err = selectModels(cmd.Context(), name, "")
					if errors.Is(err, errCancelled) {
						return nil
					}
					if err != nil {
						return err
					}
				}
			} else if saved, err := loadIntegration(name); err == nil && len(saved.Models) > 0 && !configFlag {
				savedModels := filterDisabledCloudModels(saved.Models)
				if len(savedModels) != len(saved.Models) {
					_ = SaveIntegration(name, savedModels)
				}
				if len(savedModels) == 0 {
					// All saved models were cloud — fall through to picker
					models, err = selectModels(cmd.Context(), name, "")
					if errors.Is(err, errCancelled) {
						return nil
					}
					if err != nil {
						return err
					}
				} else {
					models = savedModels
					return runIntegration(name, models[0], passArgs)
				}
			} else {
				var err error
				models, err = selectModels(cmd.Context(), name, "")
				if errors.Is(err, errCancelled) {
					return nil
				}
				if err != nil {
					return err
				}
			}

			if editor, isEditor := r.(Editor); isEditor {
				paths := editor.Paths()
				if len(paths) > 0 {
					fmt.Fprintf(os.Stderr, "This will modify your %s configuration:\n", r)
					for _, p := range paths {
						fmt.Fprintf(os.Stderr, "  %s\n", p)
					}
					fmt.Fprintf(os.Stderr, "Backups will be saved to %s/\n\n", backupDir())

					if ok, _ := confirmPrompt("Proceed?"); !ok {
						return nil
					}
				}
			}

			if err := SaveIntegration(name, models); err != nil {
				return fmt.Errorf("failed to save: %w", err)
			}

			if editor, isEditor := r.(Editor); isEditor {
				if err := editor.Edit(models); err != nil {
					return fmt.Errorf("setup failed: %w", err)
				}
			}

			if _, isEditor := r.(Editor); isEditor {
				if len(models) == 1 {
					fmt.Fprintf(os.Stderr, "Added %s to %s\n", models[0], r)
				} else {
					fmt.Fprintf(os.Stderr, "Added %d models to %s (default: %s)\n", len(models), r, models[0])
				}
			}

			if configFlag {
				if launch, _ := confirmPrompt(fmt.Sprintf("\nLaunch %s now?", r)); launch {
					return runIntegration(name, models[0], passArgs)
				}
				fmt.Fprintf(os.Stderr, "Run 'ollama launch %s' to start with %s\n", strings.ToLower(name), models[0])
				return nil
			}

			return runIntegration(name, models[0], passArgs)
		},
	}

	cmd.Flags().StringVar(&modelFlag, "model", "", "Model to use")
	cmd.Flags().BoolVar(&configFlag, "config", false, "Configure without launching")
	return cmd
}

type modelInfo struct {
	Name        string
	Remote      bool
	ToolCapable bool
}

// buildModelList merges existing models with recommendations, sorts them, and returns
// the ordered items along with maps of existing and cloud model names.
func buildModelList(existing []modelInfo, preChecked []string, current string) (items []ModelItem, orderedChecked []string, existingModels, cloudModels map[string]bool) {
	existingModels = make(map[string]bool)
	cloudModels = make(map[string]bool)
	recommended := make(map[string]bool)
	var hasLocalModel, hasCloudModel bool

	recDesc := make(map[string]string)
	for _, rec := range recommendedModels {
		recommended[rec.Name] = true
		recDesc[rec.Name] = rec.Description
	}

	for _, m := range existing {
		existingModels[m.Name] = true
		if m.Remote {
			cloudModels[m.Name] = true
			hasCloudModel = true
		} else {
			hasLocalModel = true
		}
		displayName := strings.TrimSuffix(m.Name, ":latest")
		existingModels[displayName] = true
		item := ModelItem{Name: displayName, Recommended: recommended[displayName], Description: recDesc[displayName]}
		items = append(items, item)
	}

	for _, rec := range recommendedModels {
		if existingModels[rec.Name] || existingModels[rec.Name+":latest"] {
			continue
		}
		items = append(items, rec)
		if isCloudModelName(rec.Name) {
			cloudModels[rec.Name] = true
		}
	}

	checked := make(map[string]bool, len(preChecked))
	for _, n := range preChecked {
		checked[n] = true
	}

	// Resolve current to full name (e.g., "llama3.2" -> "llama3.2:latest")
	for _, item := range items {
		if item.Name == current || strings.HasPrefix(item.Name, current+":") {
			current = item.Name
			break
		}
	}

	if checked[current] {
		preChecked = append([]string{current}, slices.DeleteFunc(preChecked, func(m string) bool { return m == current })...)
	}

	// Non-existing models get "install?" suffix and are pushed to the bottom.
	// When user has no models, preserve recommended order.
	notInstalled := make(map[string]bool)
	for i := range items {
		if !existingModels[items[i].Name] {
			notInstalled[items[i].Name] = true
			var parts []string
			if items[i].Description != "" {
				parts = append(parts, items[i].Description)
			}
			if vram := recommendedVRAM[items[i].Name]; vram != "" {
				parts = append(parts, vram)
			}
			parts = append(parts, "(not downloaded)")
			items[i].Description = strings.Join(parts, ", ")
		}
	}

	// Build a recommended rank map to preserve ordering within tiers.
	recRank := make(map[string]int)
	for i, rec := range recommendedModels {
		recRank[rec.Name] = i + 1 // 1-indexed; 0 means not recommended
	}

	onlyLocal := hasLocalModel && !hasCloudModel

	if hasLocalModel || hasCloudModel {
		slices.SortStableFunc(items, func(a, b ModelItem) int {
			ac, bc := checked[a.Name], checked[b.Name]
			aNew, bNew := notInstalled[a.Name], notInstalled[b.Name]
			aRec, bRec := recRank[a.Name] > 0, recRank[b.Name] > 0
			aCloud, bCloud := cloudModels[a.Name], cloudModels[b.Name]

			// Checked/pre-selected always first
			if ac != bc {
				if ac {
					return -1
				}
				return 1
			}

			// Recommended above non-recommended
			if aRec != bRec {
				if aRec {
					return -1
				}
				return 1
			}

			// Both recommended
			if aRec && bRec {
				if aCloud != bCloud {
					if onlyLocal {
						// Local before cloud when only local installed
						if aCloud {
							return 1
						}
						return -1
					}
					// Cloud before local in mixed case
					if aCloud {
						return -1
					}
					return 1
				}
				return recRank[a.Name] - recRank[b.Name]
			}

			// Both non-recommended: installed before not-installed
			if aNew != bNew {
				if aNew {
					return 1
				}
				return -1
			}

			return strings.Compare(strings.ToLower(a.Name), strings.ToLower(b.Name))
		})
	}

	return items, preChecked, existingModels, cloudModels
}

// IsCloudModelDisabled reports whether the given model name looks like a cloud
// model and cloud features are currently disabled on the server.
func IsCloudModelDisabled(ctx context.Context, name string) bool {
	if !isCloudModelName(name) {
		return false
	}
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return false
	}
	disabled, _ := cloudStatusDisabled(ctx, client)
	return disabled
}

func isCloudModelName(name string) bool {
	return strings.HasSuffix(name, ":cloud") || strings.HasSuffix(name, "-cloud")
}

func filterCloudModels(existing []modelInfo) []modelInfo {
	filtered := existing[:0]
	for _, m := range existing {
		if !m.Remote {
			filtered = append(filtered, m)
		}
	}
	return filtered
}

// filterDisabledCloudModels removes cloud models from a list when cloud is disabled.
func filterDisabledCloudModels(models []string) []string {
	var filtered []string
	for _, m := range models {
		if !IsCloudModelDisabled(context.Background(), m) {
			filtered = append(filtered, m)
		}
	}
	return filtered
}

func filterCloudItems(items []ModelItem) []ModelItem {
	filtered := items[:0]
	for _, item := range items {
		if !isCloudModelName(item.Name) {
			filtered = append(filtered, item)
		}
	}
	return filtered
}

func isCloudModel(ctx context.Context, client *api.Client, name string) bool {
	if client == nil {
		return false
	}
	resp, err := client.Show(ctx, &api.ShowRequest{Model: name})
	if err != nil {
		return false
	}
	return resp.RemoteModel != ""
}

// GetModelItems returns a list of model items including recommendations for the TUI.
// It includes all locally available models plus recommended models that aren't installed.
func GetModelItems(ctx context.Context) ([]ModelItem, map[string]bool) {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return nil, nil
	}

	models, err := client.List(ctx)
	if err != nil {
		return nil, nil
	}

	var existing []modelInfo
	for _, m := range models.Models {
		existing = append(existing, modelInfo{Name: m.Name, Remote: m.RemoteModel != ""})
	}

	cloudDisabled, _ := cloudStatusDisabled(ctx, client)
	if cloudDisabled {
		existing = filterCloudModels(existing)
	}

	lastModel := LastModel()
	var preChecked []string
	if lastModel != "" {
		preChecked = []string{lastModel}
	}

	items, _, existingModels, _ := buildModelList(existing, preChecked, lastModel)

	if cloudDisabled {
		items = filterCloudItems(items)
	}

	return items, existingModels
}

func cloudStatusDisabled(ctx context.Context, client *api.Client) (disabled bool, known bool) {
	status, err := client.CloudStatusExperimental(ctx)
	if err != nil {
		var statusErr api.StatusError
		if errors.As(err, &statusErr) && statusErr.StatusCode == http.StatusNotFound {
			return false, false
		}
		return false, false
	}
	return status.Cloud.Disabled, true
}

func pullModel(ctx context.Context, client *api.Client, model string) error {
	p := progress.NewProgress(os.Stderr)
	defer p.Stop()

	bars := make(map[string]*progress.Bar)
	var status string
	var spinner *progress.Spinner

	fn := func(resp api.ProgressResponse) error {
		if resp.Digest != "" {
			if resp.Completed == 0 {
				return nil
			}

			if spinner != nil {
				spinner.Stop()
			}

			bar, ok := bars[resp.Digest]
			if !ok {
				name, isDigest := strings.CutPrefix(resp.Digest, "sha256:")
				name = strings.TrimSpace(name)
				if isDigest {
					name = name[:min(12, len(name))]
				}
				bar = progress.NewBar(fmt.Sprintf("pulling %s:", name), resp.Total, resp.Completed)
				bars[resp.Digest] = bar
				p.Add(resp.Digest, bar)
			}

			bar.Set(resp.Completed)
		} else if status != resp.Status {
			if spinner != nil {
				spinner.Stop()
			}

			status = resp.Status
			spinner = progress.NewSpinner(status)
			p.Add(status, spinner)
		}

		return nil
	}

	request := api.PullRequest{Name: model}
	return client.Pull(ctx, &request, fn)
}
