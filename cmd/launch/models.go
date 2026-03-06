package launch

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
	"github.com/ollama/ollama/cmd/config"
	internalcloud "github.com/ollama/ollama/internal/cloud"
	"github.com/ollama/ollama/internal/modelref"
	"github.com/ollama/ollama/progress"
)

var recommendedModels = []ModelItem{
	{Name: "minimax-m2.5:cloud", Description: "Fast, efficient coding and real-world productivity", Recommended: true},
	{Name: "glm-5:cloud", Description: "Reasoning and code generation", Recommended: true},
	{Name: "kimi-k2.5:cloud", Description: "Multimodal reasoning with subagents", Recommended: true},
	{Name: "glm-4.7-flash", Description: "Reasoning and code generation locally", Recommended: true},
	{Name: "qwen3:8b", Description: "Efficient all-purpose assistant", Recommended: true},
}

var recommendedVRAM = map[string]string{
	"glm-4.7-flash": "~25GB",
	"qwen3:8b":      "~11GB",
}

// cloudModelLimit holds context and output token limits for a cloud model.
type cloudModelLimit struct {
	Context int
	Output  int
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
	"glm-5":               {Context: 202_752, Output: 131_072},
	"gpt-oss:120b":        {Context: 131_072, Output: 131_072},
	"gpt-oss:20b":         {Context: 131_072, Output: 131_072},
	"kimi-k2:1t":          {Context: 262_144, Output: 262_144},
	"kimi-k2.5":           {Context: 262_144, Output: 262_144},
	"kimi-k2-thinking":    {Context: 262_144, Output: 262_144},
	"nemotron-3-nano:30b": {Context: 1_048_576, Output: 131_072},
	"qwen3-coder:480b":    {Context: 262_144, Output: 65_536},
	"qwen3-coder-next":    {Context: 262_144, Output: 32_768},
	"qwen3-next:80b":      {Context: 262_144, Output: 32_768},
	"qwen3.5":             {Context: 262_144, Output: 32_768},
}

// lookupCloudModelLimit returns the token limits for a cloud model.
// It normalizes explicit cloud source suffixes before checking the shared limit map.
func lookupCloudModelLimit(name string) (cloudModelLimit, bool) {
	base, stripped := modelref.StripCloudSourceTag(name)
	if stripped {
		if l, ok := cloudModelLimits[base]; ok {
			return l, true
		}
	}
	return cloudModelLimit{}, false
}

// MissingModelPolicy controls how model-not-found errors should be handled.
type MissingModelPolicy int

const (
	// MissingModelPromptPull prompts the user to download missing local models.
	MissingModelPromptPull MissingModelPolicy = iota
	// MissingModelFail returns an error for missing local models without prompting.
	MissingModelFail
)

// OpenBrowser opens the URL in the user's browser.
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

// EnsureAuth ensures the user is signed in before cloud-backed models run.
func EnsureAuth(ctx context.Context, client *api.Client, cloudModels map[string]bool, selected []string) error {
	var selectedCloudModels []string
	for _, m := range selected {
		if cloudModels[m] {
			selectedCloudModels = append(selectedCloudModels, m)
		}
	}
	if len(selectedCloudModels) == 0 {
		return nil
	}
	if disabled, known := CloudStatusDisabled(ctx, client); known && disabled {
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
		if errors.Is(err, ErrCancelled) {
			return ErrCancelled
		}
		if err != nil {
			return fmt.Errorf("%s requires sign in", modelList)
		}
		return nil
	}

	yes, err := ConfirmPrompt(fmt.Sprintf("sign in to use %s?", modelList))
	if errors.Is(err, ErrCancelled) {
		return ErrCancelled
	}
	if err != nil {
		return err
	}
	if !yes {
		return ErrCancelled
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

func selectModelsWithSelectors(ctx context.Context, name, current string, single SingleSelector, multi MultiSelector) ([]string, error) {
	key, runner, err := LookupIntegration(name)
	if err != nil {
		return nil, err
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

	cloudDisabled, _ := CloudStatusDisabled(ctx, client)
	if cloudDisabled {
		existing = FilterCloudModels(existing)
	}

	var preChecked []string
	if saved, err := config.LoadIntegration(key); err == nil {
		preChecked = saved.Models
	} else if editor, ok := runner.(Editor); ok {
		preChecked = editor.Models()
	}

	items, preChecked, existingModels, cloudModels := BuildModelList(existing, preChecked, current)
	if cloudDisabled {
		items = FilterCloudItems(items)
	}
	if len(items) == 0 {
		return nil, fmt.Errorf("no models available")
	}

	var selected []string
	if _, ok := runner.(Editor); ok {
		selected, err = multi(fmt.Sprintf("Select models for %s:", runner), items, preChecked)
		if err != nil {
			return nil, err
		}
	} else {
		prompt := fmt.Sprintf("Select model for %s:", runner)
		if _, ok := runner.(AliasConfigurer); ok {
			prompt = fmt.Sprintf("Select Primary model for %s:", runner)
		}
		model, err := single(prompt, items, current)
		if err != nil {
			return nil, err
		}
		selected = []string{model}
	}

	var toPull []string
	for _, m := range selected {
		if !existingModels[m] && !IsCloudModelName(m) {
			toPull = append(toPull, m)
		}
	}
	if len(toPull) > 0 {
		msg := fmt.Sprintf("Download %s?", strings.Join(toPull, ", "))
		if ok, err := ConfirmPrompt(msg); err != nil {
			return nil, err
		} else if !ok {
			return nil, errCancelled
		}
		for _, m := range toPull {
			fmt.Fprintf(os.Stderr, "\n")
			if err := pullModel(ctx, client, m, false); err != nil {
				return nil, fmt.Errorf("failed to pull %s: %w", m, err)
			}
		}
	}

	if err := EnsureAuth(ctx, client, cloudModels, selected); err != nil {
		return nil, err
	}

	return selected, nil
}

func selectModels(ctx context.Context, name, current string) ([]string, error) {
	return selectModelsWithSelectors(ctx, name, current, DefaultSingleSelector, DefaultMultiSelector)
}

func pullIfNeeded(ctx context.Context, client *api.Client, existingModels map[string]bool, model string) error {
	if IsCloudModelName(model) || existingModels[model] {
		return nil
	}
	return confirmAndPull(ctx, client, model)
}

// ShowOrPull checks if a model exists via client.Show and offers to pull it if not found.
func ShowOrPull(ctx context.Context, client *api.Client, model string) error {
	return ShowOrPullWithPolicy(ctx, client, model, MissingModelPromptPull)
}

// ShowOrPullWithPolicy checks if a model exists and applies the provided missing-model policy.
func ShowOrPullWithPolicy(ctx context.Context, client *api.Client, model string, policy MissingModelPolicy) error {
	if _, err := client.Show(ctx, &api.ShowRequest{Model: model}); err == nil {
		return nil
	} else {
		var statusErr api.StatusError
		if !errors.As(err, &statusErr) || statusErr.StatusCode != http.StatusNotFound {
			return err
		}
	}

	if IsCloudModelName(model) {
		return nil
	}

	switch policy {
	case MissingModelFail:
		return fmt.Errorf("model %q not found; run 'ollama pull %s' first", model, model)
	default:
		return confirmAndPull(ctx, client, model)
	}
}

func confirmAndPull(ctx context.Context, client *api.Client, model string) error {
	if ok, err := ConfirmPrompt(fmt.Sprintf("Download %s?", model)); err != nil {
		return err
	} else if !ok {
		return errCancelled
	}
	fmt.Fprintf(os.Stderr, "\n")
	if err := pullModel(ctx, client, model, false); err != nil {
		return fmt.Errorf("failed to pull %s: %w", model, err)
	}
	return nil
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
		existing = append(existing, modelInfo{Name: m.Name, Remote: m.RemoteModel != ""})
	}

	cloudDisabled, _ := CloudStatusDisabled(ctx, client)
	if cloudDisabled {
		existing = FilterCloudModels(existing)
	}

	items, _, existingModels, cloudModels := BuildModelList(existing, nil, "")
	if cloudDisabled {
		items = FilterCloudItems(items)
	}
	if len(items) == 0 {
		return nil, nil, nil, nil, fmt.Errorf("no models available, run 'ollama pull <model>' first")
	}

	return items, existingModels, cloudModels, client, nil
}

func resolveEditorModels(name string, models []string, picker func() ([]string, error)) ([]string, error) {
	filtered := filterDisabledCloudModels(models)
	if len(filtered) != len(models) {
		if err := config.SaveIntegration(name, filtered); err != nil {
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
	if err := config.SaveIntegration(name, selected); err != nil {
		return nil, fmt.Errorf("failed to save: %w", err)
	}
	return selected, nil
}

// PrepareEditorIntegration persists models and applies editor-managed config files.
func PrepareEditorIntegration(name string, runner Runner, editor Editor, models []string) error {
	if ok, err := confirmEditorEdit(runner, editor); err != nil {
		return err
	} else if !ok {
		return errCancelled
	}
	if err := editor.Edit(models); err != nil {
		return fmt.Errorf("setup failed: %w", err)
	}
	if err := config.SaveIntegration(name, models); err != nil {
		return fmt.Errorf("failed to save: %w", err)
	}
	return nil
}

// RunIntegration executes a configured integration with the selected model.
func RunIntegration(name, modelName string, args []string) error {
	_, runner, err := LookupIntegration(name)
	if err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "\nLaunching %s with %s...\n", runner, modelName)
	return runner.Run(modelName, args)
}

func confirmEditorEdit(runner Runner, editor Editor) (bool, error) {
	paths := editor.Paths()
	if len(paths) == 0 {
		return true, nil
	}

	fmt.Fprintf(os.Stderr, "This will modify your %s configuration:\n", runner)
	for _, path := range paths {
		fmt.Fprintf(os.Stderr, "  %s\n", path)
	}
	fmt.Fprintf(os.Stderr, "Backups will be saved to %s/\n\n", backupDir())

	return ConfirmPrompt("Proceed?")
}

// BuildModelList merges existing models with recommendations for selection UIs.
func BuildModelList(existing []modelInfo, preChecked []string, current string) (items []ModelItem, orderedChecked []string, existingModels, cloudModels map[string]bool) {
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
		if IsCloudModelName(rec.Name) {
			cloudModels[rec.Name] = true
		}
	}

	checked := make(map[string]bool, len(preChecked))
	for _, n := range preChecked {
		checked[n] = true
	}

	for _, item := range items {
		if item.Name == current || strings.HasPrefix(item.Name, current+":") {
			current = item.Name
			break
		}
	}
	if checked[current] {
		preChecked = append([]string{current}, slices.DeleteFunc(preChecked, func(m string) bool { return m == current })...)
	}

	notInstalled := make(map[string]bool)
	for i := range items {
		if !existingModels[items[i].Name] && !cloudModels[items[i].Name] {
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

	recRank := make(map[string]int)
	for i, rec := range recommendedModels {
		recRank[rec.Name] = i + 1
	}

	onlyLocal := hasLocalModel && !hasCloudModel

	if hasLocalModel || hasCloudModel {
		slices.SortStableFunc(items, func(a, b ModelItem) int {
			ac, bc := checked[a.Name], checked[b.Name]
			aNew, bNew := notInstalled[a.Name], notInstalled[b.Name]
			aRec, bRec := recRank[a.Name] > 0, recRank[b.Name] > 0
			aCloud, bCloud := cloudModels[a.Name], cloudModels[b.Name]

			if ac != bc {
				if ac {
					return -1
				}
				return 1
			}
			if aRec != bRec {
				if aRec {
					return -1
				}
				return 1
			}
			if aRec && bRec {
				if aCloud != bCloud {
					if onlyLocal {
						if aCloud {
							return 1
						}
						return -1
					}
					if aCloud {
						return -1
					}
					return 1
				}
				return recRank[a.Name] - recRank[b.Name]
			}
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

// IsCloudModelDisabled reports whether the given model name looks like a cloud model and cloud features are disabled.
func IsCloudModelDisabled(ctx context.Context, name string) bool {
	if !IsCloudModelName(name) {
		return false
	}
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return false
	}
	disabled, _ := CloudStatusDisabled(ctx, client)
	return disabled
}

// IsCloudModelName reports whether the model name has an explicit cloud source.
func IsCloudModelName(name string) bool {
	return modelref.HasExplicitCloudSource(name)
}

// FilterCloudModels drops remote-only models from the given inventory.
func FilterCloudModels(existing []modelInfo) []modelInfo {
	filtered := existing[:0]
	for _, m := range existing {
		if !m.Remote {
			filtered = append(filtered, m)
		}
	}
	return filtered
}

func filterDisabledCloudModels(models []string) []string {
	var filtered []string
	for _, m := range models {
		if !IsCloudModelDisabled(context.Background(), m) {
			filtered = append(filtered, m)
		}
	}
	return filtered
}

// FilterCloudItems removes cloud models from selection items.
func FilterCloudItems(items []ModelItem) []ModelItem {
	filtered := items[:0]
	for _, item := range items {
		if !IsCloudModelName(item.Name) {
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

	cloudDisabled, _ := CloudStatusDisabled(ctx, client)
	if cloudDisabled {
		existing = FilterCloudModels(existing)
	}

	lastModel := config.LastModel()
	var preChecked []string
	if lastModel != "" {
		preChecked = []string{lastModel}
	}

	items, _, existingModels, _ := BuildModelList(existing, preChecked, lastModel)
	if cloudDisabled {
		items = FilterCloudItems(items)
	}

	return items, existingModels
}

// CloudStatusDisabled returns whether cloud usage is currently disabled.
func CloudStatusDisabled(ctx context.Context, client *api.Client) (disabled bool, known bool) {
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

// TODO(parthsareen): this duplicates the pull progress UI in cmd.PullHandler.
// Move the shared pull rendering to a small utility once the package boundary settles.
func pullModel(ctx context.Context, client *api.Client, model string, insecure bool) error {
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

	request := api.PullRequest{Name: model, Insecure: insecure}
	return client.Pull(ctx, &request, fn)
}
