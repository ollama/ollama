package launch

import (
	"context"
	"errors"
	"fmt"
	"math"
	"net/http"
	"os"
	"os/exec"
	"runtime"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/cmd/config"
	"github.com/ollama/ollama/format"
	internalcloud "github.com/ollama/ollama/internal/cloud"
	"github.com/ollama/ollama/internal/modelref"
	"github.com/ollama/ollama/progress"
)

var recommendedModels = []ModelItem{
	{Name: "kimi-k2.6:cloud", Description: "State-of-the-art coding, long-horizon execution, and multimodal agent swarm capability", Recommended: true, ContextLength: 262_144, MaxOutputTokens: 262_144},
	{Name: "qwen3.5:cloud", Description: "Reasoning, coding, and agentic tool use with vision", Recommended: true, ContextLength: 262_144, MaxOutputTokens: 32_768},
	{Name: "glm-5.1:cloud", Description: "Reasoning and code generation", Recommended: true, ContextLength: 202_752, MaxOutputTokens: 131_072},
	{Name: "minimax-m2.7:cloud", Description: "Fast, efficient coding and real-world productivity", Recommended: true, ContextLength: 204_800, MaxOutputTokens: 128_000},
	{Name: "gemma4", Description: "Reasoning and code generation locally", Recommended: true, VRAMBytes: 12 * format.GigaByte},
	{Name: "qwen3.5", Description: "Reasoning, coding, and visual understanding locally", Recommended: true, VRAMBytes: 14 * format.GigaByte},
}

func displayVRAM(vramBytes int64) string {
	if vramBytes <= 0 {
		return ""
	}
	gb := float64(vramBytes) / format.GigaByte
	if gb == math.Trunc(gb) {
		return fmt.Sprintf("~%.0fGB", gb)
	}
	return fmt.Sprintf("~%.1fGB", gb)
}

// cloudModelLimit holds context and output token limits for a cloud model.
type cloudModelLimit struct {
	Context int
	Output  int
}

// extraCloudModelLimits maps cloud model base names to token limits for models
// that are not already covered by recommendedModels fallback entries.
// TODO(parthsareen): grab context/output limits from model info instead of hardcoding
var extraCloudModelLimits = map[string]cloudModelLimit{
	"cogito-2.1:671b":     {Context: 163_840, Output: 65_536},
	"deepseek-v3.1:671b":  {Context: 163_840, Output: 163_840},
	"deepseek-v3.2":       {Context: 163_840, Output: 65_536},
	"gemma4:31b":          {Context: 262_144, Output: 131_072},
	"glm-4.6":             {Context: 202_752, Output: 131_072},
	"glm-4.7":             {Context: 202_752, Output: 131_072},
	"glm-5":               {Context: 202_752, Output: 131_072},
	"glm-5.1":             {Context: 202_752, Output: 131_072},
	"gpt-oss:120b":        {Context: 131_072, Output: 131_072},
	"gpt-oss:20b":         {Context: 131_072, Output: 131_072},
	"kimi-k2:1t":          {Context: 262_144, Output: 262_144},
	"kimi-k2.5":           {Context: 262_144, Output: 262_144},
	"kimi-k2.6":           {Context: 262_144, Output: 262_144},
	"kimi-k2-thinking":    {Context: 262_144, Output: 262_144},
	"nemotron-3-nano:30b": {Context: 1_048_576, Output: 131_072},
	"qwen3-coder:480b":    {Context: 262_144, Output: 65_536},
	"qwen3-coder-next":    {Context: 262_144, Output: 32_768},
	"qwen3-next:80b":      {Context: 262_144, Output: 32_768},
	"qwen3.5":             {Context: 262_144, Output: 32_768},
}

var cloudModelLimits = mergeCloudModelLimits(cloudModelLimitsFromRecommendations(recommendedModels), extraCloudModelLimits)

var (
	dynamicCloudModelLimitsMu sync.RWMutex
	dynamicCloudModelLimits   = map[string]cloudModelLimit{}
)

// lookupCloudModelLimit returns the token limits for a cloud model.
// It normalizes explicit cloud source suffixes before checking the shared limit map.
func lookupCloudModelLimit(name string) (cloudModelLimit, bool) {
	base, stripped := modelref.StripCloudSourceTag(name)
	if stripped {
		dynamicCloudModelLimitsMu.RLock()
		l, ok := dynamicCloudModelLimits[base]
		dynamicCloudModelLimitsMu.RUnlock()
		if ok {
			return l, true
		}
		if l, ok := cloudModelLimits[base]; ok {
			return l, true
		}
	}
	return cloudModelLimit{}, false
}

func setDynamicCloudModelLimits(limits map[string]cloudModelLimit) {
	dynamicCloudModelLimitsMu.Lock()
	defer dynamicCloudModelLimitsMu.Unlock()
	if limits == nil {
		dynamicCloudModelLimits = map[string]cloudModelLimit{}
		return
	}
	cp := make(map[string]cloudModelLimit, len(limits))
	for k, v := range limits {
		cp[k] = v
	}
	dynamicCloudModelLimits = cp
}

func cloudModelLimitsFromRecommendations(recommendations []ModelItem) map[string]cloudModelLimit {
	limits := make(map[string]cloudModelLimit, len(recommendations))
	for _, rec := range recommendations {
		if !isCloudModelName(rec.Name) || rec.ContextLength <= 0 || rec.MaxOutputTokens <= 0 {
			continue
		}
		base, stripped := modelref.StripCloudSourceTag(rec.Name)
		if !stripped || base == "" {
			continue
		}
		limits[base] = cloudModelLimit{
			Context: rec.ContextLength,
			Output:  rec.MaxOutputTokens,
		}
	}
	return limits
}

func mergeCloudModelLimits(base map[string]cloudModelLimit, overlay map[string]cloudModelLimit) map[string]cloudModelLimit {
	out := make(map[string]cloudModelLimit, len(base)+len(overlay))
	for name, limit := range base {
		out[name] = limit
	}
	for name, limit := range overlay {
		out[name] = limit
	}
	return out
}

// missingModelPolicy controls how model-not-found errors should be handled.
type missingModelPolicy int

const (
	// missingModelPromptPull prompts the user to download missing local models.
	missingModelPromptPull missingModelPolicy = iota
	// missingModelAutoPull downloads missing local models without prompting.
	missingModelAutoPull
	// missingModelFail returns an error for missing local models without prompting.
	missingModelFail
)

// OpenBrowser opens the URL in the user's browser.
func OpenBrowser(url string) {
	switch runtime.GOOS {
	case "darwin":
		_ = exec.Command("open", url).Start()
	case "linux":
		// Skip on headless systems where no display server is available
		if os.Getenv("DISPLAY") == "" && os.Getenv("WAYLAND_DISPLAY") == "" {
			return
		}
		_ = exec.Command("xdg-open", url).Start()
	case "windows":
		_ = exec.Command("rundll32", "url.dll,FileProtocolHandler", url).Start()
	}
}

// ensureAuth ensures the user is signed in before cloud-backed models run.
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
	return ensureCloudAuth(ctx, client, strings.Join(selectedCloudModels, ", "))
}

func ensureCloudAuth(ctx context.Context, client *api.Client, modelList string) error {
	if disabled, known := cloudStatusDisabled(ctx, client); known && disabled {
		return errors.New(internalcloud.DisabledError("remote inference is unavailable"))
	}

	user, err := client.Whoami(ctx)
	if err == nil && user != nil && user.Name != "" {
		return nil
	}

	var aErr api.AuthorizationError
	if !errors.As(err, &aErr) || aErr.SigninURL == "" {
		if err != nil {
			return err
		}
		return fmt.Errorf("%s requires sign in", modelList)
	}

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

// showOrPullWithPolicy checks if a model exists and applies the provided missing-model policy.
func showOrPullWithPolicy(ctx context.Context, client *api.Client, model string, policy missingModelPolicy, isCloudModel bool) error {
	if _, err := client.Show(ctx, &api.ShowRequest{Model: model}); err == nil {
		return nil
	} else {
		var statusErr api.StatusError
		if !errors.As(err, &statusErr) || statusErr.StatusCode != http.StatusNotFound {
			return err
		}
	}

	if isCloudModel {
		if disabled, known := cloudStatusDisabled(ctx, client); known && disabled {
			return errors.New(internalcloud.DisabledError("remote inference is unavailable"))
		}
		return fmt.Errorf("model %q not found", model)
	}

	switch policy {
	case missingModelAutoPull:
		return pullMissingModel(ctx, client, model)
	case missingModelFail:
		return fmt.Errorf("model %q not found; run 'ollama pull %s' first, or use --yes to auto-pull", model, model)
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
	return pullMissingModel(ctx, client, model)
}

func pullMissingModel(ctx context.Context, client *api.Client, model string) error {
	if err := pullModel(ctx, client, model, false); err != nil {
		return fmt.Errorf("failed to pull %s: %w", model, err)
	}
	return nil
}

// prepareEditorIntegration persists models and applies editor-managed config files.
func prepareEditorIntegration(name string, editor Editor, models []string) error {
	if err := editor.Edit(models); err != nil {
		return fmt.Errorf("setup failed: %w", err)
	}
	if err := config.SaveIntegration(name, models); err != nil {
		return fmt.Errorf("failed to save: %w", err)
	}
	return nil
}

func prepareManagedSingleIntegration(name string, managed ManagedSingleModel, model string, models []string) error {
	models = dedupeModelList(append([]string{model}, models...))
	var err error
	if withModels, ok := managed.(ManagedModelListConfigurer); ok {
		err = withModels.ConfigureWithModels(model, models)
	} else {
		err = managed.Configure(model)
	}
	if err != nil {
		return fmt.Errorf("setup failed: %w", err)
	}
	if err := config.SaveIntegration(name, []string{model}); err != nil {
		return fmt.Errorf("failed to save: %w", err)
	}
	return nil
}

func prepareManagedAutodiscoveryIntegration(name string, autodiscovery ManagedAutodiscoveryIntegration, model string) error {
	if err := autodiscovery.ConfigureAutodiscovery(); err != nil {
		return fmt.Errorf("setup failed: %w", err)
	}
	if err := config.SaveIntegration(name, []string{model}); err != nil {
		return fmt.Errorf("failed to save: %w", err)
	}
	return nil
}

// buildModelList merges existing models with recommendations for selection UIs.
func buildModelList(existing []modelInfo, preChecked []string, current string) (items []ModelItem, orderedChecked []string, existingModels, cloudModels map[string]bool) {
	return buildModelListWithRecommendations(existing, recommendedModels, preChecked, current)
}

func buildModelListWithRecommendations(existing []modelInfo, recommendations []ModelItem, preChecked []string, current string) (items []ModelItem, orderedChecked []string, existingModels, cloudModels map[string]bool) {
	existingModels = make(map[string]bool)
	cloudModels = make(map[string]bool)
	recommended := make(map[string]bool)
	var hasLocalModel, hasCloudModel bool

	recDesc := make(map[string]string)
	for _, rec := range recommendations {
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

	for _, rec := range recommendations {
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

	if current != "" {
		matchedCurrent := false
		for _, item := range items {
			if item.Name == current {
				current = item.Name
				matchedCurrent = true
				break
			}
		}
		if !matchedCurrent {
			for _, item := range items {
				if strings.HasPrefix(item.Name, current+":") {
					current = item.Name
					break
				}
			}
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
			if vram := displayVRAM(items[i].VRAMBytes); vram != "" {
				parts = append(parts, vram)
			}
			parts = append(parts, "(not downloaded)")
			items[i].Description = strings.Join(parts, ", ")
		}
	}

	recRank := make(map[string]int)
	for i, rec := range recommendations {
		recRank[rec.Name] = i + 1
	}

	if hasLocalModel || hasCloudModel {
		// Keep the Recommended section pinned to recommendation order. Checked
		// and default-model priority only apply within the More section.
		slices.SortStableFunc(items, func(a, b ModelItem) int {
			ac, bc := checked[a.Name], checked[b.Name]
			aNew, bNew := notInstalled[a.Name], notInstalled[b.Name]
			aRec, bRec := recRank[a.Name] > 0, recRank[b.Name] > 0
			if aRec != bRec {
				if aRec {
					return -1
				}
				return 1
			}
			if aRec && bRec {
				return recRank[a.Name] - recRank[b.Name]
			}
			if ac != bc {
				if ac {
					return -1
				}
				return 1
			}
			// Among checked non-recommended items - put the default first
			if ac && !aRec && current != "" {
				aCurrent := a.Name == current
				bCurrent := b.Name == current
				if aCurrent != bCurrent {
					if aCurrent {
						return -1
					}
					return 1
				}
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

// isCloudModelName reports whether the model name has an explicit cloud source.
func isCloudModelName(name string) bool {
	return modelref.HasExplicitCloudSource(name)
}

// filterCloudModels drops remote-only models from the given inventory.
func filterCloudModels(existing []modelInfo) []modelInfo {
	filtered := existing[:0]
	for _, m := range existing {
		if !m.Remote {
			filtered = append(filtered, m)
		}
	}
	return filtered
}

// filterCloudItems removes cloud models from selection items.
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

// cloudStatusDisabled returns whether cloud usage is currently disabled.
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
