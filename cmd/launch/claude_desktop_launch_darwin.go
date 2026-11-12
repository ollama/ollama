package launch

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"time"

	"github.com/ollama/ollama/cmd/config"
	"github.com/ollama/ollama/cmd/internal/fileutil"
	"github.com/ollama/ollama/internal/proxy"
	"golang.org/x/sys/unix"
)

func (c *launcherClient) launchClaudeDesktop(ctx context.Context, desktop *ClaudeDesktop, req IntegrationLaunchRequest) error {
	if err := claudeDesktopContextError(ctx); err != nil {
		return err
	}
	if len(req.ExtraArgs) > 0 {
		return errors.New("claude-desktop does not accept extra arguments")
	}
	if err := ensureClaudeDesktopGateway(); err != nil {
		return fmt.Errorf("%w; the Ollama app must be running", err)
	}
	saved, err := config.LoadIntegration(claudeDesktopIntegrationName)
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	candidate := make(map[string]string)
	if saved != nil {
		candidate = maps.Clone(saved.Aliases)
		if len(candidate) == 0 {
			candidate = proxy.ClaudeDesktopMappings(proxy.SelectClaudeDesktopModels(nil, saved.Models))
		}
	}
	catalog, err := fetchClaudeDesktopCatalog(ctx)
	if err != nil {
		return err
	}
	defaults := claudeDesktopCatalogMappings(catalog)
	standard := defaults["claude-sonnet-5"]
	if standard == "" && len(catalog) > 0 {
		standard = catalog[0].DisplayName
	}
	for _, route := range proxy.ClaudeDesktopRoutes() {
		if defaults[route.ID] == "" && standard != "" {
			defaults[route.ID] = standard
		}
	}
	if req.ForceConfigure && req.ModelOverride == "" {
		candidate, err = c.configureClaudeDesktopMappings(ctx, candidate, defaults)
		if err != nil {
			return err
		}
		req.ConfigureOnly = false // The editor explicitly offers Save and launch.
	} else if req.ModelOverride != "" {
		if candidate == nil {
			candidate = make(map[string]string)
		}
		candidate["claude-sonnet-5"] = req.ModelOverride
	} else if len(candidate) == 0 {
		candidate = defaults
	}
	models := dedupeModelList(claudeDesktopMappingModels(candidate))
	if len(models) == 0 {
		return errors.New("no Claude Desktop models are available; use --config to choose a model")
	}
	if err := c.ensureModelsReadyFor(ctx, models, "Claude Desktop", claudeDesktopIntegrationName); err != nil {
		return err
	}
	autoMode, err := ClaudeDesktopAutoModeEnabled()
	if err != nil {
		return err
	}
	cloud := false
	for _, model := range models {
		cloud = cloud || isCloudModelName(model)
	}
	autoMode = autoMode && cloud
	targets, err := claudeDesktopTargetPaths()
	if err != nil {
		return err
	}
	if saved != nil && maps.Equal(saved.Aliases, candidate) && maps.Equal(claudeDesktopCatalogMappings(catalog), candidate) && claudeDesktopTargetsConfigured(targets, autoMode) && claudeDesktopPinnedCatalogMatches(targets, catalog) {
		if req.ConfigureOnly {
			return nil
		}
		if err := claudeDesktopContextError(ctx); err != nil {
			return err
		}
		return desktop.Open()
	}
	apply := func(ctx context.Context) error {
		return applyClaudeDesktopMappings(ctx, candidate, func() error {
			// The app gateway resolves metadata and checks current Cloud, account,
			// plan, and local-model access before advertising these routes.
			catalog, err := fetchClaudeDesktopCatalog(ctx)
			if err != nil {
				return err
			}
			if !maps.Equal(claudeDesktopCatalogMappings(catalog), candidate) {
				return errors.New("selected Claude Desktop mappings are unavailable from the Ollama app; check model access or update the app and try again")
			}
			if err := claudeDesktopContextError(ctx); err != nil {
				return err
			}
			return configureClaudeDesktopTargets(targets, claudeDesktopGatewayBaseURL, "ollama", autoMode, catalog)
		})
	}
	if err := applyClaudeDesktopChangeContext(ctx, "Restart Claude Desktop to apply Ollama settings?", apply, !req.ConfigureOnly); err != nil {
		return err
	}
	fmt.Fprintln(os.Stderr, desktop.ConfigurationSuccessMessage())
	return nil
}

func claudeDesktopCatalogMappings(catalog []proxy.ClaudeDesktopCatalogModel) map[string]string {
	mappings := make(map[string]string, len(catalog))
	for _, model := range catalog {
		mappings[model.ID] = model.DisplayName
	}
	return mappings
}

// Change only the Claude selection. Raw JSON preserves unrelated settings and
// unknown fields; rollback re-reads the file so it also preserves later edits.
func applyClaudeDesktopMappings(ctx context.Context, mappings map[string]string, apply func() error) error {
	home, err := os.UserHomeDir()
	if err != nil {
		return err
	}
	path := filepath.Join(home, ".ollama", "config.json")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	lock, err := lockClaudeDesktopSettings(ctx, path+".lock")
	if err != nil {
		return err
	}
	defer lock.Close()
	read := func() (map[string]json.RawMessage, map[string]json.RawMessage, error) {
		document := make(map[string]json.RawMessage)
		integrations := make(map[string]json.RawMessage)
		data, err := os.ReadFile(path)
		if errors.Is(err, os.ErrNotExist) {
			return document, integrations, nil
		}
		if err == nil {
			err = json.Unmarshal(data, &document)
		}
		if err == nil && document != nil && document["integrations"] != nil {
			err = json.Unmarshal(document["integrations"], &integrations)
		}
		if err == nil && (document == nil || integrations == nil) {
			err = errors.New("Ollama settings must contain JSON objects")
		}
		return document, integrations, err
	}
	write := func(document, integrations map[string]json.RawMessage) error {
		data, err := json.Marshal(integrations)
		if err != nil {
			return err
		}
		document["integrations"] = data
		data, err = json.MarshalIndent(document, "", "  ")
		if err != nil {
			return err
		}
		return fileutil.WriteWithBackup(path, data)
	}
	document, integrations, err := read()
	if err != nil {
		return err
	}
	before := integrations[claudeDesktopIntegrationName]
	selection := make(map[string]json.RawMessage)
	if len(before) > 0 {
		if err := json.Unmarshal(before, &selection); err != nil {
			return err
		}
		if selection == nil {
			return errors.New("Claude Desktop settings must be a JSON object")
		}
	}
	selection["models"], _ = json.Marshal(claudeDesktopMappingModels(mappings))
	selection["aliases"], _ = json.Marshal(mappings)
	after, _ := json.Marshal(selection)
	integrations[claudeDesktopIntegrationName] = after
	if err := claudeDesktopContextError(ctx); err != nil {
		return err
	}
	if err := write(document, integrations); err != nil {
		return err
	}
	if applyErr := apply(); applyErr != nil {
		document, integrations, err = read()
		if err != nil {
			return errors.Join(applyErr, err)
		}
		var currentSelection map[string]json.RawMessage
		if err := json.Unmarshal(integrations[claudeDesktopIntegrationName], &currentSelection); err != nil {
			return errors.Join(applyErr, err)
		}
		current, _ := json.Marshal(currentSelection)
		if string(current) != string(after) {
			return errors.Join(applyErr, errors.New("Claude Desktop settings changed during setup; preserving the newer selection"))
		}
		if before == nil {
			delete(integrations, claudeDesktopIntegrationName)
		} else {
			integrations[claudeDesktopIntegrationName] = before
		}
		rollbackErr := write(document, integrations)
		// Refresh the gateway back to the restored selection; no inference runs.
		if rollbackErr == nil {
			_, rollbackErr = fetchClaudeDesktopCatalog(context.Background())
		}
		return errors.Join(applyErr, rollbackErr)
	}
	return nil
}

func lockClaudeDesktopSettings(ctx context.Context, path string) (*os.File, error) {
	file, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return nil, err
	}
	ticker := time.NewTicker(20 * time.Millisecond)
	defer ticker.Stop()
	for {
		if err := claudeDesktopContextError(ctx); err != nil {
			_ = file.Close()
			return nil, err
		}
		if err := unix.Flock(int(file.Fd()), unix.LOCK_EX|unix.LOCK_NB); err == nil {
			return file, nil // Closing the file releases the process lock.
		} else if !errors.Is(err, unix.EWOULDBLOCK) && !errors.Is(err, unix.EINTR) {
			_ = file.Close()
			return nil, err
		}
		select {
		case <-ctx.Done():
		case <-ticker.C:
		}
	}
}
