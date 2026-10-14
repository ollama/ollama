package launch

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/cmd/internal/fileutil"
)

var (
	codexAppOpenProfile  = defaultCodexAppOpenDirectProfile
	codexAppDirectHealth = func() error {
		client, err := api.ClientFromEnvironment()
		if err != nil {
			return err
		}
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		return client.Heartbeat(ctx)
	}
)

// CodexDesktop manages the macOS app's separate Ollama profile.
type CodexDesktop struct{ CodexApp }

func (c *CodexDesktop) Supported() error {
	if codexAppGOOS != "darwin" {
		return errors.New("ChatGPT desktop connection is only supported on macOS")
	}
	return nil
}

func (c *CodexDesktop) OllamaConfigured() bool {
	return codexAppDirectConfigured() || c.CodexApp.OllamaConfigured()
}

func (c *CodexDesktop) Running() bool {
	return codexAppProfileIsRunning() || (codexAppNeedsLegacyRestore() && codexAppIsRunning())
}

func (c *CodexDesktop) UseOllamaFromDesktop(primary string, models []LaunchModel, restartConfirmed bool) error {
	return c.updateModels(primary, models, true, restartConfirmed)
}

func (c *CodexDesktop) UpdateOllamaModelsFromDesktop(primary string, models []LaunchModel, restartConfirmed bool) error {
	return c.updateModels(primary, models, false, restartConfirmed)
}

func (c *CodexDesktop) updateModels(primary string, models []LaunchModel, openWhenStopped, restartConfirmed bool) error {
	if err := c.Supported(); err != nil {
		return err
	}
	if !c.Installed() {
		return errors.New("ChatGPT is not installed")
	}
	if err := codexAppDirectHealth(); err != nil {
		return err
	}
	return applyCodexAppDirectProfile(func() error {
		return configureCodexAppDirect(primary, models)
	}, openWhenStopped, restartConfirmed)
}

func (c *CodexDesktop) RestoreFromDesktop(restartConfirmed bool) error {
	if err := c.Supported(); err != nil {
		return err
	}
	return applyCodexAppDirectProfile(disconnectCodexAppDirect, false, restartConfirmed)
}

func (c *CodexDesktop) RestartFromDesktop(restartConfirmed bool) error {
	if err := c.Supported(); err != nil {
		return err
	}
	if !c.Installed() {
		return errors.New("ChatGPT is not installed")
	}
	if !c.OllamaConfigured() {
		return errors.New("ChatGPT is not configured to use Ollama")
	}
	return applyCodexAppDirectProfile(repairCodexAppDirect, true, restartConfirmed)
}

func (c *CodexDesktop) OllamaRequestCount() uint64 {
	if codexAppDirectConfigured() {
		return codexAppDirectRequestCount()
	}
	return c.CodexApp.OllamaRequestCount()
}

func codexAppDirectConfigPath() (string, error) {
	root, err := codexAppOllamaProfileRoot()
	if err != nil {
		return "", err
	}
	return filepath.Join(root, "codex-home", "config.toml"), nil
}

func codexAppDirectConfigured() bool {
	path, err := codexAppDirectConfigPath()
	if err != nil {
		return false
	}
	_, err = os.Stat(path)
	return err == nil
}

func codexAppDirectCurrentModel() string {
	path, err := codexAppDirectConfigPath()
	if err != nil {
		return ""
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	config, err := codexParseConfig(string(data))
	if err != nil || config.RootString(codexRootModelProviderKey) != codexAppProfileName ||
		codexNormalizeURL(config.ProviderString(codexAppProfileName, "base_url")) != codexNormalizeURL(codexBaseURL()) {
		return ""
	}
	catalog, err := os.ReadFile(config.RootString(codexRootModelCatalogJSONKey))
	if err != nil {
		return ""
	}
	parsed, err := parseCodexAppModelCatalog(catalog)
	if err != nil {
		return ""
	}
	model := config.RootString(codexRootModelKey)
	for _, entry := range parsed.Models {
		slug, _ := codexAppRawCatalogSlug(entry)
		if codexAppCatalogModelKey(slug) == codexAppCatalogModelKey(model) {
			return model
		}
	}
	return ""
}

func configureCodexAppDirect(primary string, models []LaunchModel) error {
	primary = strings.TrimSpace(primary)
	if primary == "" {
		return errors.New("chatgpt requires a model")
	}
	models = codexAppCatalogModels(primary, models)
	entries := make([]json.RawMessage, 0, len(models))
	baseInstructions := codexAppBaseInstructions()
	for i, model := range models {
		entry, err := json.Marshal(codexAppCatalogEntry(model.Name, codexAppModelMetadataFromLaunchModel(model), i, baseInstructions))
		if err != nil {
			return err
		}
		entries = append(entries, entry)
	}
	return writeCodexAppDirectProfile(primary, codexAppRawModelCatalog{Models: entries})
}

func writeCodexAppDirectProfile(primary string, catalog codexAppRawModelCatalog) error {
	path, err := codexAppDirectConfigPath()
	if err != nil {
		return err
	}
	data, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		data, err = os.ReadFile(path + ".disconnected")
	}
	if err != nil && !os.IsNotExist(err) {
		return err
	}
	text := string(data)
	if err := codexValidateConfigText(text); err != nil {
		return err
	}
	catalogPath := codexAppModelCatalogPathForConfig(path)
	text = codexRemoveRootValue(text, codexRootProfileKey)
	text = codexRemoveRootValue(text, codexRootOpenAIBaseURLKey)
	text = codexSetRootStringValue(text, codexRootModelKey, primary)
	text = codexSetRootStringValue(text, codexRootModelProviderKey, codexAppProfileName)
	text = codexSetRootStringValue(text, codexRootModelCatalogJSONKey, catalogPath)
	text = codexSetRootStringValue(text, "service_tier", "default")
	text = codexAppRemoveOwnedSections(text)
	text += fmt.Sprintf("\n%s\nname = \"Ollama\"\nbase_url = %q\nwire_api = \"responses\"\nrequires_openai_auth = false\n", codexProviderHeaderFor(codexAppProfileName), codexBaseURL())
	text = codexAppSetReasoningEfforts(text, codexAppReasoningEffortsForConfig(text))
	if err := codexValidateConfigText(text); err != nil {
		return err
	}
	catalogData, err := json.MarshalIndent(catalog, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	if err := fileutil.WriteWithBackup(catalogPath, append(catalogData, '\n'), codexAppIntegrationName); err != nil {
		return err
	}
	createdAuth, err := ensureCodexAppManagedAuth(path)
	if err != nil {
		return err
	}
	if err := fileutil.WriteWithBackup(path, []byte(text), codexAppIntegrationName); err != nil {
		if createdAuth {
			err = errors.Join(err, removeCodexAppManagedAuth(path))
		}
		return err
	}
	return nil
}

func codexAppNeedsLegacyRestore() bool {
	path, err := codexConfigPath()
	if err != nil {
		return false
	}
	data, _ := os.ReadFile(path)
	if codexAppRootReferencesOwnedConfig(string(data)) {
		return true
	}
	_, err = os.Stat(codexAppRestoreStatePath())
	return err == nil
}

func applyCodexAppDirectProfile(change func() error, openWhenStopped, restartConfirmed bool) (result error) {
	legacy := codexAppNeedsLegacyRestore()
	regularRunning := legacy && codexAppIsRunning()
	profileRunning := codexAppProfileIsRunning()
	if (regularRunning || profileRunning) && !restartConfirmed {
		return ErrCodexAppRestartConfirmationRequired
	}
	if profileRunning {
		if err := codexAppStopProfile(); err != nil {
			return err
		}
	}
	regularRunning = legacy && codexAppIsRunning()
	if regularRunning {
		if !restartConfirmed {
			return ErrCodexAppRestartConfirmationRequired
		}
		if err := codexAppStopForRestart(nil); err != nil {
			if profileRunning {
				return errors.Join(err, codexAppOpenProfile())
			}
			return err
		}
		defer func() {
			if regularRunning {
				result = errors.Join(result, codexAppOpenApp(nil))
			}
		}()
	}
	if err := change(); err != nil {
		if profileRunning {
			return errors.Join(err, codexAppOpenProfile())
		}
		return err
	}
	if legacy {
		if err := restoreCodexAppProfile(); err != nil {
			return err
		}
	}
	if regularRunning {
		regularRunning = false
		if err := codexAppOpenApp(nil); err != nil {
			return err
		}
	}
	if codexAppDirectConfigured() && (openWhenStopped || profileRunning) {
		path, err := codexAppDirectConfigPath()
		if err != nil {
			return err
		}
		if err := resetCodexAppRequestCountAt(filepath.Join(filepath.Dir(path), "session-start")); err != nil {
			return err
		}
		return codexAppOpenProfile()
	}
	return nil
}

func disconnectCodexAppDirect() error {
	path, err := codexAppDirectConfigPath()
	if err != nil {
		return err
	}
	// Keep settings and conversations for the next connection.
	if err := os.Rename(path, path+".disconnected"); err != nil && !os.IsNotExist(err) {
		return err
	}
	return nil
}

func repairCodexAppDirect() error {
	if codexAppDirectConfigured() {
		if codexAppDirectCurrentModel() == "" {
			return errors.New("ChatGPT's Ollama profile needs its model selection saved again")
		}
		path, err := codexAppDirectConfigPath()
		if err != nil {
			return err
		}
		_, err = ensureCodexAppManagedAuth(path)
		return err
	}
	models, err := codexAppRoutingModels()
	if err != nil {
		return err
	}
	path, err := codexAppModelCatalogPath()
	if err != nil {
		return err
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	catalog, err := parseCodexAppModelCatalog(data)
	if err != nil {
		return err
	}
	selected := codexAppRawModelCatalog{}
	for _, entry := range catalog.Models {
		slug, _ := codexAppRawCatalogSlug(entry)
		for _, name := range models {
			if codexAppCatalogModelKey(slug) == codexAppCatalogModelKey(name) {
				selected.Models = append(selected.Models, entry)
				break
			}
		}
	}
	if len(selected.Models) == 0 {
		return errors.New("ChatGPT's Ollama model catalog is empty")
	}
	return writeCodexAppDirectProfile((&CodexApp{}).CurrentModel(), selected)
}

func codexAppDirectOpenArgs(appPath, configPath, userDataDir string) []string {
	args := []string{"-n", "--env", "CODEX_HOME=" + filepath.Dir(configPath), "--env", "CODEX_ELECTRON_USER_DATA_PATH=" + userDataDir}
	if appPath == "" {
		args = append(args, "-b", codexAppBundleID)
	} else {
		args = append(args, appPath)
	}
	return append(args, "--args", "--user-data-dir="+userDataDir)
}

func defaultCodexAppOpenDirectProfile() error {
	path, err := codexAppDirectConfigPath()
	if err != nil {
		return err
	}
	userData, err := codexAppOllamaProfileUserDataDir()
	if err != nil {
		return err
	}
	if _, err := prepareCodexAppOllamaProfileUserData(userData); err != nil {
		return err
	}
	cmd := exec.Command("open", codexAppDirectOpenArgs(codexAppAppPath(), path, userData)...)
	cmd.Stdout, cmd.Stderr = os.Stdout, os.Stderr
	return cmd.Run()
}

func codexAppDirectRequestCount() uint64 {
	path, err := codexAppDirectConfigPath()
	if err != nil {
		return 0
	}
	data, err := os.ReadFile(filepath.Join(filepath.Dir(path), "session-start"))
	if err != nil {
		return 0
	}
	start, err := time.Parse(time.RFC3339Nano, strings.TrimSpace(string(data)))
	if err != nil {
		return 0
	}
	return codexAppRequests.scan(filepath.Join(filepath.Dir(path), "sessions"), start, nil)
}
