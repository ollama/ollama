package launch

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/ollama/ollama/cmd/config"
	"github.com/ollama/ollama/envconfig"
	"gopkg.in/yaml.v3"
)

const (
	tealkitIntegrationName = "tealkit"
	tealkitProviderName    = "ollama"
)

// TealKit implements Runner and ManagedSingleModel for the TealKit integration.
type TealKit struct{}

func (t *TealKit) String() string { return "TealKit" }

func (t *TealKit) installed() bool {
	_, err := t.findPath()
	return err == nil
}

func (t *TealKit) ensureInstalled() error {
	if t.installed() {
		return nil
	}
	return fmt.Errorf("TealKit is not installed.\n\nDownload the CLI or Desktop application from:\n  https://github.com/lschaffer/tealkit/releases\n\nOr build from source:\n  dart pub global activate --source path ./cli\n\nThen re-run:\n  ollama launch tealkit")
}

func (t *TealKit) findPath() (string, error) {
	if p, err := exec.LookPath("tealkit"); err == nil {
		return p, nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	names := []string{"tealkit"}
	if runtime.GOOS == "windows" {
		names = []string{"tealkit.exe", "tealkit.bat", "tealkit.cmd"}
	}

	searchDirs := []string{
		filepath.Join(home, ".tealkit", "bin"),
		filepath.Join(home, ".pub-cache", "bin"),
		filepath.Join(home, "AppData", "Local", "Pub", "Cache", "bin"),
		filepath.Join(home, ".local", "bin"),
	}

	for _, dir := range searchDirs {
		for _, name := range names {
			candidate := filepath.Join(dir, name)
			if _, err := os.Stat(candidate); err == nil {
				return candidate, nil
			}
		}
	}
	return "", exec.ErrNotFound
}

func (t *TealKit) Paths() []string {
	var paths []string
	if cfgPath, err := tealkitConfigPath(); err == nil {
		if _, err := os.Stat(cfgPath); err == nil {
			paths = append(paths, cfgPath)
		}
	}
	if _, err := os.Stat("llm.yaml"); err == nil {
		paths = append(paths, "llm.yaml")
	}
	return paths
}

func tealkitConfigPath() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".tealkit", "llm.yaml"), nil
}

func (t *TealKit) Configure(model string) error {
	if model == "" {
		return nil
	}

	cfgPath, err := tealkitConfigPath()
	if err != nil {
		cfgPath = "llm.yaml"
	}

	_ = os.MkdirAll(filepath.Dir(cfgPath), 0o755)

	configMap := make(map[string]any)
	if data, err := os.ReadFile(cfgPath); err == nil {
		_ = yaml.Unmarshal(data, &configMap)
	}

	baseURL := "http://127.0.0.1:11434"
	if u := envconfig.ConnectableHost(); u != nil && u.String() != "" {
		baseURL = u.String()
	}

	configMap["provider"] = tealkitProviderName
	configMap["model"] = model
	configMap["base_url"] = baseURL
	if _, ok := configMap["temperature"]; !ok {
		configMap["temperature"] = 0.2
	}
	if _, ok := configMap["max_tokens"]; !ok {
		configMap["max_tokens"] = 4096
	}

	out, err := yaml.Marshal(configMap)
	if err != nil {
		return err
	}

	return os.WriteFile(cfgPath, out, 0o644)
}

func (t *TealKit) CurrentModel() string {
	cfgPath, err := tealkitConfigPath()
	if err != nil {
		cfgPath = "llm.yaml"
	}
	data, err := os.ReadFile(cfgPath)
	if err != nil {
		data, err = os.ReadFile("llm.yaml")
		if err != nil {
			return ""
		}
	}

	var configMap map[string]any
	if err := yaml.Unmarshal(data, &configMap); err != nil {
		return ""
	}

	provider, _ := configMap["provider"].(string)
	if strings.ToLower(provider) != tealkitProviderName && strings.ToLower(provider) != "openai_compatible" {
		return ""
	}

	model, _ := configMap["model"].(string)
	return model
}

func (t *TealKit) Onboard() error {
	return config.MarkIntegrationOnboarded(tealkitIntegrationName)
}

func (t *TealKit) Run(model string, _ []LaunchModel, args []string) error {
	bin, err := t.findPath()
	if err != nil {
		return t.ensureInstalled()
	}

	runArgs := []string{"chat"}
	if model != "" {
		runArgs = append(runArgs, "--model", model)
	}
	runArgs = append(runArgs, args...)

	cmd := exec.Command(bin, runArgs...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Env = os.Environ()
	return cmd.Run()
}
