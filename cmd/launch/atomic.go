package launch

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/ollama/ollama/cmd/internal/fileutil"
	"github.com/ollama/ollama/envconfig"
)

const (
	atomicInstallScript     = "curl -fsSL https://atomicagent.io/install | sh"
	atomicWindowsInstallURL = "https://atomicagent.io/install.ps1"
	// Entry id the launcher owns inside Atomic Agent's provider list.
	atomicLaunchProviderID = "ollama"
)

// Atomic implements Runner and Editor for the Atomic Agent CLI
// integration (https://github.com/AtomicBot-ai/atomic-agent).
type Atomic struct{}

func (a *Atomic) String() string { return "Atomic Agent" }

func (a *Atomic) Run(_ string, _ []LaunchModel, args []string) error {
	bin, err := ensureAtomicInstalled()
	if err != nil {
		return err
	}
	cmd := exec.Command(bin, args...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

// findAtomicAgent looks for the binary on PATH first, then in the
// installers' default locations: install.sh puts a symlink-free copy in
// ~/.local/bin (which GUI-launched shells often do not have on PATH),
// and install.ps1 uses %LOCALAPPDATA%\atomic-agent on Windows.
// ATOMIC_AGENT_INSTALL_DIR is the installer-supported override.
func findAtomicAgent() (string, error) {
	if p, err := exec.LookPath("atomic-agent"); err == nil {
		return p, nil
	}

	var candidates []string
	if dir := strings.TrimSpace(os.Getenv("ATOMIC_AGENT_INSTALL_DIR")); dir != "" {
		candidates = append(candidates, filepath.Join(dir, atomicExecutableName()))
	}
	if home, err := os.UserHomeDir(); err == nil {
		candidates = append(candidates, filepath.Join(home, ".local", "bin", "atomic-agent"))
	}
	if runtime.GOOS == "windows" {
		if localAppData := os.Getenv("LOCALAPPDATA"); localAppData != "" {
			candidates = append(candidates, filepath.Join(localAppData, "atomic-agent", "atomic-agent.exe"))
		}
	}
	for _, candidate := range candidates {
		if info, err := os.Stat(candidate); err == nil && !info.IsDir() {
			return candidate, nil
		}
	}
	return "", fmt.Errorf("atomic-agent is not installed")
}

func atomicExecutableName() string {
	if runtime.GOOS == "windows" {
		return "atomic-agent.exe"
	}
	return "atomic-agent"
}

func atomicInstalled() bool {
	_, err := findAtomicAgent()
	return err == nil
}

func ensureAtomicInstalled() (string, error) {
	if bin, err := findAtomicAgent(); err == nil {
		return bin, nil
	}

	if runtime.GOOS == "windows" {
		return "", fmt.Errorf("Atomic Agent is not installed\n\nInstall it from PowerShell:\n  irm %s | iex\n\nThen re-run:\n  ollama launch atomic", atomicWindowsInstallURL)
	}

	var missing []string
	for _, dep := range []string{"sh", "curl"} {
		if _, err := exec.LookPath(dep); err != nil {
			missing = append(missing, dep)
		}
	}
	if len(missing) > 0 {
		return "", fmt.Errorf("Atomic Agent is not installed and required dependencies are missing\n\nInstall the following first:\n  %s\n\nThen re-run:\n  ollama launch atomic", strings.Join(missing, "\n  "))
	}

	ok, err := ConfirmPrompt("Atomic Agent is not installed. Install now?")
	if err != nil {
		return "", err
	}
	if !ok {
		return "", fmt.Errorf("atomic-agent installation cancelled")
	}

	fmt.Fprintf(os.Stderr, "\nInstalling Atomic Agent...\n")
	cmd := exec.Command("sh", "-c", atomicInstallScript)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("failed to install atomic-agent: %w", err)
	}

	bin, err := findAtomicAgent()
	if err != nil {
		return "", fmt.Errorf("atomic-agent was installed but the binary was not found\n\nYou may need to restart your shell")
	}
	fmt.Fprintf(os.Stderr, "%sAtomic Agent installed successfully%s\n\n", ansiGreen, ansiReset)
	return bin, nil
}

// atomicStateDir mirrors the agent's own resolution: the
// ATOMIC_AGENT_STATE_DIR override first, ~/.atomic-agent otherwise.
func atomicStateDir() (string, error) {
	if dir := strings.TrimSpace(os.Getenv("ATOMIC_AGENT_STATE_DIR")); dir != "" {
		return dir, nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".atomic-agent"), nil
}

func atomicConfigPath() (string, error) {
	dir, err := atomicStateDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, "config.json"), nil
}

// atomicOllamaBaseURL is the Ollama root the agent should talk to.
// Atomic Agent stores OpenAI-compatible base URLs without the `/v1`
// suffix (its call sites append the path themselves), so this must be
// the bare host root.
func atomicOllamaBaseURL() string {
	return strings.TrimRight(envconfig.ConnectableHost().String(), "/")
}

func (a *Atomic) Paths() []string {
	path, err := atomicConfigPath()
	if err != nil {
		return nil
	}
	if _, err := os.Stat(path); err != nil {
		return nil
	}
	return []string{path}
}

// Edit points Atomic Agent's active text provider at Ollama.
//
// The agent's config is a single JSON file. Everything the launcher
// does not own is preserved verbatim: only the `llm` block's Ollama
// entry and the active-provider pointer are touched. A missing
// `version` field is fine — the agent parses such files with its
// current schema and fills defaults (hand-written configs are a
// supported path).
func (a *Atomic) Edit(models []LaunchModel) error {
	if len(models) == 0 {
		return nil
	}

	path, err := atomicConfigPath()
	if err != nil {
		return err
	}
	config := make(map[string]any)
	if data, err := os.ReadFile(path); err == nil {
		if err := json.Unmarshal(data, &config); err != nil {
			return fmt.Errorf("failed to parse config: %w, at: %s", err, path)
		}
	} else if !os.IsNotExist(err) {
		return err
	}

	llm, _ := config["llm"].(map[string]any)
	if llm == nil {
		llm = make(map[string]any)
	}

	providers, _ := llm["providers"].([]any)

	// The agent requires the embedding provider to resolve; keep
	// whatever is configured and default fresh configs to the local
	// daemon, mirroring what the agent's own wizard persists.
	if _, ok := llm["activeEmbeddingProvider"].(string); !ok {
		llm["activeEmbeddingProvider"] = "local-llama"
		if !atomicHasProvider(providers, "local-llama") {
			providers = append(providers, map[string]any{
				"id":   "local-llama",
				"kind": "llama-server",
				"url":  "http://127.0.0.1:8080",
			})
		}
	}
	if _, ok := llm["toolTransport"].(string); !ok {
		llm["toolTransport"] = "auto"
	}

	providers = atomicUpsertOllamaProvider(providers, models[0].Name)
	llm["providers"] = providers
	llm["activeTextProvider"] = atomicLaunchProviderID
	config["llm"] = llm

	data, err := json.MarshalIndent(config, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return fileutil.WriteWithBackup(path, data, "atomic-agent")
}

func atomicHasProvider(providers []any, id string) bool {
	for _, raw := range providers {
		entry, _ := raw.(map[string]any)
		if entry != nil && entry["id"] == id {
			return true
		}
	}
	return false
}

// atomicUpsertOllamaProvider writes the launcher-owned entry. It uses
// the agent's `openai-compatible` kind, which every released version
// accepts; Ollama's compatible surface lives under `/v1` but the base
// URL is stored WITHOUT that suffix because the agent appends the path
// itself (a stored `/v1` would produce `/v1/v1/...` requests).
func atomicUpsertOllamaProvider(providers []any, model string) []any {
	entry := map[string]any{
		"id":               atomicLaunchProviderID,
		"kind":             "openai-compatible",
		"baseUrl":          atomicOllamaBaseURL(),
		"apiKeyEnvVar":     "OLLAMA_API_KEY",
		"defaultChatModel": model,
	}
	for i, raw := range providers {
		existing, _ := raw.(map[string]any)
		if existing != nil && existing["id"] == atomicLaunchProviderID {
			// Keep unknown keys the agent may have added to the entry;
			// only re-point the fields the launcher owns.
			existing["kind"] = entry["kind"]
			existing["baseUrl"] = entry["baseUrl"]
			existing["apiKeyEnvVar"] = entry["apiKeyEnvVar"]
			existing["defaultChatModel"] = model
			providers[i] = existing
			return providers
		}
	}
	return append(providers, entry)
}

func (a *Atomic) Models() []string {
	path, err := atomicConfigPath()
	if err != nil {
		return nil
	}
	config, err := fileutil.ReadJSON(path)
	if err != nil {
		return nil
	}
	llm, _ := config["llm"].(map[string]any)
	if llm == nil || llm["activeTextProvider"] != atomicLaunchProviderID {
		return nil
	}
	providers, _ := llm["providers"].([]any)
	for _, raw := range providers {
		entry, _ := raw.(map[string]any)
		if entry == nil || entry["id"] != atomicLaunchProviderID {
			continue
		}
		if model, _ := entry["defaultChatModel"].(string); model != "" {
			return []string{model}
		}
	}
	return nil
}
