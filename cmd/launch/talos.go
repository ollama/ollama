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
	"github.com/ollama/ollama/internal/modelref"
)

const (
	// The Talos installer proves the download before it unpacks anything: it
	// fetches the published sha256 and an Ed25519 release signature and refuses
	// on mismatch, so piping it into a shell is the vendor-supported install
	// path, not a shortcut around verification.
	talosInstallScript = "curl -fsSL https://talos-agent.ch/install.sh | sh"
	// Provider slugs from Talos's own model catalog (talos/catalog.py). The
	// local entry defaults to http://localhost:11434/v1 and needs no key.
	talosProviderLocal = "ollama"
	talosProviderCloud = "ollama-cloud"
	// Writable SETTING keys in Talos's config schema (talos/schema.py).
	talosProviderKey = "TALOS_MODEL_PROVIDER"
	talosModelKey    = "TALOS_MODEL"
)

var (
	talosGOOS     = runtime.GOOS
	talosLookPath = exec.LookPath
	talosCommand  = exec.Command
	talosUserHome = os.UserHomeDir
)

// Talos is intentionally not an Editor integration: Talos guards its config
// behind a schema that decides per key what a command may write, so launch
// goes through `talos config set` — the same validated surface Talos offers
// scripts — instead of rewriting the env file itself.
type Talos struct{}

func (t *Talos) String() string { return "Talos" }

func (t *Talos) Run(_ string, _ []LaunchModel, args []string) error {
	argv, err := t.command()
	if err != nil {
		return err
	}
	return talosAttachedCommand(argv, append([]string{"chat"}, args...)...).Run()
}

// Supported reports platform support separately from installation: Talos's
// installer requires a POSIX shell and Python 3.11+, and ships no Windows path.
func (t *Talos) Supported() error {
	if talosGOOS == "windows" {
		return fmt.Errorf("talos currently supports macOS and Linux")
	}
	return nil
}

func (t *Talos) Paths() []string {
	configPath, err := talosConfigPath()
	if err != nil {
		return nil
	}
	return []string{configPath}
}

// Configure points Talos at Ollama through Talos's own CLI. `config set`
// validates against the schema and writes atomically with mode 600 — a file
// rewrite from the outside would bypass both.
func (t *Talos) Configure(model string) error {
	argv, err := t.command()
	if err != nil {
		return err
	}

	provider := talosProviderLocal
	configModel := model
	// A `:cloud` model cannot run against the local daemon from Talos's side
	// without Ollama in the middle, so it goes to Talos's ollama-cloud provider,
	// whose endpoint expects the plain upstream model name.
	if base, ok := modelref.StripCloudSourceTag(model); ok {
		provider = talosProviderCloud
		configModel = base
	}

	if err := talosConfigSet(argv, talosProviderKey, provider); err != nil {
		return err
	}
	if err := talosConfigSet(argv, talosModelKey, configModel); err != nil {
		return err
	}

	if provider == talosProviderCloud && !talosCloudKeyPresent() {
		// Talos refuses secrets on its command line by design (a key there lands
		// in shell history and process listings), so launch cannot write
		// OLLAMA_API_KEY — the user has to take Talos's secret-safe path.
		fmt.Fprintf(os.Stderr, "%sCloud models need an Ollama API key in Talos. Talos does not accept keys on the command line; run `talos setup model` once, or add OLLAMA_API_KEY to your Talos env file.%s\n", ansiGray, ansiReset)
	}

	if warn := talosNonDefaultHostWarning(); warn != "" && provider == talosProviderLocal {
		fmt.Fprintf(os.Stderr, "%s%s%s\n", ansiGray, warn, ansiReset)
	}
	return nil
}

func (t *Talos) CurrentModel() string {
	values := talosConfigValues()
	provider := strings.TrimSpace(values[talosProviderKey])
	current := strings.TrimSpace(values[talosModelKey])
	if current == "" {
		return ""
	}
	switch provider {
	case talosProviderLocal:
		return current
	case talosProviderCloud:
		// Report cloud models the way launch names them so the launcher sees a
		// match instead of reconfiguring on every run.
		return current + ":cloud"
	default:
		return ""
	}
}

func (t *Talos) Onboard() error {
	return config.MarkIntegrationOnboarded("talos")
}

func (t *Talos) RequiresInteractiveOnboarding() bool {
	return false
}

func (t *Talos) installed() bool {
	_, err := t.command()
	return err == nil
}

func (t *Talos) ensureInstalled() error {
	if t.installed() {
		return nil
	}

	var missing []string
	for _, dep := range []string{"curl", "sh"} {
		if _, err := talosLookPath(dep); err != nil {
			missing = append(missing, dep)
		}
	}
	if len(missing) > 0 {
		return fmt.Errorf("Talos is not installed and required dependencies are missing\n\nInstall the following first:\n  %s\n\nThen re-run:\n  ollama launch talos", strings.Join(missing, "\n  "))
	}

	ok, err := ConfirmPrompt("Talos is not installed. Install now?")
	if err != nil {
		return err
	}
	if !ok {
		return fmt.Errorf("talos installation cancelled")
	}

	// The installer runs Talos's full test suite and its adversarial suite in
	// front of the user before it finishes, so this takes a few minutes.
	fmt.Fprintf(os.Stderr, "\nInstalling Talos...\n")
	if err := talosAttachedCommand([]string{"sh"}, "-c", talosInstallScript).Run(); err != nil {
		return fmt.Errorf("failed to install talos: %w", err)
	}

	if !t.installed() {
		return fmt.Errorf("talos was installed but was not found\n\nYou may need to restart your shell")
	}

	fmt.Fprintf(os.Stderr, "%sTalos installed successfully%s\n\n", ansiGreen, ansiReset)
	return nil
}

// command resolves how to invoke Talos. The installer puts everything under
// ~/talos (or $TALOS_PREFIX) and deliberately adds nothing to PATH, so the
// venv interpreter plus `-m talos` is the canonical invocation; a `talos`
// shim on PATH still wins when the user made one.
func (t *Talos) command() ([]string, error) {
	if path, err := talosLookPath("talos"); err == nil {
		return []string{path}, nil
	}

	prefix, err := talosPrefix()
	if err != nil {
		return nil, err
	}
	python := filepath.Join(prefix, ".venv", "bin", "python")
	if _, err := os.Stat(python); err == nil {
		return []string{python, "-m", "talos"}, nil
	}

	return nil, fmt.Errorf("talos is not installed")
}

func talosPrefix() (string, error) {
	if prefix := strings.TrimSpace(os.Getenv("TALOS_PREFIX")); prefix != "" {
		return filepath.Clean(prefix), nil
	}
	home, err := talosUserHome()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, "talos"), nil
}

func talosConfigPath() (string, error) {
	prefix, err := talosPrefix()
	if err != nil {
		return "", err
	}
	return filepath.Join(prefix, "talos.env"), nil
}

// talosSecretsEnvPath mirrors Talos's SECRETS_ENV: $TALOS_SECRETS_ENV, or the
// default location next to the other operator secrets.
func talosSecretsEnvPath() (string, error) {
	if path := strings.TrimSpace(os.Getenv("TALOS_SECRETS_ENV")); path != "" {
		return filepath.Clean(path), nil
	}
	home, err := talosUserHome()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".secrets", "talos-telegram.env"), nil
}

// talosConfigValues reads the model settings the way Talos does: talos.env
// first, the secrets file on top, and the process environment last.
func talosConfigValues() map[string]string {
	values := make(map[string]string)
	if configPath, err := talosConfigPath(); err == nil {
		if data, err := os.ReadFile(configPath); err == nil {
			for key, value := range talosParseEnvFile(data) {
				values[key] = value
			}
		}
	}
	if secretsPath, err := talosSecretsEnvPath(); err == nil {
		if data, err := os.ReadFile(secretsPath); err == nil {
			for key, value := range talosParseEnvFile(data) {
				values[key] = value
			}
		}
	}
	for _, key := range []string{talosProviderKey, talosModelKey, "OLLAMA_API_KEY"} {
		// An empty variable is no override — Talos's own loader treats it as unset.
		if value, ok := os.LookupEnv(key); ok && strings.TrimSpace(value) != "" {
			values[key] = value
		}
	}
	return values
}

func talosCloudKeyPresent() bool {
	return strings.TrimSpace(talosConfigValues()["OLLAMA_API_KEY"]) != ""
}

// talosNonDefaultHostWarning flags the one gap launch cannot close: Talos's
// local provider defaults to http://localhost:11434/v1, and the per-provider
// base URL key is POLICY in Talos's schema, which `config set` refuses — even
// with a confirmation. A custom OLLAMA_HOST therefore needs a manual edit.
func talosNonDefaultHostWarning() string {
	host := strings.TrimRight(envconfig.Host().String(), "/")
	if host == "http://127.0.0.1:11434" || host == "http://localhost:11434" {
		return ""
	}
	return fmt.Sprintf("Note: Ollama is at %s, but Talos's local provider defaults to http://localhost:11434/v1 and its base URL is a policy key launch may not write. Set TALOS_BASE_URL_OLLAMA in your Talos env file to %s/v1.", host, host)
}

func talosConfigSet(argv []string, key, value string) error {
	args := append(append([]string(nil), argv[1:]...), "config", "set", key, value)
	out, err := talosCommand(argv[0], args...).CombinedOutput()
	if err != nil {
		return fmt.Errorf("talos config set %s: %w\n%s", key, err, strings.TrimSpace(string(out)))
	}
	return nil
}

// talosParseEnvFile mirrors Talos's own flat KEY=VALUE parser: comments and
// blank lines are dropped, and matching quotes around the value are stripped.
func talosParseEnvFile(data []byte) map[string]string {
	out := make(map[string]string)
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		key, value, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		key = strings.TrimSpace(key)
		if key == "" {
			continue
		}
		value = strings.TrimSpace(value)
		if len(value) >= 2 {
			if (value[0] == '"' && value[len(value)-1] == '"') || (value[0] == '\'' && value[len(value)-1] == '\'') {
				value = value[1 : len(value)-1]
			}
		}
		out[key] = value
	}
	return out
}

func talosAttachedCommand(argv []string, args ...string) *exec.Cmd {
	all := append(append([]string(nil), argv...), args...)
	cmd := talosCommand(all[0], all[1:]...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd
}
