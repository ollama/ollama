package launch

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"os/user"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/ollama/ollama/envconfig"
)

const (
	nanoclawRepoURL     = "https://github.com/nanocoai/nanoclaw"
	nanoclawPackageName = "nanoclaw"
	// nanoclawAgentName is the assistant name launch sets for the initial chat
	// agent NanoClaw creates on first run.
	nanoclawAgentName = "Ollama"
)

// nanoclawGOOS is a seam over runtime.GOOS so the platform-specific Docker
// install hints and the Windows WSL2 guidance can be unit-tested off the target
// OS.
var nanoclawGOOS = runtime.GOOS

// Nanoclaw implements Runner for NanoClaw, a self-hosted assistant that
// orchestrates containerized agents. launch installs it into a fixed,
// launch-owned checkout and hands off to one entrypoint script that owns all
// per-group setup against the local Ollama endpoint (the frozen seam contract).
type Nanoclaw struct{}

func (n *Nanoclaw) String() string { return "NanoClaw" }

// nanoclawDir returns the fixed, launch-owned NanoClaw checkout location under
// the ~/.ollama/launch/ state directory.
func nanoclawDir() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".ollama", "launch", "nanoclaw"), nil
}

// nanoclawScriptPath returns the seam-contract entrypoint inside a checkout.
func nanoclawScriptPath(dir string) string {
	return filepath.Join(dir, "scripts", "ollama-launch.sh")
}

// nanoclawCheckoutValid reports whether dir looks like a real NanoClaw checkout
// (a directory whose package.json declares the nanoclaw package). This only
// confirms the repository is present; first-run setup state is tracked
// separately by the entrypoint script via its idempotency marker.
func nanoclawCheckoutValid(dir string) bool {
	info, err := os.Stat(dir)
	if err != nil || !info.IsDir() {
		return false
	}
	data, err := os.ReadFile(filepath.Join(dir, "package.json"))
	if err != nil {
		return false
	}
	var pkg struct {
		Name string `json:"name"`
	}
	if err := json.Unmarshal(data, &pkg); err != nil {
		return false
	}
	return pkg.Name == nanoclawPackageName
}

// nanoclawInstalled is the registry CheckInstalled probe: is the launch-owned
// NanoClaw checkout present and valid?
func nanoclawInstalled() bool {
	dir, err := nanoclawDir()
	if err != nil {
		return false
	}
	return nanoclawCheckoutValid(dir)
}

// nanoclawOnboarded reports whether NanoClaw has completed first-run setup in
// the given checkout. The setup `service` step writes data/upgrade-state.json;
// its presence is the idempotency marker the entrypoint script also checks, and
// it gates whether launch passes a first-run --display-name.
func nanoclawOnboarded(dir string) bool {
	_, err := os.Stat(filepath.Join(dir, "data", "upgrade-state.json"))
	return err == nil
}

// nanoclawCloneArgs builds the `git clone` arguments for fetching NanoClaw into
// tmp: a shallow clone of the canonical repo's default branch.
//
// It is a package variable, not a plain function, so the local-development file
// nanoclaw_localdev.go can swap in an override that honors the
// NANOCLAW_LAUNCH_REPO / NANOCLAW_LAUNCH_REF environment variables for local
// testing. That file is the ONLY local-dev seam in this package: deleting it
// (and nanoclaw_localdev_test.go) before the production push to ollama/ollama
// restores this canonical behavior with no other change required here.
var nanoclawCloneArgs = func(tmp string) []string {
	return []string{"clone", "--depth", "1", nanoclawRepoURL, tmp}
}

// ensureNanoclawInstalled is the registry EnsureInstalled action: clone NanoClaw
// into the launch-owned directory after explicit consent. It is idempotent — an
// existing valid checkout is a no-op.
func ensureNanoclawInstalled() error {
	dir, err := nanoclawDir()
	if err != nil {
		return err
	}
	if nanoclawCheckoutValid(dir) {
		return nil
	}

	// Installing here is heavyweight and partly irreversible (clones a repo,
	// builds a container image, installs a background service). Require a real
	// terminal so a non-interactive or piped session can never silently kick it
	// off — this is the hard guard against the unattended data-loss path.
	if !isInteractiveSession() {
		return fmt.Errorf("NanoClaw is not installed; run 'ollama launch nanoclaw' in an interactive terminal to install it")
	}

	// Validate every hard prerequisite before asking for consent, and report all
	// missing ones at once, so a user lacking more than one tool isn't made to
	// answer the prompts (or fix tools) one at a time.
	if err := nanoclawCheckPrerequisites(); err != nil {
		return err
	}

	ok, err := ConfirmPrompt("Install NanoClaw? This clones the NanoClaw repo, builds a Docker image (~3-10 min), and installs a background service. NanoClaw runs agent-issued actions inside sandboxed containers.")
	if err != nil {
		return err
	}
	if !ok {
		return fmt.Errorf("nanoclaw installation cancelled")
	}

	fmt.Fprintf(os.Stderr, "\nInstalling NanoClaw into %s...\n", dir)
	if err := os.MkdirAll(filepath.Dir(dir), 0o755); err != nil {
		return fmt.Errorf("failed to prepare NanoClaw install directory: %w", err)
	}

	// Clone into a temp sibling and move it into place only after it validates.
	// This keeps the fixed install dir wedge-proof: an interrupted or invalid
	// clone leaves only the temp dir behind (cleaned here), so a later run never
	// hits "git clone" refusing a non-empty destination, and a stale/invalid
	// pre-existing checkout is replaced rather than blocking the install.
	tmp := dir + ".tmp"
	if err := os.RemoveAll(tmp); err != nil {
		return fmt.Errorf("failed to clear stale NanoClaw download: %w", err)
	}

	clone := exec.Command("git", nanoclawCloneArgs(tmp)...)
	clone.Stdin = os.Stdin
	clone.Stdout = os.Stdout
	clone.Stderr = os.Stderr
	if err := clone.Run(); err != nil {
		_ = os.RemoveAll(tmp)
		return fmt.Errorf("failed to clone NanoClaw: %w", err)
	}
	if !nanoclawCheckoutValid(tmp) {
		_ = os.RemoveAll(tmp)
		return fmt.Errorf("cloned NanoClaw from %s does not look like a valid checkout", nanoclawRepoURL)
	}

	if err := os.RemoveAll(dir); err != nil {
		_ = os.RemoveAll(tmp)
		return fmt.Errorf("failed to replace existing NanoClaw directory: %w", err)
	}
	if err := os.Rename(tmp, dir); err != nil {
		_ = os.RemoveAll(tmp)
		return fmt.Errorf("failed to finalize NanoClaw install: %w", err)
	}

	fmt.Fprintf(os.Stderr, "%sNanoClaw downloaded%s\n\n", ansiGreen, ansiReset)
	return nil
}

// nanoclawCheckPrerequisites verifies git (to clone) and Docker (to build the
// image and run containers), returning one error listing every missing tool
// with install guidance. Prerequisites are never auto-installed: heavyweight,
// OS-specific, and usually need elevated privileges.
func nanoclawCheckPrerequisites() error {
	var missing []string
	if _, err := exec.LookPath("git"); err != nil {
		missing = append(missing, "git: https://git-scm.com/")
	}
	if _, err := exec.LookPath("docker"); err != nil {
		missing = append(missing, "Docker: "+nanoclawDockerInstallURL())
	}
	if len(missing) == 0 {
		return nil
	}
	return fmt.Errorf("NanoClaw is not installed and required dependencies are missing\n\nInstall the following first:\n  %s\n\nThen re-run:\n  ollama launch nanoclaw", strings.Join(missing, "\n  "))
}

// nanoclawDockerInstallURL returns the Docker install page for the current OS.
func nanoclawDockerInstallURL() string {
	switch nanoclawGOOS {
	case "darwin":
		return "https://docs.docker.com/desktop/install/mac-install/"
	case "windows":
		return "https://docs.docker.com/desktop/install/windows-install/"
	default:
		return "https://docs.docker.com/engine/install/"
	}
}

// nanoclawEnv returns the parent environment with cloud LLM provider API keys
// stripped before handing off to the entrypoint script. NanoClaw talks to the
// local Ollama endpoint (passed via --base-url) and runs agent-issued actions
// inside containers, so ambient provider credentials from the user's shell must
// not leak into the script, the Docker build, or those agents.
func nanoclawEnv() []string {
	clear := map[string]bool{
		"ANTHROPIC_API_KEY":     true,
		"ANTHROPIC_OAUTH_TOKEN": true,
		"OPENAI_API_KEY":        true,
		"GEMINI_API_KEY":        true,
		"MISTRAL_API_KEY":       true,
		"GROQ_API_KEY":          true,
		"XAI_API_KEY":           true,
		"OPENROUTER_API_KEY":    true,
	}
	env := make([]string, 0, len(os.Environ()))
	for _, e := range os.Environ() {
		key, _, _ := strings.Cut(e, "=")
		if !clear[key] {
			env = append(env, e)
		}
	}
	return env
}

// nanoclawDisplayName derives the operator display name for the initial chat
// agent created on first run, falling back to a stable default when the OS user
// is unavailable. The entrypoint script normalizes it into the agent folder.
func nanoclawDisplayName() string {
	if u, err := user.Current(); err == nil {
		if name := strings.TrimSpace(u.Username); name != "" {
			return name
		}
	}
	return "operator"
}

func (n *Nanoclaw) Run(model string, _ []LaunchModel, args []string) error {
	// Ensure the checkout exists before handing off. This is idempotent and
	// returns immediately when NanoClaw is already installed.
	if err := ensureNanoclawInstalled(); err != nil {
		return err
	}

	dir, err := nanoclawDir()
	if err != nil {
		return err
	}

	script := nanoclawScriptPath(dir)
	if _, err := os.Stat(script); err != nil {
		return fmt.Errorf("NanoClaw entrypoint not found at %s\n\nUpdate the checkout (git -C %q pull) or reinstall.", script, dir)
	}

	if _, err := exec.LookPath("bash"); err != nil {
		msg := "NanoClaw's launch script requires bash, which was not found on PATH"
		if nanoclawGOOS == "windows" {
			msg += "\n\nNanoClaw runs best under WSL2 on Windows (wsl --install)."
		}
		return fmt.Errorf("%s", msg)
	}

	// The frozen seam contract: launch passes the model id and host view of the
	// Ollama endpoint; the script rewrites the host to host.docker.internal and
	// owns per-group setup. Use ConnectableHost (not Host) so an OLLAMA_HOST on
	// 0.0.0.0/:: normalizes to a loopback the container can reach. --display-name
	// and --agent-name matter only on first run, so omit them once onboarded.
	scriptArgs := []string{
		script,
		"--model", model,
		"--base-url", envconfig.ConnectableHost().String(),
	}
	if !nanoclawOnboarded(dir) {
		scriptArgs = append(scriptArgs,
			"--display-name", nanoclawDisplayName(),
			"--agent-name", nanoclawAgentName,
		)
	}
	scriptArgs = append(scriptArgs, args...)

	// stdio is inherited so the script's consent-free prompts, progress, and the
	// final CHAT: line reach the user directly; exit codes propagate verbatim.
	cmd := exec.Command("bash", scriptArgs...)
	cmd.Dir = dir
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Env = nanoclawEnv()
	return cmd.Run()
}
