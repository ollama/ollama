package launch

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strings"

	"github.com/pelletier/go-toml/v2"

	"github.com/ollama/ollama/cmd/internal/fileutil"
	"github.com/ollama/ollama/envconfig"
	modelpkg "github.com/ollama/ollama/types/model"
)

// Kimi implements Runner for Kimi Code CLI integration.
type Kimi struct{}

const (
	kimiDefaultModelAlias     = "ollama"
	kimiDefaultMaxContextSize = 32768
	kimiCodeProviderName      = "ollama"
	kimiLegacyVersionPrefix   = "kimi, version "
)

var kimiGOOS = runtime.GOOS

func (k *Kimi) String() string { return "Kimi Code CLI" }

func (k *Kimi) args(config string, extra []string) []string {
	args := []string{"--config", config}
	args = append(args, extra...)
	return args
}

func (k *Kimi) Run(model string, models []LaunchModel, args []string) error {
	if strings.TrimSpace(model) == "" {
		return fmt.Errorf("model is required")
	}
	if err := validateKimiPassthroughArgs(args); err != nil {
		return err
	}

	bin, err := ensureKimiInstalled()
	if err != nil {
		return err
	}

	var cmdArgs []string
	if _, legacy := kimiLegacyCLIVersion(bin); legacy {
		config, err := buildKimiInlineConfig(model, resolveKimiMaxContextSize(model))
		if err != nil {
			return fmt.Errorf("failed to build kimi config: %w", err)
		}
		cmdArgs = k.args(config, args)
	} else {
		if err := writeKimiCodeConfig(model, models); err != nil {
			return fmt.Errorf("failed to configure kimi: %w", err)
		}
		cmdArgs = append([]string{"-m", kimiCodeModelAlias(model)}, args...)
	}

	cmd := exec.Command(bin, cmdArgs...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

func kimiLegacyCLIVersion(bin string) (string, bool) {
	out, err := exec.Command(bin, "--version").Output()
	if err != nil {
		return "", false
	}
	line, _, _ := strings.Cut(strings.TrimSpace(string(out)), "\n")
	version, ok := strings.CutPrefix(strings.TrimSpace(line), kimiLegacyVersionPrefix)
	if !ok {
		return "", false
	}
	return strings.TrimSpace(version), true
}

func kimiCodeModelAlias(model string) string {
	return kimiCodeProviderName + "/" + model
}

func kimiCodeConfigPath() (string, error) {
	if home := strings.TrimSpace(os.Getenv("KIMI_CODE_HOME")); home != "" {
		return filepath.Join(home, "config.toml"), nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".kimi-code", "config.toml"), nil
}

func writeKimiCodeConfig(primary string, models []LaunchModel) error {
	path, err := kimiCodeConfigPath()
	if err != nil {
		return err
	}

	cfg := map[string]any{}
	exists := false
	data, err := os.ReadFile(path)
	switch {
	case err == nil:
		exists = true
		if err := toml.Unmarshal(data, &cfg); err != nil {
			return fmt.Errorf("parse %s: %w", path, err)
		}
	case !os.IsNotExist(err):
		return err
	}

	models = codexCatalogModels(primary, models)
	aliases := make(map[string]bool, len(models))
	for _, m := range models {
		aliases[kimiCodeModelAlias(m.Name)] = true
	}

	modelsTable, _ := cfg["models"].(map[string]any)
	providersTable, _ := cfg["providers"].(map[string]any)

	defaultModel, _ := cfg["default_model"].(string)
	nextDefault := defaultModel
	if !kimiCodeDefaultSurvives(modelsTable, aliases, defaultModel) {
		nextDefault = kimiCodeModelAlias(primary)
	}

	provider := kimiCodeProviderValues(providersTable[kimiCodeProviderName])
	entries := make(map[string]map[string]any, len(models))
	for _, m := range models {
		alias := kimiCodeModelAlias(m.Name)
		entries[alias] = kimiCodeModelEntryFor(m, modelsTable[alias])
	}

	updated := kimiCodeRemoveDottedAssignments(string(data), aliases)
	updated = kimiCodeRemoveOwnedSections(updated, aliases)

	sections, err := kimiCodeSections([]string{"providers", kimiCodeProviderName}, provider)
	if err != nil {
		return err
	}
	for _, m := range models {
		alias := kimiCodeModelAlias(m.Name)
		entry := entries[alias]
		if overrides, ok := kimiCodeModelOverrides(modelsTable, alias); ok {
			entry["overrides"] = overrides
		}
		modelSections, err := kimiCodeSections([]string{"models", alias}, entry)
		if err != nil {
			return err
		}
		sections = append(sections, modelSections...)
	}
	updated = kimiCodeAppendSections(updated, sections)
	updated = codexSetRootStringValue(updated, "default_model", nextDefault)

	if err := toml.Unmarshal([]byte(updated), &map[string]any{}); err != nil {
		merged, mergeErr := kimiCodeMergedConfig(cfg, provider, entries, nextDefault)
		if mergeErr != nil {
			return fmt.Errorf("kimi config %s could not be updated: %w", path, mergeErr)
		}
		updated = string(merged)
	}

	if exists {
		if updated == string(data) {
			return nil
		}
		if kimiCodeHasUserContent(cfg) {
			ok, err := ConfirmPrompt(fmt.Sprintf("Kimi Code config %s already has settings. Update it for Ollama now? (your settings are kept and a backup is saved)", path))
			if err != nil {
				return err
			}
			if !ok {
				return fmt.Errorf("kimi config %s was not updated; re-run with --yes to update it", path)
			}
		}
	}

	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	return fileutil.WriteWithBackup(path, []byte(updated), "kimi")
}

func kimiCodeDefaultSurvives(modelsTable map[string]any, aliases map[string]bool, alias string) bool {
	if alias == "" {
		return false
	}
	if aliases[alias] {
		return true
	}
	if _, ok := modelsTable[alias]; ok {
		return !kimiCodeOwnedAlias(alias)
	}
	return false
}

func kimiCodeOwnedAlias(alias string) bool {
	return strings.HasPrefix(alias, kimiCodeProviderName+"/")
}

func kimiCodeRemoveOwnedSections(text string, aliases map[string]bool) string {
	var headers []string
	for _, line := range strings.SplitAfter(text, "\n") {
		header := strings.TrimSpace(line)
		path, ok := codexTableHeaderPath(header)
		if !ok || len(path) < 2 {
			continue
		}
		switch path[0] {
		case "providers":
			if len(path) == 2 && path[1] == kimiCodeProviderName {
				headers = append(headers, header)
			}
		case "models":
			if !aliases[path[1]] {
				continue
			}
			headers = append(headers, header)
		}
	}
	for _, header := range headers {
		text = codexRemoveSection(text, header)
	}
	return text
}

func kimiCodeRemoveDottedAssignments(text string, aliases map[string]bool) string {
	lines := strings.SplitAfter(text, "\n")
	out := make([]string, 0, len(lines))
	var table []string
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if path, ok := codexTableHeaderPath(trimmed); ok {
			table = path
			out = append(out, line)
			continue
		}
		if strings.HasPrefix(trimmed, "[[") {
			table = nil
			out = append(out, line)
			continue
		}
		if kimiCodeDottedManaged(trimmed, table, aliases) {
			continue
		}
		out = append(out, line)
	}
	return strings.Join(out, "")
}

func kimiCodeDottedManaged(line string, table []string, aliases map[string]bool) bool {
	if line == "" || strings.HasPrefix(line, "#") || strings.HasPrefix(line, "[") {
		return false
	}
	key, _, ok := strings.Cut(line, "=")
	if !ok {
		return false
	}
	path, ok := kimiCodeKeyPath(strings.TrimSpace(key))
	if !ok {
		return false
	}
	path = append(append([]string(nil), table...), path...)
	if len(path) < 2 {
		return false
	}
	if len(table) >= 2 {
		return false
	}
	switch path[0] {
	case "providers":
		return path[1] == kimiCodeProviderName
	case "models":
		return aliases[path[1]]
	}
	return false
}

func kimiCodeKeyPath(key string) ([]string, bool) {
	const probe = "__ollama_launch_probe"
	cfg := map[string]any{}
	if err := toml.Unmarshal([]byte(key+"."+probe+" = true\n"), &cfg); err != nil {
		return nil, false
	}
	return codexFindProbePath(cfg, probe, nil)
}

func kimiCodeProviderValues(existing any) map[string]any {
	values := map[string]any{
		"type":     "openai",
		"base_url": envconfig.ConnectableHost().String() + "/v1",
		"api_key":  "ollama",
	}
	table, ok := existing.(map[string]any)
	if !ok {
		return values
	}
	for key, value := range table {
		if _, ours := values[key]; !ours {
			values[key] = value
		}
	}
	return values
}

func kimiCodeModelEntryFor(m LaunchModel, existing any) map[string]any {
	entry := kimiCodeModelEntry(m)
	table, ok := existing.(map[string]any)
	if !ok {
		return entry
	}
	for key, value := range table {
		if key == "overrides" {
			continue
		}
		if _, ours := entry[key]; !ours {
			entry[key] = value
		}
	}
	return entry
}

func kimiCodeMergedConfig(cfg map[string]any, provider map[string]any, entries map[string]map[string]any, nextDefault string) ([]byte, error) {
	providers, ok := cfg["providers"].(map[string]any)
	if !ok {
		providers = map[string]any{}
		cfg["providers"] = providers
	}
	providers[kimiCodeProviderName] = provider

	modelsTable, ok := cfg["models"].(map[string]any)
	if !ok {
		modelsTable = map[string]any{}
		cfg["models"] = modelsTable
	}
	for alias, entry := range entries {
		if overrides, ok := kimiCodeModelOverrides(modelsTable, alias); ok {
			entry["overrides"] = overrides
		}
		modelsTable[alias] = entry
	}
	cfg["default_model"] = nextDefault
	return toml.Marshal(cfg)
}

func kimiCodeModelOverrides(modelsTable map[string]any, alias string) (map[string]any, bool) {
	entry, ok := modelsTable[alias].(map[string]any)
	if !ok {
		return nil, false
	}
	overrides, ok := entry["overrides"].(map[string]any)
	if !ok || len(overrides) == 0 {
		return nil, false
	}
	return overrides, true
}

func kimiCodeSections(path []string, values map[string]any) ([]string, error) {
	scalars := make(map[string]any, len(values))
	var nested []string
	for key, value := range values {
		if _, ok := value.(map[string]any); ok {
			nested = append(nested, key)
			continue
		}
		scalars[key] = value
	}
	slices.Sort(nested)

	body, err := toml.Marshal(scalars)
	if err != nil {
		return nil, err
	}
	sections := []string{kimiCodeHeader(path) + "\n" + string(body)}
	for _, key := range nested {
		child, err := kimiCodeSections(append(append([]string(nil), path...), key), values[key].(map[string]any))
		if err != nil {
			return nil, err
		}
		sections = append(sections, child...)
	}
	return sections, nil
}

func kimiCodeHeader(path []string) string {
	parts := make([]string, len(path))
	for i, part := range path {
		parts[i] = fmt.Sprintf("%q", part)
	}
	return "[" + strings.Join(parts, ".") + "]"
}

func kimiCodeAppendSections(text string, sections []string) string {
	out := text
	for _, section := range sections {
		if out != "" {
			if !strings.HasSuffix(out, "\n") {
				out += "\n"
			}
			if !strings.HasSuffix(out, "\n\n") {
				out += "\n"
			}
		}
		out += strings.TrimRight(section, "\n") + "\n"
	}
	return out
}

func kimiCodeHasUserContent(cfg map[string]any) bool {
	for key, value := range cfg {
		switch key {
		case "providers":
			table, ok := value.(map[string]any)
			if !ok {
				return true
			}
			for name := range table {
				if name != kimiCodeProviderName {
					return true
				}
			}
		case "models":
			table, ok := value.(map[string]any)
			if !ok {
				return true
			}
			for alias := range table {
				if !kimiCodeOwnedAlias(alias) {
					return true
				}
			}
		case "default_model":
		default:
			return true
		}
	}
	return false
}

func kimiCodeModelEntry(m LaunchModel) map[string]any {
	if m.Thinking == nil && m.HasCapability(modelpkg.CapabilityThinking) {
		if meta, ok := lookupServerModelMetadata(m.Name); ok {
			m.Thinking = meta.Thinking.Clone()
		}
	}
	maxContextSize := m.ContextLength
	if maxContextSize <= 0 {
		maxContextSize = resolveKimiMaxContextSize(m.Name)
	}
	entry := map[string]any{
		"provider":         kimiCodeProviderName,
		"model":            m.Name,
		"max_context_size": int64(maxContextSize),
	}

	capabilities := []string{}
	if m.ToolCapable || m.HasCapability(modelpkg.CapabilityTools) {
		capabilities = append(capabilities, "tool_use")
	}
	if m.HasCapability(modelpkg.CapabilityVision) {
		capabilities = append(capabilities, "image_in")
	}

	levels, defaultLevel := launchReasoningEffortLevels(m.Thinking)
	efforts := slices.DeleteFunc(slices.Clone(levels), func(level string) bool { return level == "none" })
	if len(efforts) > 0 || m.HasCapability(modelpkg.CapabilityThinking) {
		capabilities = append(capabilities, "thinking")
		if slices.Contains(levels, "none") {
			entry["off_effort"] = "none"
		} else if len(efforts) > 0 {
			capabilities = append(capabilities, "always_thinking")
		}
		if len(efforts) > 0 {
			entry["support_efforts"] = kimiCodeStringList(efforts)
			if slices.Contains(efforts, defaultLevel) {
				entry["default_effort"] = defaultLevel
			}
		}
	}
	entry["capabilities"] = kimiCodeStringList(capabilities)
	return entry
}

func kimiCodeStringList(values []string) []any {
	out := make([]any, len(values))
	for i, value := range values {
		out[i] = value
	}
	return out
}

func (k *Kimi) ConfigureOnly(model string, models []LaunchModel) error {
	return writeKimiCodeConfig(model, models)
}

func findKimiBinary() (string, error) {
	home, _ := os.UserHomeDir()

	kimiCodeBin := filepath.Join(home, ".kimi-code", "bin", "kimi")
	if kimiGOOS == "windows" {
		kimiCodeBin += ".exe"
	}
	if info, err := os.Stat(kimiCodeBin); err == nil && !info.IsDir() && (kimiGOOS == "windows" || info.Mode().Perm()&0o111 != 0) {
		return kimiCodeBin, nil
	}

	if path, err := exec.LookPath("kimi"); err == nil {
		return path, nil
	}

	var candidates []string
	switch kimiGOOS {
	case "windows":
		candidates = appendWindowsKimiCandidates(candidates, filepath.Join(home, ".local", "bin"))
		candidates = appendWindowsKimiCandidates(candidates, filepath.Join(home, "bin"))

		if appData := strings.TrimSpace(os.Getenv("APPDATA")); appData != "" {
			candidates = appendWindowsKimiCandidates(candidates, filepath.Join(appData, "uv", "bin"))
		}
		if localAppData := strings.TrimSpace(os.Getenv("LOCALAPPDATA")); localAppData != "" {
			candidates = appendWindowsKimiCandidates(candidates, filepath.Join(localAppData, "uv", "bin"))
		}
	default:
		candidates = append(candidates,
			filepath.Join(home, ".local", "bin", "kimi"),
			filepath.Join(home, "bin", "kimi"),
			filepath.Join(home, ".local", "share", "uv", "tools", "kimi-cli", "bin", "kimi"),
			filepath.Join(home, ".local", "share", "uv", "tools", "kimi", "bin", "kimi"),
		)

		if xdgDataHome := strings.TrimSpace(os.Getenv("XDG_DATA_HOME")); xdgDataHome != "" {
			candidates = append(candidates,
				filepath.Join(xdgDataHome, "uv", "tools", "kimi-cli", "bin", "kimi"),
				filepath.Join(xdgDataHome, "uv", "tools", "kimi", "bin", "kimi"),
			)
		}

		// WSL users can inherit Windows env vars while launching from Linux shells.
		if profile := windowsPathToWSL(os.Getenv("USERPROFILE")); profile != "" {
			candidates = appendWindowsKimiCandidates(candidates, filepath.Join(profile, ".local", "bin"))
		}
		if appData := windowsPathToWSL(os.Getenv("APPDATA")); appData != "" {
			candidates = appendWindowsKimiCandidates(candidates, filepath.Join(appData, "uv", "bin"))
		}
		if localAppData := windowsPathToWSL(os.Getenv("LOCALAPPDATA")); localAppData != "" {
			candidates = appendWindowsKimiCandidates(candidates, filepath.Join(localAppData, "uv", "bin"))
		}
	}

	for _, candidate := range candidates {
		if info, err := os.Stat(candidate); err == nil && !info.IsDir() {
			return candidate, nil
		}
	}

	return "", fmt.Errorf("kimi binary not found")
}

func appendWindowsKimiCandidates(candidates []string, dir string) []string {
	if strings.TrimSpace(dir) == "" {
		return candidates
	}

	return append(candidates,
		filepath.Join(dir, "kimi.exe"),
		filepath.Join(dir, "kimi.cmd"),
		filepath.Join(dir, "kimi.bat"),
	)
}

func windowsPathToWSL(path string) string {
	trimmed := strings.TrimSpace(path)
	if len(trimmed) < 3 || trimmed[1] != ':' {
		return ""
	}

	drive := strings.ToLower(string(trimmed[0]))
	rest := strings.ReplaceAll(trimmed[2:], "\\", "/")
	rest = strings.TrimPrefix(rest, "/")
	if rest == "" {
		return filepath.Join("/mnt", drive)
	}

	return filepath.Join("/mnt", drive, rest)
}

func validateKimiPassthroughArgs(args []string) error {
	for _, arg := range args {
		switch {
		case arg == "--config", strings.HasPrefix(arg, "--config="):
			return fmt.Errorf("conflicting extra argument %q: ollama launch kimi manages --config", arg)
		case arg == "--config-file", strings.HasPrefix(arg, "--config-file="):
			return fmt.Errorf("conflicting extra argument %q: ollama launch kimi manages --config-file", arg)
		case arg == "--model", strings.HasPrefix(arg, "--model="):
			return fmt.Errorf("conflicting extra argument %q: ollama launch kimi manages --model", arg)
		case arg == "-m", strings.HasPrefix(arg, "-m="):
			return fmt.Errorf("conflicting extra argument %q: ollama launch kimi manages -m/--model", arg)
		}
	}
	return nil
}

func buildKimiInlineConfig(model string, maxContextSize int) (string, error) {
	cfg := map[string]any{
		"default_model": kimiDefaultModelAlias,
		"providers": map[string]any{
			kimiDefaultModelAlias: map[string]any{
				"type":     "openai_legacy",
				"base_url": envconfig.ConnectableHost().String() + "/v1",
				"api_key":  "ollama",
			},
		},
		"models": map[string]any{
			kimiDefaultModelAlias: map[string]any{
				"provider":         kimiDefaultModelAlias,
				"model":            model,
				"max_context_size": maxContextSize,
			},
		},
	}

	data, err := json.Marshal(cfg)
	if err != nil {
		return "", err
	}
	return string(data), nil
}

func resolveKimiMaxContextSize(model string) int {
	if l, ok := lookupCloudModelLimit(model); ok {
		return l.Context
	}

	if n := showContextLength(model); n > 0 {
		return n
	}

	return kimiDefaultMaxContextSize
}

func ensureKimiInstalled() (string, error) {
	if path, err := findKimiBinary(); err == nil {
		return path, nil
	}

	if err := checkKimiInstallerDependencies(); err != nil {
		return "", err
	}

	ok, err := ConfirmPrompt("Kimi is not installed. Install now?")
	if err != nil {
		return "", err
	}
	if !ok {
		return "", fmt.Errorf("kimi installation cancelled")
	}

	bin, args, err := kimiInstallerCommand(kimiGOOS)
	if err != nil {
		return "", err
	}

	fmt.Fprintf(os.Stderr, "\nInstalling Kimi...\n")
	cmd := exec.Command(bin, args...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("failed to install kimi: %w", err)
	}

	path, err := findKimiBinary()
	if err != nil {
		return "", fmt.Errorf("kimi was installed but the binary was not found on PATH\n\nYou may need to restart your shell")
	}

	fmt.Fprintf(os.Stderr, "%sKimi installed successfully%s\n\n", ansiGreen, ansiReset)
	return path, nil
}

func checkKimiInstallerDependencies() error {
	switch kimiGOOS {
	case "windows":
		if _, err := exec.LookPath("powershell"); err != nil {
			return fmt.Errorf("kimi is not installed and required dependencies are missing\n\nInstall the following first:\n  PowerShell: https://learn.microsoft.com/powershell/\n\nThen re-run:\n  ollama launch kimi")
		}
	default:
		var missing []string
		if _, err := exec.LookPath("curl"); err != nil {
			missing = append(missing, "curl: https://curl.se/")
		}
		if _, err := exec.LookPath("bash"); err != nil {
			missing = append(missing, "bash: https://www.gnu.org/software/bash/")
		}
		if len(missing) > 0 {
			return fmt.Errorf("kimi is not installed and required dependencies are missing\n\nInstall the following first:\n  %s\n\nThen re-run:\n  ollama launch kimi", strings.Join(missing, "\n  "))
		}
	}
	return nil
}

func kimiInstallerCommand(goos string) (string, []string, error) {
	switch goos {
	case "windows":
		return "powershell", []string{
			"-NoProfile",
			"-ExecutionPolicy",
			"Bypass",
			"-Command",
			"Invoke-RestMethod https://code.kimi.com/install.ps1 | Invoke-Expression",
		}, nil
	case "darwin", "linux":
		return "bash", []string{
			"-c",
			"curl -LsSf https://code.kimi.com/install.sh | bash",
		}, nil
	default:
		return "", nil, fmt.Errorf("unsupported platform for kimi install: %s", goos)
	}
}
