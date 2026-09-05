package launch

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/ollama/ollama/cmd/internal/fileutil"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/types/model"
	"github.com/pelletier/go-toml/v2"
	"golang.org/x/mod/semver"
)

// Codex implements Runner for Codex integration
type Codex struct {
	resolveContext func(LaunchModel, bool) contextWindowResolution
}

func (c *Codex) String() string { return "Codex CLI" }

const (
	codexProfileName           = "ollama-launch"
	codexProviderName          = "Ollama"
	codexFallbackContextWindow = 128_000
	codexRestoreSuccess        = "Codex launch configuration removed."

	codexRootProfileKey          = "profile"
	codexRootModelKey            = "model"
	codexRootModelProviderKey    = "model_provider"
	codexRootModelCatalogJSONKey = "model_catalog_json"
	codexRootOpenAIBaseURLKey    = "openai_base_url"
)

func (c *Codex) args(model, modelCatalogPath string, contextWindow int, extra []string) ([]string, error) {
	if err := codexValidateExtraArgs(extra); err != nil {
		return nil, err
	}

	args := []string{"--profile", codexProfileName}
	for _, override := range codexManagedConfigOverrides(modelCatalogPath) {
		args = append(args, "-c", override)
	}
	if contextWindow > 0 {
		args = append(args, "-c", fmt.Sprintf("model_context_window=%d", contextWindow))
	}
	if model != "" {
		args = append(args, "-m", model)
	}
	args = append(args, extra...)
	return args, nil
}

func (c *Codex) Run(model string, models []LaunchModel, args []string) error {
	if err := codexValidateExtraArgs(args); err != nil {
		return fmt.Errorf("failed to configure codex: %w", err)
	}
	if err := checkCodexVersion(); err != nil {
		return err
	}

	launchModel := codexCatalogModel(model, models)
	resolution := c.contextWindow(launchModel)
	codexWarnUnverifiedContext(model, resolution)
	if resolution.ContextLength > 0 {
		launchModel.ContextLength = resolution.ContextLength
	}

	inheritedContext, err := codexConfiguredContextWindow()
	if err != nil {
		return fmt.Errorf("failed to configure codex: %w", err)
	}
	effectiveContext, cleanArgs, corrected, err := codexReconcileContextWindow(resolution.ContextLength, inheritedContext, args)
	if err != nil {
		return fmt.Errorf("failed to configure codex: %w", err)
	}
	if corrected {
		fmt.Fprintf(os.Stderr, "%s  Warning: configured model_context_window exceeds Ollama's %d-token context; using %d%s\n", ansiYellow, resolution.ContextLength, effectiveContext, ansiReset)
	}

	if err := ensureCodexConfig(model, []LaunchModel{launchModel}); err != nil {
		return fmt.Errorf("failed to configure codex: %w", err)
	}

	catalogPath, err := codexModelCatalogPath()
	if err != nil {
		return fmt.Errorf("failed to configure codex: %w", err)
	}

	codexArgs, err := c.args(model, catalogPath, effectiveContext, cleanArgs)
	if err != nil {
		return fmt.Errorf("failed to configure codex: %w", err)
	}

	cmd := exec.Command("codex", codexArgs...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Env = append(os.Environ(),
		"OPENAI_API_KEY=ollama",
	)
	return cmd.Run()
}

func (c *Codex) contextWindow(model LaunchModel) contextWindowResolution {
	if c != nil && c.resolveContext != nil {
		return c.resolveContext(model, true)
	}
	return resolveLaunchContextFromEnvironment(model, true)
}

func codexWarnUnverifiedContext(model string, resolution contextWindowResolution) {
	if resolution.RuntimeVerified || isCloudModelName(model) {
		return
	}
	if resolution.ContextLength > 0 {
		fmt.Fprintf(os.Stderr, "%s  Warning: could not verify %s's loaded context; using %d from %s%s\n", ansiYellow, model, resolution.ContextLength, resolution.Source, ansiReset)
		return
	}
	fmt.Fprintf(os.Stderr, "%s  Warning: could not determine %s's context; Codex will use its existing fallback%s\n", ansiYellow, model, ansiReset)
}

func codexConfiguredContextWindow() (int, error) {
	configPath, err := codexConfigPath()
	if err != nil {
		return 0, err
	}
	data, err := os.ReadFile(configPath)
	if err != nil {
		if os.IsNotExist(err) {
			return 0, nil
		}
		return 0, err
	}
	config, err := codexParseConfig(string(data))
	if err != nil {
		return 0, err
	}
	value, _ := config.Int("model_context_window")
	return value, nil
}

func codexReconcileContextWindow(available, inherited int, extra []string) (int, []string, bool, error) {
	if available <= 0 {
		return 0, append([]string(nil), extra...), false, nil
	}

	requested, found, cleaned, err := codexExtractContextWindowOverride(extra)
	if err != nil {
		return 0, nil, false, err
	}
	if !found {
		requested = inherited
	}
	if requested <= 0 {
		requested = available
	}
	corrected := requested > available
	if corrected {
		requested = available
	}
	return requested, cleaned, corrected, nil
}

func codexExtractContextWindowOverride(args []string) (int, bool, []string, error) {
	cleaned := make([]string, 0, len(args))
	var requested int
	var found bool
	for i := 0; i < len(args); i++ {
		arg := args[i]
		if (arg == "-c" || arg == "--config") && i+1 < len(args) {
			if value, ok, err := codexContextWindowAssignment(args[i+1]); ok || err != nil {
				if err != nil {
					return 0, false, nil, err
				}
				requested, found = value, true
				i++
				continue
			}
			cleaned = append(cleaned, arg, args[i+1])
			i++
			continue
		}

		assignment := ""
		switch {
		case strings.HasPrefix(arg, "--config="):
			assignment = strings.TrimPrefix(arg, "--config=")
		case strings.HasPrefix(arg, "-c") && len(arg) > len("-c"):
			assignment = strings.TrimPrefix(strings.TrimPrefix(arg, "-c"), "=")
		}
		if assignment != "" {
			if value, ok, err := codexContextWindowAssignment(assignment); ok || err != nil {
				if err != nil {
					return 0, false, nil, err
				}
				requested, found = value, true
				continue
			}
		}
		cleaned = append(cleaned, arg)
	}
	return requested, found, cleaned, nil
}

func codexContextWindowAssignment(assignment string) (int, bool, error) {
	key, value, ok := strings.Cut(strings.TrimSpace(assignment), "=")
	if !ok || strings.Trim(strings.TrimSpace(key), `"'`) != "model_context_window" {
		return 0, false, nil
	}
	value = strings.ReplaceAll(strings.TrimSpace(value), "_", "")
	n, err := strconv.Atoi(value)
	if err != nil || n <= 0 {
		return 0, true, fmt.Errorf("invalid model_context_window override %q", assignment)
	}
	return n, true, nil
}

func (c *Codex) Restore() error {
	configPath, err := codexConfigPath()
	if err != nil {
		return err
	}

	if err := removeCodexProfileConfig(); err != nil {
		return codexRestoreFailure(configPath, err)
	}
	if err := removeCodexModelCatalogIfUnused(configPath); err != nil {
		return codexRestoreFailure(configPath, err)
	}
	return nil
}

func (c *Codex) RestoreSuccessMessage() string {
	return codexRestoreSuccess
}

func (c *Codex) SkipRestoreInstallCheck() bool {
	return true
}

func codexRestoreFailure(configPath string, err error) error {
	return fmt.Errorf("restore Codex config: %w\n\nRestore did not complete. Check these files before retrying:\n  Codex config: %s\n  CLI profile: %s\n  CLI model catalog: %s\n  Backups: %s",
		err,
		configPath,
		codexProfileConfigPathForConfig(configPath),
		codexModelCatalogPathForConfig(configPath),
		fileutil.BackupDir(),
	)
}

func removeCodexProfileConfig() error {
	profilePath, err := codexProfileConfigPath()
	if err != nil {
		return err
	}
	return removeCodexFile(profilePath)
}

func removeCodexModelCatalogIfUnused(configPath string) error {
	catalogPath := codexModelCatalogPathForConfig(configPath)
	data, err := os.ReadFile(configPath)
	if err != nil && !os.IsNotExist(err) {
		return err
	}
	if err == nil {
		config, parseErr := codexParseConfig(string(data))
		if parseErr != nil {
			return parseErr
		}
		if config.RootString(codexRootModelCatalogJSONKey) == catalogPath {
			return nil
		}
	}
	return removeCodexFile(catalogPath)
}

func removeCodexFile(path string) error {
	if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
		return err
	}
	return nil
}

func codexValidateExtraArgs(args []string) error {
	for i, arg := range args {
		switch {
		case arg == "-p", strings.HasPrefix(arg, "-p"):
			return fmt.Errorf("conflicting extra argument %q: ollama launch codex manages --profile", arg)
		case arg == "--profile", strings.HasPrefix(arg, "--profile="):
			return fmt.Errorf("conflicting extra argument %q: ollama launch codex manages --profile", arg)
		case arg == "-m", strings.HasPrefix(arg, "-m"):
			return fmt.Errorf("conflicting extra argument %q: ollama launch codex manages --model", arg)
		case arg == "--model", strings.HasPrefix(arg, "--model="):
			return fmt.Errorf("conflicting extra argument %q: ollama launch codex manages --model", arg)
		case arg == "-c", arg == "--config":
			if i+1 < len(args) && codexConfigOverrideConflicts(args[i+1]) {
				return fmt.Errorf("conflicting extra config %q: ollama launch codex manages provider and model catalog config", args[i+1])
			}
		case strings.HasPrefix(arg, "-c") && len(arg) > len("-c"):
			if codexConfigOverrideConflicts(strings.TrimPrefix(arg, "-c")) {
				return fmt.Errorf("conflicting extra config %q: ollama launch codex manages provider and model catalog config", arg)
			}
		case strings.HasPrefix(arg, "--config="):
			if codexConfigOverrideConflicts(strings.TrimPrefix(arg, "--config=")) {
				return fmt.Errorf("conflicting extra config %q: ollama launch codex manages provider and model catalog config", arg)
			}
		}
	}
	return nil
}

func codexManagedConfigOverrides(modelCatalogPath string) []string {
	overrides := []string{
		fmt.Sprintf("%s=%q", codexRootModelProviderKey, codexProfileName),
		fmt.Sprintf("model_providers.%s.name=%q", codexProfileName, codexProviderName),
		fmt.Sprintf("model_providers.%s.base_url=%q", codexProfileName, codexBaseURL()),
		fmt.Sprintf("model_providers.%s.wire_api=%q", codexProfileName, "responses"),
	}
	if modelCatalogPath != "" {
		overrides = append(overrides, fmt.Sprintf("%s=%q", codexRootModelCatalogJSONKey, modelCatalogPath))
	}
	return overrides
}

func codexConfigOverrideConflicts(value string) bool {
	key, _, ok := strings.Cut(strings.TrimSpace(value), "=")
	if !ok {
		return false
	}
	key = strings.TrimSpace(key)
	key = strings.Trim(key, `"'`)
	switch {
	case key == codexRootProfileKey,
		key == codexRootModelKey,
		key == codexRootModelProviderKey,
		key == codexRootModelCatalogJSONKey:
		return true
	case strings.HasPrefix(key, "model_providers."):
		return true
	}
	return false
}

// ensureCodexConfig writes a Codex profile file and model catalog so Codex uses
// the local Ollama server without changing app-visible root config.
func ensureCodexConfig(modelName string, models []LaunchModel) error {
	configPath, err := codexConfigPath()
	if err != nil {
		return err
	}

	codexDir := filepath.Dir(configPath)
	if err := os.MkdirAll(codexDir, 0o755); err != nil {
		return err
	}
	if err := cleanupCodexLegacyProfileConfig(configPath); err != nil {
		return err
	}

	catalogPath := codexModelCatalogPathForConfig(configPath)
	if err := writeCodexModelCatalog(catalogPath, codexCatalogModel(modelName, models)); err != nil {
		return err
	}

	profilePath := codexProfileConfigPathForConfig(configPath)
	return writeCodexProfileConfig(profilePath, modelName, catalogPath)
}

func codexConfigPath() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".codex", "config.toml"), nil
}

func codexModelCatalogPath() (string, error) {
	configPath, err := codexConfigPath()
	if err != nil {
		return "", err
	}
	return codexModelCatalogPathForConfig(configPath), nil
}

func codexModelCatalogPathForConfig(configPath string) string {
	return filepath.Join(filepath.Dir(configPath), "model.json")
}

func codexProfileConfigPath() (string, error) {
	configPath, err := codexConfigPath()
	if err != nil {
		return "", err
	}
	return codexProfileConfigPathForConfig(configPath), nil
}

func codexProfileConfigPathForConfig(configPath string) string {
	return codexNamedProfileConfigPathForConfig(configPath, codexProfileName)
}

func codexNamedProfileConfigPathForConfig(configPath, profileName string) string {
	return filepath.Join(filepath.Dir(configPath), profileName+".config.toml")
}

func cleanupCodexLegacyProfileConfig(configPath string) error {
	content, err := os.ReadFile(configPath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	text := string(content)
	parsed, err := codexParseConfig(text)
	if err != nil {
		return err
	}

	updated := text
	if profile, ok := parsed.RootStringOK(codexRootProfileKey); ok && profile == codexProfileName {
		updated = codexRemoveRootValue(updated, codexRootProfileKey)
	}
	if parsed.Exists("profiles", codexProfileName) {
		updated = codexRemoveSection(updated, codexProfileHeader())
	}
	if updated == text {
		return nil
	}
	if err := codexValidateConfigText(updated); err != nil {
		return err
	}
	return fileutil.WriteWithBackup(configPath, []byte(updated), "")
}

// writeCodexProfileConfig ensures ~/.codex/ollama-launch.config.toml selects
// the Ollama provider and catalog for CLI launches without changing root config.
func writeCodexProfileConfig(profilePath, model, modelCatalogPath string) error {
	return writeCodexNamedProfileConfig(profilePath, codexProfileName, model, modelCatalogPath, "")
}

func writeCodexNamedProfileConfig(profilePath, profileName, model, modelCatalogPath, backupSubdir string) error {
	baseURL := codexBaseURL()

	var lines []string
	if strings.TrimSpace(model) != "" {
		lines = append(lines, fmt.Sprintf("%s = %q", codexRootModelKey, model))
	}
	lines = append(lines, fmt.Sprintf("%s = %q", codexRootModelProviderKey, profileName))
	if strings.TrimSpace(modelCatalogPath) != "" {
		lines = append(lines, fmt.Sprintf("%s = %q", codexRootModelCatalogJSONKey, modelCatalogPath))
	}
	text := strings.Join(lines, "\n") + "\n\n"
	text += strings.Join([]string{
		codexProviderHeaderFor(profileName),
		fmt.Sprintf("name = %q", codexProviderName),
		fmt.Sprintf("base_url = %q", baseURL),
		`wire_api = "responses"`,
		"",
	}, "\n")

	parsed, err := codexParseConfig(text)
	if err != nil {
		return err
	}
	if err := codexValidateProfileConfigText(parsed, profileName, model, modelCatalogPath, baseURL); err != nil {
		return err
	}

	if err := os.MkdirAll(filepath.Dir(profilePath), 0o755); err != nil {
		return err
	}
	return fileutil.WriteWithBackup(profilePath, []byte(text), backupSubdir)
}

func codexBaseURL() string {
	return strings.TrimRight(envconfig.ConnectableHost().String(), "/") + "/v1/"
}

func codexProfileHeader() string {
	return codexProfileHeaderFor(codexProfileName)
}

func codexProviderHeader() string {
	return codexProviderHeaderFor(codexProfileName)
}

func codexProfileHeaderFor(profileName string) string {
	return fmt.Sprintf("[profiles.%s]", profileName)
}

func codexProviderHeaderFor(profileName string) string {
	return fmt.Sprintf("[model_providers.%s]", profileName)
}

func codexValidateProfileConfigText(config codexParsedConfig, profileName, model, modelCatalogPath, baseURL string) error {
	if config.Exists("profiles", profileName) {
		return fmt.Errorf("generated Codex config still contains legacy profiles.%s table", profileName)
	}
	for _, check := range []struct {
		path []string
		want string
	}{
		{[]string{"model_providers", profileName, "name"}, codexProviderName},
		{[]string{"model_providers", profileName, "base_url"}, baseURL},
		{[]string{"model_providers", profileName, "wire_api"}, "responses"},
	} {
		if got, ok := config.String(check.path...); !ok || got != check.want {
			return fmt.Errorf("generated Codex config missing %s = %q", strings.Join(check.path, "."), check.want)
		}
	}
	if got, ok := config.RootStringOK(codexRootProfileKey); ok {
		return fmt.Errorf("generated Codex config still contains legacy profile = %q", got)
	}
	if got := config.RootString(codexRootModelProviderKey); got != profileName {
		return fmt.Errorf("generated Codex config missing model_provider = %q", profileName)
	}
	if model != "" {
		if got := config.RootString(codexRootModelKey); got != model {
			return fmt.Errorf("generated Codex config missing model = %q", model)
		}
	}
	if modelCatalogPath != "" {
		if got := config.RootString(codexRootModelCatalogJSONKey); got != modelCatalogPath {
			return fmt.Errorf("generated Codex config missing model_catalog_json = %q", modelCatalogPath)
		}
	}
	return nil
}

func codexRemoveSection(text, header string) string {
	targetPath, ok := codexTableHeaderPath(header)
	if !ok {
		return text
	}
	start, end, found := codexSectionRange(text, targetPath)
	if !found {
		return text
	}
	return text[:start] + text[end:]
}

type codexParsedConfig struct {
	values map[string]any
}

func (c codexParsedConfig) Int(path ...string) (int, bool) {
	if len(path) == 0 {
		return 0, false
	}
	var current any = c.values
	for _, part := range path {
		table, ok := current.(map[string]any)
		if !ok {
			return 0, false
		}
		current, ok = table[part]
		if !ok {
			return 0, false
		}
	}
	switch value := current.(type) {
	case int64:
		return int(value), value > 0
	case int:
		return value, value > 0
	default:
		return 0, false
	}
}

func (c codexParsedConfig) String(path ...string) (string, bool) {
	if len(path) == 0 {
		return "", false
	}
	var current any = c.values
	for _, part := range path {
		table, ok := current.(map[string]any)
		if !ok {
			return "", false
		}
		current, ok = table[part]
		if !ok {
			return "", false
		}
	}
	value, ok := current.(string)
	if !ok {
		return "", false
	}
	return value, true
}

func (c codexParsedConfig) Exists(path ...string) bool {
	if len(path) == 0 {
		return false
	}
	var current any = c.values
	for _, part := range path {
		table, ok := current.(map[string]any)
		if !ok {
			return false
		}
		current, ok = table[part]
		if !ok {
			return false
		}
	}
	return true
}

func (c codexParsedConfig) RootString(key string) string {
	value, _ := c.RootStringOK(key)
	return value
}

func (c codexParsedConfig) RootStringOK(key string) (string, bool) {
	return c.String(key)
}

func (c codexParsedConfig) ProfileString(profileName, key string) string {
	value, _ := c.String("profiles", profileName, key)
	return value
}

func (c codexParsedConfig) ProviderString(profileName, key string) string {
	value, _ := c.String("model_providers", profileName, key)
	return value
}

func codexRootStringValue(text, key string) string {
	config, err := codexParseConfig(text)
	if err != nil {
		return ""
	}
	return config.RootString(key)
}

func codexRootStringValueOK(text, key string) (string, bool) {
	config, err := codexParseConfig(text)
	if err != nil {
		return "", false
	}
	return config.RootStringOK(key)
}

func codexStringValue(text string, path ...string) (string, bool) {
	config, err := codexParseConfig(text)
	if err != nil {
		return "", false
	}
	return config.String(path...)
}

func codexSectionStringValue(text, header, key string) string {
	path, ok := codexTableHeaderPath(header)
	if !ok {
		return ""
	}
	value, _ := codexStringValue(text, append(path, key)...)
	return value
}

func codexParseConfig(text string) (codexParsedConfig, error) {
	values, err := codexParseConfigText(text)
	if err != nil {
		return codexParsedConfig{}, err
	}
	return codexParsedConfig{values: values}, nil
}

func codexParseConfigText(text string) (map[string]any, error) {
	cfg := map[string]any{}
	if strings.TrimSpace(text) == "" {
		return cfg, nil
	}
	if err := toml.Unmarshal([]byte(text), &cfg); err != nil {
		return nil, fmt.Errorf("invalid Codex config TOML: %w", err)
	}
	return cfg, nil
}

func codexValidateConfigText(text string) error {
	_, err := codexParseConfig(text)
	return err
}

func codexSectionRange(text string, targetPath []string) (int, int, bool) {
	lines := strings.SplitAfter(text, "\n")
	offset := 0
	start := -1
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if !strings.HasPrefix(trimmed, "[") || strings.HasPrefix(trimmed, "#") {
			offset += len(line)
			continue
		}
		if start >= 0 {
			return start, offset, true
		}
		if path, ok := codexTableHeaderPath(trimmed); ok && codexSamePath(path, targetPath) {
			start = offset
		}
		offset += len(line)
	}
	if start >= 0 {
		return start, len(text), true
	}
	return 0, 0, false
}

func codexTableHeaderPath(header string) ([]string, bool) {
	trimmed := strings.TrimSpace(header)
	if !strings.HasPrefix(trimmed, "[") || strings.HasPrefix(trimmed, "[[") {
		return nil, false
	}

	const probeKey = "__ollama_launch_probe"
	cfg := map[string]any{}
	if err := toml.Unmarshal([]byte(trimmed+"\n"+probeKey+" = true\n"), &cfg); err != nil {
		return nil, false
	}
	return codexFindProbePath(cfg, probeKey, nil)
}

func codexFindProbePath(value any, probeKey string, path []string) ([]string, bool) {
	table, ok := value.(map[string]any)
	if !ok {
		return nil, false
	}
	if probe, ok := table[probeKey].(bool); ok && probe {
		return path, true
	}
	for key, child := range table {
		if key == probeKey {
			continue
		}
		if childPath, ok := codexFindProbePath(child, probeKey, append(path, key)); ok {
			return childPath, true
		}
	}
	return nil, false
}

func codexSamePath(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func codexSetRootStringValue(text, key, value string) string {
	lines := strings.SplitAfter(text, "\n")
	rootEnd := len(lines)
	for i, line := range lines {
		if strings.HasPrefix(strings.TrimSpace(line), "[") {
			rootEnd = i
			break
		}
	}

	assignment := fmt.Sprintf("%s = %q", key, value)
	for i := range rootEnd {
		line := lines[i]
		trimmed := strings.TrimSpace(line)
		if trimmed == "" || strings.HasPrefix(trimmed, "#") {
			continue
		}
		if codexRootLineHasKey(trimmed, key) {
			if strings.HasSuffix(line, "\n") {
				lines[i] = assignment + "\n"
			} else {
				lines[i] = assignment
			}
			return strings.Join(lines, "")
		}
	}

	insert := assignment + "\n"
	root := strings.Join(lines[:rootEnd], "")
	rest := strings.Join(lines[rootEnd:], "")
	if root != "" && !strings.HasSuffix(root, "\n") {
		root += "\n"
	}
	if rest != "" && !strings.HasSuffix(insert, "\n\n") {
		insert += "\n"
	}
	return root + insert + rest
}

func codexRemoveRootValue(text, key string) string {
	lines := strings.SplitAfter(text, "\n")
	rootEnd := len(lines)
	for i, line := range lines {
		if strings.HasPrefix(strings.TrimSpace(line), "[") {
			rootEnd = i
			break
		}
	}

	out := make([]string, 0, len(lines))
	for i, line := range lines {
		if i < rootEnd {
			trimmed := strings.TrimSpace(line)
			if trimmed != "" && !strings.HasPrefix(trimmed, "#") && codexRootLineHasKey(trimmed, key) {
				continue
			}
		}
		out = append(out, line)
	}
	return strings.Join(out, "")
}

func codexRootLineHasKey(line, key string) bool {
	cfg := map[string]any{}
	if err := toml.Unmarshal([]byte(line+"\n"), &cfg); err != nil {
		return false
	}
	_, ok := cfg[key]
	return ok
}

func codexCatalogModel(modelName string, models []LaunchModel) LaunchModel {
	if model, ok := findLaunchModel(models, modelName); ok {
		model.Name = modelName
		return model.WithCloudLimits()
	}
	return fallbackLaunchModel(modelName)
}

func writeCodexModelCatalog(catalogPath string, model LaunchModel) error {
	entry := buildCodexModelEntry(model)

	catalog := map[string]any{
		"models": []any{entry},
	}

	data, err := json.MarshalIndent(catalog, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(catalogPath, data, 0o644)
}

func buildCodexModelEntry(launchModel LaunchModel) map[string]any {
	modelName := launchModel.Name
	contextWindow := codexFallbackContextWindow
	systemPrompt := ""

	if launchModel.ContextLength > 0 {
		contextWindow = launchModel.ContextLength
	} else if launchModel.Details.ContextLength > 0 {
		contextWindow = launchModel.Details.ContextLength
	}
	if l, ok := lookupCloudModelLimit(modelName); ok {
		contextWindow = l.Context
	}

	modalities := []string{"text"}
	if launchModel.HasCapability(model.CapabilityVision) {
		modalities = append(modalities, "image")
	}

	truncationMode := "bytes"
	if isCloudModelName(modelName) {
		truncationMode = "tokens"
	}

	return map[string]any{
		"slug":                         modelName,
		"display_name":                 modelName,
		"context_window":               contextWindow,
		"shell_type":                   "default",
		"visibility":                   "list",
		"supported_in_api":             true,
		"priority":                     0,
		"truncation_policy":            map[string]any{"mode": truncationMode, "limit": 10000},
		"input_modalities":             modalities,
		"base_instructions":            systemPrompt,
		"support_verbosity":            true,
		"default_verbosity":            "low",
		"supports_parallel_tool_calls": false,
		"supports_reasoning_summaries": false,
		"supported_reasoning_levels":   []any{},
		"experimental_supported_tools": []any{},
	}
}

func checkCodexVersion() error {
	if _, err := exec.LookPath("codex"); err != nil {
		return fmt.Errorf("codex is not installed, install with: npm install -g @openai/codex")
	}

	out, err := exec.Command("codex", "--version").Output()
	if err != nil {
		return fmt.Errorf("failed to get codex version: %w", err)
	}

	// Parse output like "codex-cli 0.87.0"
	fields := strings.Fields(strings.TrimSpace(string(out)))
	if len(fields) < 2 {
		return fmt.Errorf("unexpected codex version output: %s", string(out))
	}

	version := "v" + fields[len(fields)-1]
	minVersion := "v0.134.0"

	if semver.Compare(version, minVersion) < 0 {
		return fmt.Errorf("codex version %s is too old, minimum required is %s, update with: npm update -g @openai/codex", fields[len(fields)-1], "0.134.0")
	}

	return nil
}
