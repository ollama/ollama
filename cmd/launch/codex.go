package launch

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/ollama/ollama/cmd/internal/fileutil"
	"github.com/ollama/ollama/envconfig"
	"github.com/pelletier/go-toml/v2"
	"golang.org/x/mod/semver"
)

// Codex implements Runner for Codex integration
type Codex struct{}

func (c *Codex) String() string { return "Codex" }

const (
	codexProfileName  = "ollama-launch"
	codexProviderName = "Ollama"

	codexRootProfileKey          = "profile"
	codexRootModelKey            = "model"
	codexRootModelProviderKey    = "model_provider"
	codexRootModelCatalogJSONKey = "model_catalog_json"
)

func (c *Codex) args(model string, extra []string) []string {
	args := []string{"--profile", codexProfileName}
	if model != "" {
		args = append(args, "-m", model)
	}
	args = append(args, extra...)
	return args
}

func (c *Codex) Run(model string, args []string) error {
	if err := checkCodexVersion(); err != nil {
		return err
	}

	if err := ensureCodexConfig(); err != nil {
		return fmt.Errorf("failed to configure codex: %w", err)
	}

	cmd := exec.Command("codex", c.args(model, args)...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Env = append(os.Environ(),
		"OPENAI_API_KEY=ollama",
	)
	return cmd.Run()
}

// ensureCodexConfig writes a [profiles.ollama-launch] section to ~/.codex/config.toml
// with openai_base_url pointing to the local Ollama server.
func ensureCodexConfig() error {
	configPath, err := codexConfigPath()
	if err != nil {
		return err
	}

	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		return err
	}
	return writeCodexProfile(configPath)
}

func codexConfigPath() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".codex", "config.toml"), nil
}

// writeCodexProfile ensures ~/.codex/config.toml has the ollama-launch profile
// and model provider sections with the correct base URL.
func writeCodexProfile(configPath string) error {
	return writeCodexLaunchProfile(configPath, codexLaunchProfileOptions{
		forceAPIAuth: true,
	})
}

type codexLaunchProfileOptions struct {
	activate           bool
	profileName        string
	forceAPIAuth       bool
	setRootModelConfig bool
	model              string
	modelCatalogPath   string
	backupIntegration  string
}

func writeCodexLaunchProfile(configPath string, opts codexLaunchProfileOptions) error {
	baseURL := codexBaseURL()
	profileName := codexLaunchProfileName(opts)
	profileHeader := codexProfileHeaderFor(profileName)
	providerHeader := codexProviderHeaderFor(profileName)

	content, readErr := os.ReadFile(configPath)
	text := ""
	if readErr == nil {
		text = string(content)
	} else if !os.IsNotExist(readErr) {
		return readErr
	}
	parsed, err := codexParseConfig(text)
	if err != nil {
		return err
	}

	model := strings.TrimSpace(opts.model)
	if model == "" {
		model = parsed.ProfileString(profileName, codexRootModelKey)
	}
	modelCatalogPath := strings.TrimSpace(opts.modelCatalogPath)
	if modelCatalogPath == "" {
		modelCatalogPath = parsed.ProfileString(profileName, codexRootModelCatalogJSONKey)
	}

	profileLines := []string{}
	if model != "" {
		profileLines = append(profileLines, fmt.Sprintf("%s = %q", codexRootModelKey, model))
	}
	profileLines = append(profileLines,
		fmt.Sprintf("openai_base_url = %q", baseURL),
		fmt.Sprintf("%s = %q", codexRootModelProviderKey, profileName),
	)
	if opts.forceAPIAuth {
		profileLines = append(profileLines, `forced_login_method = "api"`)
	}
	if modelCatalogPath != "" {
		profileLines = append(profileLines, fmt.Sprintf("%s = %q", codexRootModelCatalogJSONKey, modelCatalogPath))
	}

	sections := []struct {
		header string
		lines  []string
	}{
		{
			header: profileHeader,
			lines:  profileLines,
		},
		{
			header: providerHeader,
			lines: []string{
				fmt.Sprintf("name = %q", codexProviderName),
				fmt.Sprintf("base_url = %q", baseURL),
				`wire_api = "responses"`,
			},
		},
	}

	if opts.activate {
		text = codexSetRootStringValue(text, codexRootProfileKey, profileName)
	}
	if opts.setRootModelConfig {
		if model != "" {
			text = codexSetRootStringValue(text, codexRootModelKey, model)
		}
		text = codexSetRootStringValue(text, codexRootModelProviderKey, profileName)
		if modelCatalogPath != "" {
			text = codexSetRootStringValue(text, codexRootModelCatalogJSONKey, modelCatalogPath)
		}
	}

	for _, s := range sections {
		text = codexUpsertSection(text, s.header, s.lines)
	}
	parsed, err = codexParseConfig(text)
	if err != nil {
		return err
	}
	if err := codexValidateLaunchProfileText(parsed, profileName, opts, model, modelCatalogPath, baseURL); err != nil {
		return err
	}

	if err := os.MkdirAll(filepath.Dir(configPath), 0o755); err != nil {
		return err
	}
	return fileutil.WriteWithBackup(configPath, []byte(text), opts.backupIntegration)
}

func codexLaunchProfileName(opts codexLaunchProfileOptions) string {
	if name := strings.TrimSpace(opts.profileName); name != "" {
		return name
	}
	return codexProfileName
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

func codexValidateLaunchProfileText(config codexParsedConfig, profileName string, opts codexLaunchProfileOptions, model, modelCatalogPath, baseURL string) error {
	for _, check := range []struct {
		path []string
		want string
	}{
		{[]string{"profiles", profileName, "openai_base_url"}, baseURL},
		{[]string{"profiles", profileName, codexRootModelProviderKey}, profileName},
		{[]string{"model_providers", profileName, "name"}, codexProviderName},
		{[]string{"model_providers", profileName, "base_url"}, baseURL},
		{[]string{"model_providers", profileName, "wire_api"}, "responses"},
	} {
		if got, ok := config.String(check.path...); !ok || got != check.want {
			return fmt.Errorf("generated Codex config missing %s = %q", strings.Join(check.path, "."), check.want)
		}
	}
	if opts.forceAPIAuth {
		if got, ok := config.String("profiles", profileName, "forced_login_method"); !ok || got != "api" {
			return fmt.Errorf("generated Codex config missing profiles.%s.forced_login_method = %q", profileName, "api")
		}
	}
	if model != "" {
		if got, ok := config.String("profiles", profileName, codexRootModelKey); !ok || got != model {
			return fmt.Errorf("generated Codex config missing profiles.%s.model = %q", profileName, model)
		}
	}
	if modelCatalogPath != "" {
		if got, ok := config.String("profiles", profileName, codexRootModelCatalogJSONKey); !ok || got != modelCatalogPath {
			return fmt.Errorf("generated Codex config missing profiles.%s.model_catalog_json = %q", profileName, modelCatalogPath)
		}
	}
	if opts.activate {
		if got := config.RootString(codexRootProfileKey); got != profileName {
			return fmt.Errorf("generated Codex config missing profile = %q", profileName)
		}
	}
	if opts.setRootModelConfig {
		if model != "" {
			if got := config.RootString(codexRootModelKey); got != model {
				return fmt.Errorf("generated Codex config missing model = %q", model)
			}
		}
		if got := config.RootString(codexRootModelProviderKey); got != profileName {
			return fmt.Errorf("generated Codex config missing model_provider = %q", profileName)
		}
		if modelCatalogPath != "" {
			if got := config.RootString(codexRootModelCatalogJSONKey); got != modelCatalogPath {
				return fmt.Errorf("generated Codex config missing model_catalog_json = %q", modelCatalogPath)
			}
		}
	}
	return nil
}

func codexUpsertSection(text, header string, lines []string) string {
	block := strings.Join(append([]string{header}, lines...), "\n") + "\n"

	if targetPath, ok := codexTableHeaderPath(header); ok {
		if start, end, found := codexSectionRange(text, targetPath); found {
			return text[:start] + block + text[end:]
		}
	}

	if text != "" && !strings.HasSuffix(text, "\n") {
		text += "\n"
	}
	if text != "" {
		text += "\n"
	}
	return text + block
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
	minVersion := "v0.81.0"

	if semver.Compare(version, minVersion) < 0 {
		return fmt.Errorf("codex version %s is too old, minimum required is %s, update with: npm update -g @openai/codex", fields[len(fields)-1], "0.81.0")
	}

	return nil
}
