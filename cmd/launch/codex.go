package launch

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/types/model"
	"golang.org/x/mod/semver"
)

// Codex implements Runner for Codex integration
type Codex struct{}

func (c *Codex) String() string { return "Codex" }

const codexProfileName = "ollama-launch"
const codexCatalogFileName = "ollama-launch-models.json"

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

	if err := ensureCodexConfig(model); err != nil {
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

// ensureCodexConfig writes a Codex profile and model catalog so Codex uses the
// local Ollama server and has model metadata available.
func ensureCodexConfig(modelName string) error {
	home, err := os.UserHomeDir()
	if err != nil {
		return err
	}

	codexDir := filepath.Join(home, ".codex")
	if err := os.MkdirAll(codexDir, 0o755); err != nil {
		return err
	}

	catalogPath := filepath.Join(codexDir, codexCatalogFileName)
	if err := writeCodexModelCatalog(catalogPath, modelName); err != nil {
		return err
	}

	configPath := filepath.Join(codexDir, "config.toml")
	return writeCodexProfile(configPath, catalogPath)
}

// writeCodexProfile ensures ~/.codex/config.toml has the ollama-launch profile
// and model provider sections with the correct base URL.
func writeCodexProfile(configPath, catalogPath string) error {
	baseURL := envconfig.Host().String() + "/v1/"

	sections := []struct {
		header string
		lines  []string
	}{
		{
			header: fmt.Sprintf("[profiles.%s]", codexProfileName),
			lines: []string{
				fmt.Sprintf("openai_base_url = %q", baseURL),
				`forced_login_method = "api"`,
				fmt.Sprintf("model_provider = %q", codexProfileName),
				fmt.Sprintf("model_catalog_json = %q", catalogPath),
			},
		},
		{
			header: fmt.Sprintf("[model_providers.%s]", codexProfileName),
			lines: []string{
				`name = "Ollama"`,
				fmt.Sprintf("base_url = %q", baseURL),
			},
		},
	}

	content, readErr := os.ReadFile(configPath)
	text := ""
	if readErr == nil {
		text = string(content)
	}

	for _, s := range sections {
		block := strings.Join(append([]string{s.header}, s.lines...), "\n") + "\n"

		if idx := strings.Index(text, s.header); idx >= 0 {
			// Replace the existing section up to the next section header.
			rest := text[idx+len(s.header):]
			if endIdx := strings.Index(rest, "\n["); endIdx >= 0 {
				text = text[:idx] + block + rest[endIdx+1:]
			} else {
				text = text[:idx] + block
			}
		} else {
			// Append the section.
			if text != "" && !strings.HasSuffix(text, "\n") {
				text += "\n"
			}
			if text != "" {
				text += "\n"
			}
			text += block
		}
	}

	return os.WriteFile(configPath, []byte(text), 0o644)
}

func writeCodexModelCatalog(catalogPath, modelName string) error {
	entry := buildCodexModelEntry(modelName)

	catalog := map[string]any{
		"models": []any{entry},
	}

	data, err := json.MarshalIndent(catalog, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(catalogPath, data, 0o644)
}

func buildCodexModelEntry(modelName string) map[string]any {
	contextWindow := 0
	hasVision := false
	hasThinking := false
	systemPrompt := ""

	if l, ok := lookupCloudModelLimit(modelName); ok {
		contextWindow = l.Context
	}

	client := api.NewClient(envconfig.Host(), http.DefaultClient)
	resp, err := client.Show(context.Background(), &api.ShowRequest{Model: modelName})
	if err == nil {
		systemPrompt = resp.System
		if slices.Contains(resp.Capabilities, model.CapabilityVision) {
			hasVision = true
		}
		if slices.Contains(resp.Capabilities, model.CapabilityThinking) {
			hasThinking = true
		}

		if !isCloudModelName(modelName) {
			if n, ok := modelInfoContextLength(resp.ModelInfo); ok {
				contextWindow = n
			}
			if resp.Details.Format != "safetensors" {
				if ctxLen := envconfig.ContextLength(); ctxLen > 0 {
					contextWindow = int(ctxLen)
				}
				if numCtx := parseNumCtx(resp.Parameters); numCtx > 0 {
					contextWindow = numCtx
				}
			}
		}
	}

	modalities := []string{"text"}
	if hasVision {
		modalities = append(modalities, "image")
	}

	reasoningLevels := []any{}
	if hasThinking {
		reasoningLevels = []any{
			map[string]any{"effort": "low", "description": "Fast responses with lighter reasoning"},
			map[string]any{"effort": "medium", "description": "Balances speed and reasoning depth"},
			map[string]any{"effort": "high", "description": "Greater reasoning depth for complex problems"},
		}
	}

	truncationMode := "bytes"
	if isCloudModelName(modelName) {
		truncationMode = "tokens"
	}

	return map[string]any{
		"slug":                         modelName,
		"display_name":                 modelName,
		"context_window":               contextWindow,
		"apply_patch_tool_type":        "function",
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
		"supports_reasoning_summaries": hasThinking,
		"supported_reasoning_levels":   reasoningLevels,
		"experimental_supported_tools": []any{},
	}
}

func parseNumCtx(parameters string) int {
	for _, line := range strings.Split(parameters, "\n") {
		fields := strings.Fields(line)
		if len(fields) == 2 && fields[0] == "num_ctx" {
			if v, err := strconv.ParseFloat(fields[1], 64); err == nil {
				return int(v)
			}
		}
	}

	return 0
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
