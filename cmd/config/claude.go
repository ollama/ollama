package config

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
)

// Claude implements Runner and AliasConfigurer for Claude Code integration
type Claude struct{}

// Compile-time check that Claude implements AliasConfigurer
var _ AliasConfigurer = (*Claude)(nil)

func (c *Claude) String() string { return "Claude Code" }

func (c *Claude) args(model string, extra []string) []string {
	var args []string
	if model != "" {
		args = append(args, "--model", model)
	}
	args = append(args, extra...)
	return args
}

func (c *Claude) findPath() (string, error) {
	if p, err := exec.LookPath("claude"); err == nil {
		return p, nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	name := "claude"
	if runtime.GOOS == "windows" {
		name = "claude.exe"
	}
	fallback := filepath.Join(home, ".claude", "local", name)
	if _, err := os.Stat(fallback); err != nil {
		return "", err
	}
	return fallback, nil
}

func (c *Claude) Run(model string, args []string) error {
	claudePath, err := c.findPath()
	if err != nil {
		return fmt.Errorf("claude is not installed, install from https://code.claude.com/docs/en/quickstart")
	}

	cmd := exec.Command(claudePath, c.args(model, args)...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	env := append(os.Environ(),
		"ANTHROPIC_BASE_URL="+envconfig.Host().String(),
		"ANTHROPIC_API_KEY=",
		"ANTHROPIC_AUTH_TOKEN=ollama",
	)

	env = append(env, c.modelEnvVars(model)...)

	cmd.Env = env
	return cmd.Run()
}

// modelEnvVars returns Claude Code env vars that route all model tiers through Ollama.
func (c *Claude) modelEnvVars(model string) []string {
	primary := model
	fast := model
	if cfg, err := loadIntegration("claude"); err == nil && cfg.Aliases != nil {
		if p := cfg.Aliases["primary"]; p != "" {
			primary = p
		}
		if f := cfg.Aliases["fast"]; f != "" {
			fast = f
		}
	}
	return []string{
		"ANTHROPIC_DEFAULT_OPUS_MODEL=" + primary,
		"ANTHROPIC_DEFAULT_SONNET_MODEL=" + primary,
		"ANTHROPIC_DEFAULT_HAIKU_MODEL=" + fast,
		"CLAUDE_CODE_SUBAGENT_MODEL=" + primary,
	}
}

// ConfigureAliases sets up model aliases for Claude Code.
// model: the model to use (if empty, user will be prompted to select)
// aliases: existing alias configuration to preserve/update
// Cloud-only: subagent routing (fast model) is gated to cloud models only until
// there is a better strategy for prompt caching on local models.
func (c *Claude) ConfigureAliases(ctx context.Context, model string, existingAliases map[string]string, force bool) (map[string]string, bool, error) {
	aliases := make(map[string]string)
	for k, v := range existingAliases {
		aliases[k] = v
	}

	if model != "" {
		aliases["primary"] = model
	}

	if !force && aliases["primary"] != "" {
		client, _ := api.ClientFromEnvironment()
		if isCloudModel(ctx, client, aliases["primary"]) {
			if isCloudModel(ctx, client, aliases["fast"]) {
				return aliases, false, nil
			}
		} else {
			delete(aliases, "fast")
			return aliases, false, nil
		}
	}

	items, existingModels, cloudModels, client, err := listModels(ctx)
	if err != nil {
		return nil, false, err
	}

	fmt.Fprintf(os.Stderr, "\n%sModel Configuration%s\n\n", ansiBold, ansiReset)

	if aliases["primary"] == "" || force {
		primary, err := DefaultSingleSelector("Select model:", items)
		if err != nil {
			return nil, false, err
		}
		if err := pullIfNeeded(ctx, client, existingModels, primary); err != nil {
			return nil, false, err
		}
		if err := ensureAuth(ctx, client, cloudModels, []string{primary}); err != nil {
			return nil, false, err
		}
		aliases["primary"] = primary
	}

	if isCloudModel(ctx, client, aliases["primary"]) {
		if aliases["fast"] == "" || !isCloudModel(ctx, client, aliases["fast"]) {
			aliases["fast"] = aliases["primary"]
		}
	} else {
		delete(aliases, "fast")
	}

	return aliases, true, nil
}

// SetAliases syncs the configured aliases to the Ollama server using prefix matching.
// Cloud-only: for local models (fast is empty), we delete any existing aliases to
// prevent stale routing to a previous cloud model.
func (c *Claude) SetAliases(ctx context.Context, aliases map[string]string) error {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	prefixes := []string{"claude-sonnet-", "claude-haiku-"}

	if aliases["fast"] == "" {
		for _, prefix := range prefixes {
			_ = client.DeleteAliasExperimental(ctx, &api.AliasDeleteRequest{Alias: prefix})
		}
		return nil
	}

	prefixAliases := map[string]string{
		"claude-sonnet-": aliases["primary"],
		"claude-haiku-":  aliases["fast"],
	}

	var errs []string
	for prefix, target := range prefixAliases {
		req := &api.AliasRequest{
			Alias:          prefix,
			Target:         target,
			PrefixMatching: true,
		}
		if err := client.SetAliasExperimental(ctx, req); err != nil {
			errs = append(errs, prefix)
		}
	}

	if len(errs) > 0 {
		return fmt.Errorf("failed to set aliases: %v", errs)
	}
	return nil
}
