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
	cmd.Env = append(os.Environ(),
		"ANTHROPIC_BASE_URL="+envconfig.Host().String(),
		"ANTHROPIC_API_KEY=",
		"ANTHROPIC_AUTH_TOKEN=ollama",
	)
	return cmd.Run()
}

// ConfigureAliases sets up Primary and Fast model aliases for Claude Code.
func (c *Claude) ConfigureAliases(ctx context.Context, primaryModel string, existing map[string]string, force bool) (map[string]string, bool, error) {
	aliases := make(map[string]string)
	for k, v := range existing {
		aliases[k] = v
	}

	if primaryModel != "" {
		aliases["primary"] = primaryModel
	}

	if !force && aliases["primary"] != "" && aliases["fast"] != "" {
		return aliases, false, nil
	}

	items, existingModels, cloudModels, client, err := listModels(ctx)
	if err != nil {
		return nil, false, err
	}

	fmt.Fprintf(os.Stderr, "\n%sModel Configuration%s\n", ansiBold, ansiReset)
	fmt.Fprintf(os.Stderr, "%sClaude Code uses multiple models for various tasks%s\n\n", ansiGray, ansiReset)

	fmt.Fprintf(os.Stderr, "%sPrimary%s\n", ansiBold, ansiReset)
	fmt.Fprintf(os.Stderr, "%sHandles complex reasoning: planning, code generation, debugging.%s\n\n", ansiGray, ansiReset)

	if aliases["primary"] == "" || force {
		primary, err := selectPrompt("Select Primary model:", items)
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
	} else {
		fmt.Fprintf(os.Stderr, "  %s\n\n", aliases["primary"])
	}

	fmt.Fprintf(os.Stderr, "%sFast%s\n", ansiBold, ansiReset)
	fmt.Fprintf(os.Stderr, "%sHandles quick operations: file searches, simple edits, status checks.%s\n", ansiGray, ansiReset)
	fmt.Fprintf(os.Stderr, "%sSmaller models work well and respond faster.%s\n\n", ansiGray, ansiReset)

	if aliases["fast"] == "" || force {
		fast, err := selectPrompt("Select Fast model:", items)
		if err != nil {
			return nil, false, err
		}
		if err := pullIfNeeded(ctx, client, existingModels, fast); err != nil {
			return nil, false, err
		}
		if err := ensureAuth(ctx, client, cloudModels, []string{fast}); err != nil {
			return nil, false, err
		}
		aliases["fast"] = fast
	}

	return aliases, true, nil
}

// SetAliases syncs the configured aliases to the Ollama server using prefix matching.
func (c *Claude) SetAliases(ctx context.Context, aliases map[string]string) error {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
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
		if err := client.SetAlias(ctx, req); err != nil {
			errs = append(errs, prefix)
		}
	}

	if len(errs) > 0 {
		return fmt.Errorf("failed to set aliases: %v", errs)
	}
	return nil
}
