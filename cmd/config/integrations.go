package config

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"os"
	"os/exec"
	"slices"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/spf13/cobra"
)

type envVar struct {
	Name  string
	Value string
}

type integration struct {
	Name             string
	DisplayName      string
	Command          string
	EnvVars          func(model string) []envVar
	Args             func(model string) []string
	Setup            func(models []string) error
	CheckInstall     func() error
	ConfigPaths      func() []string // paths that will be modified
	ConfiguredModels func() []string // models already configured
}

// checkCommand returns an error if the command is not installed.
func checkCommand(cmd, installInstructions string) func() error {
	return func() error {
		if _, err := exec.LookPath(cmd); err != nil {
			return fmt.Errorf("%s is not installed, %s", cmd, installInstructions)
		}
		return nil
	}
}

var integrations = map[string]*integration{
	"claude":   claudeIntegration,
	"codex":    codexIntegration,
	"droid":    droidIntegration,
	"opencode": openCodeIntegration,
}

func selectIntegration() (string, error) {
	if len(integrations) == 0 {
		return "", fmt.Errorf("no integrations available")
	}

	names := slices.Sorted(maps.Keys(integrations))
	var items []selectItem
	for _, name := range names {
		integ := integrations[name]
		description := integ.DisplayName
		if conn, err := loadIntegration(name); err == nil && conn.defaultModel() != "" {
			description = fmt.Sprintf("%s (%s)", integ.DisplayName, conn.defaultModel())
		}
		items = append(items, selectItem{Name: integ.Name, Description: description})
	}

	return selectPrompt("Select integration:", items)
}

func selectModels(ctx context.Context, name, current string) ([]string, error) {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return nil, err
	}

	models, err := client.List(ctx)
	if err != nil {
		return nil, err
	}

	if len(models.Models) == 0 {
		return nil, fmt.Errorf("no models available, run 'ollama pull <model>' first")
	}

	var items []selectItem
	cloudModels := make(map[string]bool)
	for _, m := range models.Models {
		if m.RemoteModel != "" {
			cloudModels[m.Name] = true
		}
		items = append(items, selectItem{Name: m.Name})
	}

	if len(items) == 0 {
		return nil, fmt.Errorf("no local models available, run 'ollama pull <model>' first")
	}

	integ := integrations[strings.ToLower(name)]

	// Get previously configured models (saved config takes precedence)
	var preChecked []string
	if saved, err := loadIntegration(name); err == nil {
		preChecked = saved.Models
	} else if integ != nil && integ.ConfiguredModels != nil {
		preChecked = integ.ConfiguredModels()
	}
	checked := make(map[string]bool, len(preChecked))
	for _, n := range preChecked {
		checked[n] = true
	}

	// Resolve current to full name (e.g., "llama3.2" -> "llama3.2:latest")
	for _, item := range items {
		if item.Name == current || strings.HasPrefix(item.Name, current+":") {
			current = item.Name
			break
		}
	}

	// If current model is configured, move to front of preChecked
	if checked[current] {
		preChecked = append([]string{current}, slices.DeleteFunc(preChecked, func(m string) bool { return m == current })...)
	}

	// Sort: checked first, then alphabetical
	slices.SortFunc(items, func(a, b selectItem) int {
		ac, bc := checked[a.Name], checked[b.Name]
		if ac != bc {
			if ac {
				return -1
			}
			return 1
		}
		return strings.Compare(strings.ToLower(a.Name), strings.ToLower(b.Name))
	})

	supportsMultiModel := integ != nil && integ.Setup != nil

	var selected []string
	if supportsMultiModel {
		selected, err = multiSelectPrompt(fmt.Sprintf("Select models for %s:", integ.DisplayName), items, preChecked)
		if err != nil {
			return nil, err
		}
	} else {
		model, err := selectPrompt(fmt.Sprintf("Select model for %s:", integ.DisplayName), items)
		if err != nil {
			return nil, err
		}
		selected = []string{model}
	}

	for _, model := range selected {
		if cloudModels[model] {
			if err := ensureSignedIn(ctx, client); err != nil {
				return nil, err
			}
			break
		}
	}

	return selected, nil
}

func ensureSignedIn(ctx context.Context, client *api.Client) error {
	user, err := client.Whoami(ctx)
	if err == nil && user != nil && user.Name != "" {
		return nil
	}

	var aErr api.AuthorizationError
	if !errors.As(err, &aErr) || aErr.SigninURL == "" {
		return err
	}

	yes, err := confirmPrompt("Sign in to ollama.com?")
	if err != nil || !yes {
		return fmt.Errorf("sign in required for cloud models")
	}

	fmt.Fprintf(os.Stderr, "\nTo sign in, navigate to:\n    %s\n\n", aErr.SigninURL)
	fmt.Fprintf(os.Stderr, "\033[90mwaiting for sign in to complete...\033[0m")

	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			fmt.Fprintf(os.Stderr, "\n")
			return ctx.Err()
		case <-ticker.C:
			u, err := client.Whoami(ctx)
			if err == nil && u != nil && u.Name != "" {
				fmt.Fprintf(os.Stderr, "\r\033[K\033[A\r\033[K\033[1msigned in:\033[0m %s\n", u.Name)
				return nil
			}
			fmt.Fprintf(os.Stderr, ".")
		}
	}
}

func hasLocalModel(models []string) bool {
	return slices.ContainsFunc(models, func(m string) bool {
		return !strings.Contains(m, "cloud")
	})
}

func printModelsAdded(integ *integration, models []string) {
	if integ.Setup != nil {
		if len(models) == 1 {
			fmt.Fprintf(os.Stderr, "Added %s to %s\n", models[0], integ.DisplayName)
		} else {
			fmt.Fprintf(os.Stderr, "Added %d models to %s (default: %s)\n", len(models), integ.DisplayName, models[0])
		}
	}

	if hasLocalModel(models) {
		fmt.Fprintln(os.Stderr)
		fmt.Fprintln(os.Stderr, "Coding agents work best with at least 64k context. Either:")
		fmt.Fprintln(os.Stderr, "  - Set the context slider in Ollama app settings")
		fmt.Fprintln(os.Stderr, "  - Run: OLLAMA_CONTEXT_LENGTH=64000 ollama serve")
	}
}

func runIntegration(name, modelName string) error {
	integ, ok := integrations[strings.ToLower(name)]
	if !ok {
		return fmt.Errorf("unknown integration: %s", name)
	}

	if err := integ.CheckInstall(); err != nil {
		return err
	}

	if integ.Setup != nil {
		models := []string{modelName}
		if config, err := loadIntegration(name); err == nil && len(config.Models) > 0 {
			models = config.Models
		}
		if err := integ.Setup(models); err != nil {
			return fmt.Errorf("setup failed: %w", err)
		}
	}

	proc := exec.Command(integ.Command, integ.Args(modelName)...)
	proc.Stdin = os.Stdin
	proc.Stdout = os.Stdout
	proc.Stderr = os.Stderr
	proc.Env = os.Environ()
	for _, env := range integ.EnvVars(modelName) {
		proc.Env = append(proc.Env, fmt.Sprintf("%s=%s", env.Name, env.Value))
	}

	fmt.Fprintf(os.Stderr, "\nLaunching %s with %s...\n", integ.DisplayName, modelName)
	return proc.Run()
}

func handleCancelled(err error) (bool, error) {
	if errors.Is(err, errCancelled) {
		return true, nil
	}
	return false, err
}

// ConfigCmd returns the cobra command for configuring integrations.
func ConfigCmd(checkServerHeartbeat func(cmd *cobra.Command, args []string) error) *cobra.Command {
	var modelFlag string
	var launchFlag bool

	cmd := &cobra.Command{
		Use:   "config [INTEGRATION]",
		Short: "Configure an external integration to use Ollama",
		Long: `Configure an external application to use Ollama models.

Supported integrations:
  claude    Claude Code
  codex     Codex
  droid     Droid
  opencode  OpenCode

Examples:
  ollama config
  ollama config claude
  ollama config droid --launch`,
		Args:    cobra.MaximumNArgs(1),
		PreRunE: checkServerHeartbeat,
		RunE: func(cmd *cobra.Command, args []string) error {
			var name string
			if len(args) > 0 {
				name = args[0]
			} else {
				var err error
				name, err = selectIntegration()
				cancelled, err := handleCancelled(err)
				if cancelled {
					return nil
				}
				if err != nil {
					return err
				}
			}

			integ, ok := integrations[strings.ToLower(name)]
			if !ok {
				return fmt.Errorf("unknown integration: %s", name)
			}

			// If --launch without --model, use saved config if available
			if launchFlag && modelFlag == "" {
				if config, err := loadIntegration(name); err == nil && config.defaultModel() != "" {
					return runIntegration(name, config.defaultModel())
				}
			}

			var models []string
			if modelFlag != "" {
				// When --model is specified, merge with existing models (new model becomes default)
				models = []string{modelFlag}
				if existing, err := loadIntegration(name); err == nil && len(existing.Models) > 0 {
					for _, m := range existing.Models {
						if m != modelFlag {
							models = append(models, m)
						}
					}
				}
			} else {
				var err error
				models, err = selectModels(cmd.Context(), name, "")
				cancelled, err := handleCancelled(err)
				if cancelled {
					return nil
				}
				if err != nil {
					return err
				}
			}

			if integ.Setup != nil && integ.ConfigPaths != nil {
				paths := integ.ConfigPaths()
				if len(paths) > 0 {
					fmt.Fprintf(os.Stderr, "This will modify your %s configuration:\n", integ.DisplayName)
					for _, p := range paths {
						fmt.Fprintf(os.Stderr, "  %s\n", p)
					}
					fmt.Fprintf(os.Stderr, "Backups will be saved to %s/\n\n", backupDir())

					if ok, _ := confirmPrompt("Proceed?"); !ok {
						return nil
					}
				}
			}

			if err := saveIntegration(name, models); err != nil {
				return fmt.Errorf("failed to save: %w", err)
			}

			if integ.Setup != nil {
				if err := integ.Setup(models); err != nil {
					return fmt.Errorf("setup failed: %w", err)
				}
			}

			printModelsAdded(integ, models)

			if launchFlag {
				return runIntegration(name, models[0])
			}

			if launch, _ := confirmPrompt(fmt.Sprintf("\nLaunch %s now?", integ.DisplayName)); launch {
				return runIntegration(name, models[0])
			}

			fmt.Fprintf(os.Stderr, "Run 'ollama config %s --launch' to start with %s\n", strings.ToLower(name), models[0])
			return nil
		},
	}

	cmd.Flags().StringVar(&modelFlag, "model", "", "Model to use")
	cmd.Flags().BoolVar(&launchFlag, "launch", false, "Launch the integration after configuring")
	return cmd
}
