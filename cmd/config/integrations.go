package config

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"os"
	"os/exec"
	"runtime"
	"slices"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/spf13/cobra"
)

// Runners execute the launching of a model with the integration - claude, codex
// Editors can edit config files (supports multi-model selection) - opencode, droid
// They are composable interfaces where in some cases an editor is also a runner - opencode, droid
// Runner can run an integration with a model.

type Runner interface {
	Run(model string, args []string) error
	// String returns the human-readable name of the integration
	String() string
}

// Editor can edit config files (supports multi-model selection)
type Editor interface {
	// Paths returns the paths to the config files for the integration
	Paths() []string
	// Edit updates the config files for the integration with the given models
	Edit(models []string) error
	// Models returns the models currently configured for the integration
	// TODO(parthsareen): add error return to Models()
	Models() []string
}

// integrations is the registry of available integrations.
var integrations = map[string]Runner{
	"claude":   &Claude{},
	"clawdbot": &Openclaw{},
	"codex":    &Codex{},
	"moltbot":  &Openclaw{},
	"droid":    &Droid{},
	"opencode": &OpenCode{},
	"openclaw": &Openclaw{},
}

// integrationAliases are hidden from the interactive selector but work as CLI arguments.
var integrationAliases = map[string]bool{
	"clawdbot": true,
	"moltbot":  true,
}

func selectIntegration() (string, error) {
	if len(integrations) == 0 {
		return "", fmt.Errorf("no integrations available")
	}

	names := slices.Sorted(maps.Keys(integrations))
	var items []selectItem
	for _, name := range names {
		if integrationAliases[name] {
			continue
		}
		r := integrations[name]
		description := r.String()
		if conn, err := loadIntegration(name); err == nil && len(conn.Models) > 0 {
			description = fmt.Sprintf("%s (%s)", r.String(), conn.Models[0])
		}
		items = append(items, selectItem{Name: name, Description: description})
	}

	return selectPrompt("Select integration:", items)
}

// selectModels lets the user select models for an integration
func selectModels(ctx context.Context, name, current string) ([]string, error) {
	r, ok := integrations[name]
	if !ok {
		return nil, fmt.Errorf("unknown integration: %s", name)
	}

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

	// Get previously configured models (saved config takes precedence)
	var preChecked []string
	if saved, err := loadIntegration(name); err == nil {
		preChecked = saved.Models
	} else if editor, ok := r.(Editor); ok {
		preChecked = editor.Models()
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

	var selected []string
	// only editors support multi-model selection
	if _, ok := r.(Editor); ok {
		selected, err = multiSelectPrompt(fmt.Sprintf("Select models for %s:", r), items, preChecked)
		if err != nil {
			return nil, err
		}
	} else {
		model, err := selectPrompt(fmt.Sprintf("Select model for %s:", r), items)
		if err != nil {
			return nil, err
		}
		selected = []string{model}
	}

	// if any model in selected is a cloud model, ensure signed in
	var selectedCloudModels []string
	for _, m := range selected {
		if cloudModels[m] {
			selectedCloudModels = append(selectedCloudModels, m)
		}
	}
	if len(selectedCloudModels) > 0 {
		// ensure user is signed in
		user, err := client.Whoami(ctx)
		if err == nil && user != nil && user.Name != "" {
			return selected, nil
		}

		var aErr api.AuthorizationError
		if !errors.As(err, &aErr) || aErr.SigninURL == "" {
			return nil, err
		}

		modelList := strings.Join(selectedCloudModels, ", ")
		yes, err := confirmPrompt(fmt.Sprintf("sign in to use %s?", modelList))
		if err != nil || !yes {
			return nil, fmt.Errorf("%s requires sign in", modelList)
		}

		fmt.Fprintf(os.Stderr, "\nTo sign in, navigate to:\n    %s\n\n", aErr.SigninURL)

		// TODO(parthsareen): extract into auth package for cmd
		// Auto-open browser (best effort, fail silently)
		switch runtime.GOOS {
		case "darwin":
			_ = exec.Command("open", aErr.SigninURL).Start()
		case "linux":
			_ = exec.Command("xdg-open", aErr.SigninURL).Start()
		case "windows":
			_ = exec.Command("rundll32", "url.dll,FileProtocolHandler", aErr.SigninURL).Start()
		}

		spinnerFrames := []string{"|", "/", "-", "\\"}
		frame := 0

		fmt.Fprintf(os.Stderr, "\033[90mwaiting for sign in to complete... %s\033[0m", spinnerFrames[0])

		ticker := time.NewTicker(200 * time.Millisecond)
		defer ticker.Stop()

		for {
			select {
			case <-ctx.Done():
				fmt.Fprintf(os.Stderr, "\r\033[K")
				return nil, ctx.Err()
			case <-ticker.C:
				frame++
				fmt.Fprintf(os.Stderr, "\r\033[90mwaiting for sign in to complete... %s\033[0m", spinnerFrames[frame%len(spinnerFrames)])

				// poll every 10th frame (~2 seconds)
				if frame%10 == 0 {
					u, err := client.Whoami(ctx)
					if err == nil && u != nil && u.Name != "" {
						fmt.Fprintf(os.Stderr, "\r\033[K\033[A\r\033[K\033[1msigned in:\033[0m %s\n", u.Name)
						return selected, nil
					}
				}
			}
		}
	}

	return selected, nil
}

func runIntegration(name, modelName string, args []string) error {
	r, ok := integrations[name]
	if !ok {
		return fmt.Errorf("unknown integration: %s", name)
	}
	fmt.Fprintf(os.Stderr, "\nLaunching %s with %s...\n", r, modelName)
	return r.Run(modelName, args)
}

// LaunchCmd returns the cobra command for launching integrations.
func LaunchCmd(checkServerHeartbeat func(cmd *cobra.Command, args []string) error) *cobra.Command {
	var modelFlag string
	var configFlag bool

	cmd := &cobra.Command{
		Use:   "launch [INTEGRATION] [-- [EXTRA_ARGS...]]",
		Short: "Launch an integration with Ollama",
		Long: `Launch an integration configured with Ollama models.

Supported integrations:
  claude    Claude Code
  codex     Codex
  droid     Droid
  opencode  OpenCode
  openclaw  OpenClaw (aliases: clawdbot, moltbot)

Examples:
  ollama launch
  ollama launch claude
  ollama launch claude --model <model>
  ollama launch droid --config (does not auto-launch)
  ollama launch codex -- -p myprofile (pass extra args to integration)
  ollama launch codex -- --sandbox workspace-write`,
		Args:    cobra.ArbitraryArgs,
		PreRunE: checkServerHeartbeat,
		RunE: func(cmd *cobra.Command, args []string) error {
			// Extract integration name and args to pass through using -- separator
			var name string
			var passArgs []string
			dashIdx := cmd.ArgsLenAtDash()

			if dashIdx == -1 {
				// No "--" separator: only allow 0 or 1 args (integration name)
				if len(args) > 1 {
					return fmt.Errorf("unexpected arguments: %v\nUse '--' to pass extra arguments to the integration", args[1:])
				}
				if len(args) == 1 {
					name = args[0]
				}
			} else {
				// "--" was used: args before it = integration name, args after = passthrough
				if dashIdx > 1 {
					return fmt.Errorf("expected at most 1 integration name before '--', got %d", dashIdx)
				}
				if dashIdx == 1 {
					name = args[0]
				}
				passArgs = args[dashIdx:]
			}

			if name == "" {
				var err error
				name, err = selectIntegration()
				if errors.Is(err, errCancelled) {
					return nil
				}
				if err != nil {
					return err
				}
			}

			r, ok := integrations[strings.ToLower(name)]
			if !ok {
				return fmt.Errorf("unknown integration: %s", name)
			}

			// If launching without --model, use saved config if available
			if !configFlag && modelFlag == "" {
				if config, err := loadIntegration(name); err == nil && len(config.Models) > 0 {
					return runIntegration(name, config.Models[0], passArgs)
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
				if errors.Is(err, errCancelled) {
					return nil
				}
				if err != nil {
					return err
				}
			}

			if editor, isEditor := r.(Editor); isEditor {
				paths := editor.Paths()
				if len(paths) > 0 {
					fmt.Fprintf(os.Stderr, "This will modify your %s configuration:\n", r)
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

			if editor, isEditor := r.(Editor); isEditor {
				if err := editor.Edit(models); err != nil {
					return fmt.Errorf("setup failed: %w", err)
				}
			}

			if _, isEditor := r.(Editor); isEditor {
				if len(models) == 1 {
					fmt.Fprintf(os.Stderr, "Added %s to %s\n", models[0], r)
				} else {
					fmt.Fprintf(os.Stderr, "Added %d models to %s (default: %s)\n", len(models), r, models[0])
				}
			}

			if configFlag {
				if launch, _ := confirmPrompt(fmt.Sprintf("\nLaunch %s now?", r)); launch {
					return runIntegration(name, models[0], passArgs)
				}
				fmt.Fprintf(os.Stderr, "Run 'ollama launch %s' to start with %s\n", strings.ToLower(name), models[0])
				return nil
			}

			return runIntegration(name, models[0], passArgs)
		},
	}

	cmd.Flags().StringVar(&modelFlag, "model", "", "Model to use")
	cmd.Flags().BoolVar(&configFlag, "config", false, "Configure without launching")
	return cmd
}
