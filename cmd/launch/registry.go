package launch

import (
	"fmt"
	"os/exec"
	"slices"
	"strings"

	"github.com/ollama/ollama/cmd/config"
)

// IntegrationInstallSpec describes how launcher should detect and guide installation.
type IntegrationInstallSpec struct {
	CheckInstalled  func() bool
	EnsureInstalled func() error
	URL             string
	Command         []string
}

// IntegrationSpec is the canonical registry entry for one integration.
type IntegrationSpec struct {
	Name        string
	Runner      Runner
	Aliases     []string
	Hidden      bool
	Description string
	Install     IntegrationInstallSpec
}

// IntegrationInfo contains display information about a registered integration.
type IntegrationInfo struct {
	Name        string
	DisplayName string
	Description string
}

var launcherIntegrationOrder = []string{"opencode", "droid", "pi", "cline"}

var integrationSpecs = []*IntegrationSpec{
	{
		Name:        "claude",
		Runner:      &Claude{},
		Description: "Anthropic's coding tool with subagents",
		Install: IntegrationInstallSpec{
			CheckInstalled: func() bool {
				_, err := (&Claude{}).findPath()
				return err == nil
			},
			URL: "https://code.claude.com/docs/en/quickstart",
		},
	},
	{
		Name:        "cline",
		Runner:      &Cline{},
		Description: "Autonomous coding agent with parallel execution",
		Install: IntegrationInstallSpec{
			CheckInstalled: func() bool {
				_, err := exec.LookPath("cline")
				return err == nil
			},
			Command: []string{"npm", "install", "-g", "cline"},
		},
	},
	{
		Name:        "codex",
		Runner:      &Codex{},
		Description: "OpenAI's open-source coding agent",
		Install: IntegrationInstallSpec{
			CheckInstalled: func() bool {
				_, err := exec.LookPath("codex")
				return err == nil
			},
			URL:     "https://developers.openai.com/codex/cli/",
			Command: []string{"npm", "install", "-g", "@openai/codex"},
		},
	},
	{
		Name:        "droid",
		Runner:      &Droid{},
		Description: "Factory's coding agent across terminal and IDEs",
		Install: IntegrationInstallSpec{
			CheckInstalled: func() bool {
				_, err := exec.LookPath("droid")
				return err == nil
			},
			URL: "https://docs.factory.ai/cli/getting-started/quickstart",
		},
	},
	{
		Name:        "opencode",
		Runner:      &OpenCode{},
		Description: "Anomaly's open-source coding agent",
		Install: IntegrationInstallSpec{
			CheckInstalled: func() bool {
				_, err := exec.LookPath("opencode")
				return err == nil
			},
			URL: "https://opencode.ai",
		},
	},
	{
		Name:        "openclaw",
		Runner:      &Openclaw{},
		Aliases:     []string{"clawdbot", "moltbot"},
		Description: "Personal AI with 100+ skills",
		Install: IntegrationInstallSpec{
			CheckInstalled: func() bool {
				if _, err := exec.LookPath("openclaw"); err == nil {
					return true
				}
				if _, err := exec.LookPath("clawdbot"); err == nil {
					return true
				}
				return false
			},
			EnsureInstalled: func() error {
				_, err := ensureOpenclawInstalled()
				return err
			},
			URL: "https://docs.openclaw.ai",
		},
	},
	{
		Name:        "pi",
		Runner:      &Pi{},
		Description: "Minimal AI agent toolkit with plugin support",
		Install: IntegrationInstallSpec{
			CheckInstalled: func() bool {
				_, err := exec.LookPath("pi")
				return err == nil
			},
			Command: []string{"npm", "install", "-g", "@mariozechner/pi-coding-agent"},
		},
	},
}

var integrationSpecsByName map[string]*IntegrationSpec

func init() {
	rebuildIntegrationSpecIndexes()
}

func hyperlink(url, text string) string {
	return fmt.Sprintf("\033]8;;%s\033\\%s\033]8;;\033\\", url, text)
}

func rebuildIntegrationSpecIndexes() {
	integrationSpecsByName = make(map[string]*IntegrationSpec, len(integrationSpecs))

	canonical := make(map[string]bool, len(integrationSpecs))
	for _, spec := range integrationSpecs {
		key := strings.ToLower(spec.Name)
		if key == "" {
			panic("launch: integration spec missing name")
		}
		if canonical[key] {
			panic(fmt.Sprintf("launch: duplicate integration name %q", key))
		}
		canonical[key] = true
		integrationSpecsByName[key] = spec
	}

	seenAliases := make(map[string]string)
	for _, spec := range integrationSpecs {
		for _, alias := range spec.Aliases {
			key := strings.ToLower(alias)
			if key == "" {
				panic(fmt.Sprintf("launch: integration %q has empty alias", spec.Name))
			}
			if canonical[key] {
				panic(fmt.Sprintf("launch: alias %q collides with canonical integration name", key))
			}
			if owner, exists := seenAliases[key]; exists {
				panic(fmt.Sprintf("launch: alias %q collides between %q and %q", key, owner, spec.Name))
			}
			seenAliases[key] = spec.Name
			integrationSpecsByName[key] = spec
		}
	}

	orderSeen := make(map[string]bool, len(launcherIntegrationOrder))
	for _, name := range launcherIntegrationOrder {
		key := strings.ToLower(name)
		if orderSeen[key] {
			panic(fmt.Sprintf("launch: duplicate launcher order entry %q", key))
		}
		orderSeen[key] = true

		spec, ok := integrationSpecsByName[key]
		if !ok {
			panic(fmt.Sprintf("launch: unknown launcher order entry %q", key))
		}
		if spec.Name != key {
			panic(fmt.Sprintf("launch: launcher order entry %q must use canonical name, not alias", key))
		}
		if spec.Hidden {
			panic(fmt.Sprintf("launch: hidden integration %q cannot appear in launcher order", key))
		}
	}
}

// LookupIntegrationSpec resolves either a canonical integration name or alias to its spec.
func LookupIntegrationSpec(name string) (*IntegrationSpec, error) {
	spec, ok := integrationSpecsByName[strings.ToLower(name)]
	if !ok {
		return nil, fmt.Errorf("unknown integration: %s", name)
	}
	return spec, nil
}

// OverrideIntegration replaces one registry entry's runner and returns a restore function.
func OverrideIntegration(name string, runner Runner) func() {
	spec, err := LookupIntegrationSpec(name)
	if err != nil {
		key := strings.ToLower(name)
		integrationSpecsByName[key] = &IntegrationSpec{Name: key, Runner: runner}
		return func() {
			delete(integrationSpecsByName, key)
		}
	}

	original := spec.Runner
	spec.Runner = runner
	return func() {
		spec.Runner = original
	}
}

// LookupIntegration resolves a registry name to the canonical key and runner.
func LookupIntegration(name string) (string, Runner, error) {
	spec, err := LookupIntegrationSpec(name)
	if err != nil {
		return "", nil, err
	}
	return spec.Name, spec.Runner, nil
}

// ListVisibleIntegrationSpecs returns the canonical integrations that should appear in interactive UIs.
func ListVisibleIntegrationSpecs() []IntegrationSpec {
	visible := make([]IntegrationSpec, 0, len(integrationSpecs))
	for _, spec := range integrationSpecs {
		if spec.Hidden {
			continue
		}
		visible = append(visible, *spec)
	}

	orderRank := make(map[string]int, len(launcherIntegrationOrder))
	for i, name := range launcherIntegrationOrder {
		orderRank[name] = i + 1
	}

	slices.SortFunc(visible, func(a, b IntegrationSpec) int {
		aRank, bRank := orderRank[a.Name], orderRank[b.Name]
		if aRank > 0 && bRank > 0 {
			return aRank - bRank
		}
		if aRank > 0 {
			return 1
		}
		if bRank > 0 {
			return -1
		}
		return strings.Compare(a.Name, b.Name)
	})

	return visible
}

// ListIntegrationInfos returns the registered integrations in launcher display order.
func ListIntegrationInfos() []IntegrationInfo {
	visible := ListVisibleIntegrationSpecs()
	infos := make([]IntegrationInfo, 0, len(visible))
	for _, spec := range visible {
		infos = append(infos, IntegrationInfo{
			Name:        spec.Name,
			DisplayName: spec.Runner.String(),
			Description: spec.Description,
		})
	}
	return infos
}

// IntegrationSelectionItems returns the sorted integration items shown by launcher selection UIs.
func IntegrationSelectionItems() ([]ModelItem, error) {
	visible := ListVisibleIntegrationSpecs()
	if len(visible) == 0 {
		return nil, fmt.Errorf("no integrations available")
	}

	items := make([]ModelItem, 0, len(visible))
	for _, spec := range visible {
		description := spec.Runner.String()
		if conn, err := config.LoadIntegration(spec.Name); err == nil && len(conn.Models) > 0 {
			description = fmt.Sprintf("%s (%s)", spec.Runner.String(), conn.Models[0])
		}
		items = append(items, ModelItem{Name: spec.Name, Description: description})
	}
	return items, nil
}

// IsIntegrationInstalled checks if an integration binary is installed.
func IsIntegrationInstalled(name string) bool {
	spec, err := LookupIntegrationSpec(name)
	if err != nil {
		return true
	}
	if spec.Install.CheckInstalled == nil {
		return true
	}
	return spec.Install.CheckInstalled()
}

// AutoInstallable returns true if the integration can be automatically installed when missing.
func AutoInstallable(name string) bool {
	spec, err := LookupIntegrationSpec(name)
	if err != nil {
		return false
	}
	return spec.Install.EnsureInstalled != nil
}

// EnsureInstalled checks if an auto-installable integration is present and offers to install it if missing.
func EnsureInstalled(name string) error {
	spec, err := LookupIntegrationSpec(name)
	if err != nil {
		return err
	}
	if spec.Install.EnsureInstalled == nil || IsIntegrationInstalled(name) {
		return nil
	}
	return spec.Install.EnsureInstalled()
}

// IsEditorIntegration returns true if the named integration uses multi-model selection.
func IsEditorIntegration(name string) bool {
	_, runner, err := LookupIntegration(name)
	if err != nil {
		return false
	}
	_, isEditor := runner.(Editor)
	return isEditor
}

// IntegrationInstallHint returns a user-friendly install hint for the given integration.
func IntegrationInstallHint(name string) string {
	spec, err := LookupIntegrationSpec(name)
	if err != nil {
		return ""
	}
	if spec.Install.URL != "" {
		return "Install from " + hyperlink(spec.Install.URL, spec.Install.URL)
	}
	if len(spec.Install.Command) > 0 {
		return "Install with: " + strings.Join(spec.Install.Command, " ")
	}
	return ""
}

// EnsureIntegrationInstalled installs auto-installable integrations when missing.
func EnsureIntegrationInstalled(name string, runner Runner) error {
	if IsIntegrationInstalled(name) {
		return nil
	}
	if AutoInstallable(name) {
		return EnsureInstalled(name)
	}
	return IntegrationInstallError(name, runner)
}

// IntegrationInstallError reports a user-facing install error for missing integrations.
func IntegrationInstallError(name string, runner Runner) error {
	spec, err := LookupIntegrationSpec(name)
	if err != nil {
		return fmt.Errorf("%s is not installed", runner)
	}
	switch {
	case spec.Install.URL != "":
		return fmt.Errorf("%s is not installed, install from %s", spec.Name, spec.Install.URL)
	case len(spec.Install.Command) > 0:
		return fmt.Errorf("%s is not installed, install with: %s", spec.Name, strings.Join(spec.Install.Command, " "))
	default:
		return fmt.Errorf("%s is not installed", runner)
	}
}
