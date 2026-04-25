package launch

import (
	"testing"

	"github.com/spf13/cobra"
)

type mockLaunchConfigurator struct {
	req IntegrationLaunchRequest
}

func (m *mockLaunchConfigurator) ConfigureLaunch(req IntegrationLaunchRequest) {
	m.req = req
}

func (m *mockLaunchConfigurator) Run(model string, args []string) error {
	return nil
}

func (m *mockLaunchConfigurator) String() string {
	return "mock"
}

func TestLaunchCmdFlagsParsing(t *testing.T) {
	mockCheck := func(cmd *cobra.Command, args []string) error { return nil }
	mockTUI := func(cmd *cobra.Command) {}

	cmd := LaunchCmd(mockCheck, mockTUI)

	flags := []string{"think", "config-scope", "provider-mode", "experimental"}
	for _, f := range flags {
		if cmd.Flags().Lookup(f) == nil {
			t.Errorf("expected flag %s to be registered", f)
		}
	}

	if cmd.Flags().Lookup("think").DefValue != "auto" {
		t.Errorf("expected default think to be auto")
	}
	if cmd.Flags().Lookup("config-scope").DefValue != "user" {
		t.Errorf("expected default config-scope to be user")
	}
	if cmd.Flags().Lookup("provider-mode").DefValue != "hybrid" {
		t.Errorf("expected default provider-mode to be hybrid")
	}
	if cmd.Flags().Lookup("experimental").DefValue != "false" {
		t.Errorf("expected default experimental to be false")
	}
}

func TestLaunchConfigurator(t *testing.T) {
	runner := &mockLaunchConfigurator{}
	integrationSpecsByName["mock-configurator"] = &IntegrationSpec{
		Name:   "mock-configurator",
		Runner: runner,
	}
	defer delete(integrationSpecsByName, "mock-configurator")

	req := IntegrationLaunchRequest{
		Name:          "mock-configurator",
		ModelOverride: "test-model",
		ConfigureOnly: true,
		Think:         "on",
		ConfigScope:   "project",
		ProviderMode:  "config",
		Experimental:  true,
		Policy: &LaunchPolicy{
			Confirm:      LaunchConfirmAutoApprove,
			MissingModel: LaunchMissingModelAutoPull,
		},
	}

	if lc, ok := interface{}(runner).(LaunchConfigurator); ok {
		lc.ConfigureLaunch(req)
	} else {
		t.Fatalf("runner does not implement LaunchConfigurator")
	}

	if runner.req.Think != "on" {
		t.Errorf("expected think to be 'on', got %v", runner.req.Think)
	}
	if runner.req.ConfigScope != "project" {
		t.Errorf("expected config-scope to be 'project', got %v", runner.req.ConfigScope)
	}
	if runner.req.ProviderMode != "config" {
		t.Errorf("expected provider-mode to be 'config', got %v", runner.req.ProviderMode)
	}
	if runner.req.Experimental != true {
		t.Errorf("expected experimental to be true, got %v", runner.req.Experimental)
	}
}
