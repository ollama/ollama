package launch

import (
	"testing"
)

func TestTealKitInterfaces(t *testing.T) {
	tk := &TealKit{}

	t.Run("String", func(t *testing.T) {
		if got := tk.String(); got != "TealKit" {
			t.Errorf("String() = %q, want %q", got, "TealKit")
		}
	})

	t.Run("implements Runner", func(t *testing.T) {
		var _ Runner = tk
	})

	t.Run("implements ManagedSingleModel", func(t *testing.T) {
		var _ ManagedSingleModel = tk
	})
}

func TestTealKitRegistration(t *testing.T) {
	spec, err := LookupIntegrationSpec("tealkit")
	if err != nil {
		t.Fatalf("LookupIntegrationSpec('tealkit') failed: %v", err)
	}

	if spec.Name != "tealkit" {
		t.Errorf("spec.Name = %q, want 'tealkit'", spec.Name)
	}

	aliasFound := false
	for _, alias := range spec.Aliases {
		if alias == "tealkit-cli" {
			aliasFound = true
			break
		}
	}
	if !aliasFound {
		t.Errorf("expected 'tealkit-cli' in aliases: %v", spec.Aliases)
	}
}
