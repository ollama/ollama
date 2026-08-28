package agent

import (
	"context"
	"fmt"
	"testing"
)

// --- mock environment for testing ---

type mockEnvironment struct {
	id         string
	envType    EnvironmentType
	capabilities []Capability
}

func (m *mockEnvironment) ID() string             { return m.id }
func (m *mockEnvironment) Type() EnvironmentType  { return m.envType }
func (m *mockEnvironment) Capabilities() []Capability { return m.capabilities }
func (m *mockEnvironment) SupportsCapability(c Capability) bool {
	for _, cap := range m.capabilities {
		if cap == c {
			return true
		}
	}
	return false
}

func TestLocalEnvironmentIDAndType(t *testing.T) {
	env := NewLocalEnvironment(true)
	if got := env.ID(); got != "local" {
		t.Fatalf("ID() = %q, want %q", got, "local")
	}
	if got := env.Type(); got != EnvironmentLocal {
		t.Fatalf("Type() = %q, want %q", got, EnvironmentLocal)
	}
}

func TestLocalEnvironmentCapabilities(t *testing.T) {
	tests := []struct {
		name     string
		withComp bool
		wantCaps []Capability
	}{
		{"without computer", false, []Capability{CapShell, CapFiles}},
		{"with computer", true, []Capability{CapShell, CapFiles, CapComputer}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			env := NewLocalEnvironment(tt.withComp)
			caps := env.Capabilities()
			if len(caps) != len(tt.wantCaps) {
				t.Fatalf("Capabilities() length = %d, want %d", len(caps), len(tt.wantCaps))
			}
			for i, want := range tt.wantCaps {
				if caps[i] != want {
					t.Fatalf("Capabilities()[%d] = %q, want %q", i, caps[i], want)
				}
			}
		})
	}
}

func TestLocalEnvironmentSupportsCapability(t *testing.T) {
	env := NewLocalEnvironment(true)
	if !env.SupportsCapability(CapShell) {
		t.Fatal("local environment should support shell")
	}
	if !env.SupportsCapability(CapFiles) {
		t.Fatal("local environment should support files")
	}
	if !env.SupportsCapability(CapComputer) {
		t.Fatal("local environment should support computer")
	}
	if env.SupportsCapability(CapProcesses) {
		t.Fatal("local environment should not support processes")
	}
}

func TestLocalEnvironmentHasComputerCapability(t *testing.T) {
	withComp := NewLocalEnvironment(true)
	withoutComp := NewLocalEnvironment(false)

	if !withComp.HasComputerCapability() {
		t.Fatal("expected HasComputerCapability() = true")
	}
	if withoutComp.HasComputerCapability() {
		t.Fatal("expected HasComputerCapability() = false")
	}
}

func TestDescriptor(t *testing.T) {
	env := &mockEnvironment{
		id:           "test-env",
		envType:      EnvironmentRemote,
		capabilities: []Capability{CapShell, CapComputer},
	}

	desc := Descriptor(env)
	if desc.ID != "test-env" {
		t.Fatalf("Descriptor().ID = %q, want %q", desc.ID, "test-env")
	}
	if desc.Type != EnvironmentRemote {
		t.Fatalf("Descriptor().Type = %q, want %q", desc.Type, EnvironmentRemote)
	}
	if len(desc.Capabilities) != 2 {
		t.Fatalf("Descriptor().Capabilities length = %d, want 2", len(desc.Capabilities))
	}
}

func TestDescriptors(t *testing.T) {
	envs := []Environment{
		&mockEnvironment{id: "a", envType: EnvironmentLocal, capabilities: []Capability{CapShell}},
		&mockEnvironment{id: "b", envType: EnvironmentRemote, capabilities: []Capability{CapComputer}},
	}
	descs := Descriptors(envs)
	if len(descs) != 2 {
		t.Fatalf("Descriptors() length = %d, want 2", len(descs))
	}
	if descs[0].ID != "a" || descs[1].ID != "b" {
		t.Fatalf("Descriptors() IDs = %v, want [a, b]", []string{descs[0].ID, descs[1].ID})
	}
}

func TestEnvironmentRegistryRegisterAndGet(t *testing.T) {
	r := NewEnvironmentRegistry()
	env := &mockEnvironment{id: "local", envType: EnvironmentLocal}
	r.Register(env)

	got, ok := r.Get("local")
	if !ok {
		t.Fatal("Get('local') returned false")
	}
	if got.ID() != "local" {
		t.Fatalf("Get('local').ID() = %q, want %q", got.ID(), "local")
	}
}

func TestEnvironmentRegistryDuplicateOverwrites(t *testing.T) {
	r := NewEnvironmentRegistry()
	r.Register(&mockEnvironment{id: "local", envType: EnvironmentLocal})
	r.Register(&mockEnvironment{id: "local", envType: EnvironmentContainer})

	got, ok := r.Get("local")
	if !ok {
		t.Fatal("Get('local') returned false")
	}
	if got.Type() != EnvironmentContainer {
		t.Fatalf("Get('local').Type() = %q, want %q (last wins)", got.Type(), EnvironmentContainer)
	}
}

func TestEnvironmentRegistryList(t *testing.T) {
	r := NewEnvironmentRegistry()
	r.Register(&mockEnvironment{id: "c", envType: EnvironmentCloud})
	r.Register(&mockEnvironment{id: "a", envType: EnvironmentLocal})
	r.Register(&mockEnvironment{id: "b", envType: EnvironmentRemote})

	list := r.List()
	if len(list) != 3 {
		t.Fatalf("List() length = %d, want 3", len(list))
	}
	// Should preserve insertion order
	expectedOrder := []string{"c", "a", "b"}
	for i, want := range expectedOrder {
		if list[i].ID() != want {
			t.Fatalf("List()[%d].ID() = %q, want %q", i, list[i].ID(), want)
		}
	}
}

func TestEnvironmentRegistryDescriptors(t *testing.T) {
	r := NewEnvironmentRegistry()
	r.Register(&mockEnvironment{id: "local", envType: EnvironmentLocal, capabilities: []Capability{CapShell, CapComputer}})

	descs := r.Descriptors()
	if len(descs) != 1 {
		t.Fatalf("Descriptors() length = %d, want 1", len(descs))
	}
	if descs[0].ID != "local" {
		t.Fatalf("Descriptors()[0].ID = %q, want %q", descs[0].ID, "local")
	}
}

func TestEnvironmentRegistryWithCapability(t *testing.T) {
	r := NewEnvironmentRegistry()
	r.Register(&mockEnvironment{id: "local", envType: EnvironmentLocal, capabilities: []Capability{CapShell, CapFiles, CapComputer}})
	r.Register(&mockEnvironment{id: "server", envType: EnvironmentRemote, capabilities: []Capability{CapShell, CapFiles}})
	r.Register(&mockEnvironment{id: "desktop", envType: EnvironmentRemote, capabilities: []Capability{CapShell, CapComputer}})

	computerEnvs := r.WithCapability(CapComputer)
	if len(computerEnvs) != 2 {
		t.Fatalf("WithCapability(CapComputer) length = %d, want 2", len(computerEnvs))
	}

	shellEnvs := r.WithCapability(CapShell)
	if len(shellEnvs) != 3 {
		t.Fatalf("WithCapability(CapShell) length = %d, want 3", len(shellEnvs))
	}
}

func TestEnvironmentRegistryResolveTarget(t *testing.T) {
	r := NewEnvironmentRegistry()
	r.Register(&mockEnvironment{id: "local", envType: EnvironmentLocal, capabilities: []Capability{CapShell, CapComputer}})
	r.Register(&mockEnvironment{id: "prod", envType: EnvironmentRemote, capabilities: []Capability{CapShell}})

	tests := []struct {
		name    string
		target  string
		wantID  string
		wantErr bool
	}{
		{"empty defaults to local", "", "local", false},
		{"explicit local", "local", "local", false},
		{"local uppercase", "LOCAL", "local", false},
		{"known remote", "prod", "prod", false},
		{"unknown target", "nonexistent", "", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			env, err := r.ResolveTarget(tt.target)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("ResolveTarget(%q) should have errored, got %v", tt.target, env)
				}
				return
			}
			if err != nil {
				t.Fatalf("ResolveTarget(%q) error: %v", tt.target, err)
			}
			if env.ID() != tt.wantID {
				t.Fatalf("ResolveTarget(%q).ID() = %q, want %q", tt.target, env.ID(), tt.wantID)
			}
		})
	}
}

func TestEnvironmentRegistryResolveTargetEmpty(t *testing.T) {
	r := NewEnvironmentRegistry()
	_, err := r.ResolveTarget("")
	if err == nil {
		t.Fatal("ResolveTarget('') on empty registry should error")
	}
}

func TestEnvironmentRegistryNilSafety(t *testing.T) {
	var r *EnvironmentRegistry

	// All methods should be nil-safe
	r.Register(nil)
	env, ok := r.Get("x")
	if ok || env != nil {
		t.Fatal("nil registry Get should return nil, false")
	}
	if list := r.List(); list != nil {
		t.Fatal("nil registry List should return nil")
	}
	if descs := r.Descriptors(); descs != nil {
		t.Fatal("nil registry Descriptors should return nil")
	}
	if envs := r.WithCapability(CapShell); envs != nil {
		t.Fatal("nil registry WithCapability should return nil")
	}
}

func TestEnvironmentTypeConstants(t *testing.T) {
	// Ensure environment type constants are defined as expected
	types := []EnvironmentType{
		EnvironmentLocal,
		EnvironmentContainer,
		EnvironmentVM,
		EnvironmentRemote,
		EnvironmentCloud,
	}
	seen := make(map[EnvironmentType]bool)
	for _, et := range types {
		if seen[et] {
			t.Fatalf("duplicate EnvironmentType: %q", et)
		}
		seen[et] = true
		if et == "" {
			t.Fatal("EnvironmentType must not be empty")
		}
	}
}

func TestCapabilityConstants(t *testing.T) {
	caps := []Capability{CapShell, CapFiles, CapComputer, CapProcesses}
	seen := make(map[Capability]bool)
	for _, c := range caps {
		if seen[c] {
			t.Fatalf("duplicate Capability: %q", c)
		}
		seen[c] = true
		if c == "" {
			t.Fatal("Capability must not be empty")
		}
	}
}

func TestEnvironmentRegistryContextIntegration(t *testing.T) {
	// Verify the registry works with context (it should, since Environment
	// methods are designed to be called with context).
	r := NewEnvironmentRegistry()
	r.Register(&mockEnvironment{id: "local", envType: EnvironmentLocal, capabilities: []Capability{CapComputer}})

	ctx := context.Background()
	_ = ctx // environment doesn't need context for registration/lookup

	env, err := r.ResolveTarget("local")
	if err != nil {
		t.Fatal(err)
	}
	if !env.SupportsCapability(CapComputer) {
		t.Fatal("local environment should support computer")
	}
}

func TestDefaultEnvironmentID(t *testing.T) {
	if DefaultEnvironmentID != "local" {
		t.Fatalf("DefaultEnvironmentID = %q, want %q", DefaultEnvironmentID, "local")
	}
}

// Ensure mock compliance at compile time
var _ Environment = (*mockEnvironment)(nil)

// Verify that LocalEnvironment is registered as an Environment
func TestLocalEnvironmentInterfaceCompliance(t *testing.T) {
	var env Environment = NewLocalEnvironment(true)
	if env == nil {
		t.Fatal("LocalEnvironment should implement Environment")
	}
	_ = fmt.Sprintf("%s:%s", env.ID(), env.Type())
}
