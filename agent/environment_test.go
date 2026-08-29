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
	backend    ComputerBackend
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
func (m *mockEnvironment) ComputerBackend() ComputerBackend { return m.backend }

// --- mock computer backend for testing ---

type mockComputerBackend struct{}

func (m *mockComputerBackend) Screenshot(_ context.Context) ([]byte, int, int, error) {
	return []byte{0}, 1, 1, nil
}
func (m *mockComputerBackend) Click(_ context.Context, x, y int) error    { return nil }
func (m *mockComputerBackend) DoubleClick(_ context.Context, x, y int) error { return nil }
func (m *mockComputerBackend) Move(_ context.Context, x, y int) error     { return nil }
func (m *mockComputerBackend) Type(_ context.Context, text string) error  { return nil }
func (m *mockComputerBackend) Key(_ context.Context, key string) error    { return nil }
func (m *mockComputerBackend) Scroll(_ context.Context, dx, dy int) error { return nil }

var _ ComputerBackend = (*mockComputerBackend)(nil)
var _ Environment = (*mockEnvironment)(nil)

func TestLocalEnvironmentIDAndType(t *testing.T) {
	env := NewLocalEnvironment(&mockComputerBackend{})
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
		backend  ComputerBackend
		wantCaps []Capability
	}{
		{"with backend", &mockComputerBackend{}, []Capability{CapShell, CapFiles, CapComputer}},
		{"without backend", nil, []Capability{CapShell, CapFiles}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			env := NewLocalEnvironment(tt.backend)
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
	env := NewLocalEnvironment(&mockComputerBackend{})
	if !env.SupportsCapability(CapShell) {
		t.Fatal("local should support shell")
	}
	if !env.SupportsCapability(CapFiles) {
		t.Fatal("local should support files")
	}
	if !env.SupportsCapability(CapComputer) {
		t.Fatal("local should support computer")
	}
	if env.SupportsCapability(CapProcesses) {
		t.Fatal("local should not support processes")
	}
}

func TestLocalEnvironmentComputerBackend(t *testing.T) {
	backend := &mockComputerBackend{}
	env := NewLocalEnvironment(backend)
	if env.ComputerBackend() != backend {
		t.Fatal("ComputerBackend() should return the provided backend")
	}

	envNoBackend := NewLocalEnvironment(nil)
	if envNoBackend.ComputerBackend() != nil {
		t.Fatal("ComputerBackend() should return nil when no backend provided")
	}
}

func TestLocalEnvironmentHasComputerCapability(t *testing.T) {
	withComp := NewLocalEnvironment(&mockComputerBackend{})
	withoutComp := NewLocalEnvironment(nil)

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
		t.Fatalf("Descriptor().ID = %q", desc.ID)
	}
	if desc.Type != EnvironmentRemote {
		t.Fatalf("Descriptor().Type = %q", desc.Type)
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
}

func TestEnvironmentRegistryRegisterAndGet(t *testing.T) {
	r := NewEnvironmentRegistry()
	r.Register(&mockEnvironment{id: "local", envType: EnvironmentLocal})
	got, ok := r.Get("local")
	if !ok {
		t.Fatal("Get('local') returned false")
	}
	if got.ID() != "local" {
		t.Fatalf("Get('local').ID() = %q", got.ID())
	}
}

func TestEnvironmentRegistryDuplicateOverwrites(t *testing.T) {
	r := NewEnvironmentRegistry()
	r.Register(&mockEnvironment{id: "local", envType: EnvironmentLocal})
	r.Register(&mockEnvironment{id: "local", envType: EnvironmentContainer})
	got, _ := r.Get("local")
	if got.Type() != EnvironmentContainer {
		t.Fatalf("last wins: got %q", got.Type())
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
	if len(descs) != 1 || descs[0].ID != "local" {
		t.Fatalf("Descriptors() = %v", descs)
	}
}

func TestEnvironmentRegistryWithCapability(t *testing.T) {
	r := NewEnvironmentRegistry()
	r.Register(&mockEnvironment{id: "local", envType: EnvironmentLocal, capabilities: []Capability{CapShell, CapFiles, CapComputer}})
	r.Register(&mockEnvironment{id: "server", envType: EnvironmentRemote, capabilities: []Capability{CapShell, CapFiles}})
	r.Register(&mockEnvironment{id: "desktop", envType: EnvironmentRemote, capabilities: []Capability{CapShell, CapComputer}})
	computerEnvs := r.WithCapability(CapComputer)
	if len(computerEnvs) != 2 {
		t.Fatalf("WithCapability(CapComputer) = %d, want 2", len(computerEnvs))
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
					t.Fatalf("ResolveTarget(%q) should have errored", tt.target)
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
	r.Register(nil)
	_, ok := r.Get("x")
	if ok {
		t.Fatal("nil registry Get should return false")
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
	types := []EnvironmentType{
		EnvironmentLocal, EnvironmentContainer, EnvironmentVM,
		EnvironmentRemote, EnvironmentCloud,
	}
	seen := make(map[EnvironmentType]bool)
	for _, et := range types {
		if seen[et] {
			t.Fatalf("duplicate: %q", et)
		}
		seen[et] = true
	}
}

func TestCapabilityConstants(t *testing.T) {
	caps := []Capability{CapShell, CapFiles, CapComputer, CapProcesses}
	seen := make(map[Capability]bool)
	for _, c := range caps {
		if seen[c] {
			t.Fatalf("duplicate: %q", c)
		}
		seen[c] = true
	}
}

func TestDefaultEnvironmentID(t *testing.T) {
	if DefaultEnvironmentID != "local" {
		t.Fatalf("DefaultEnvironmentID = %q", DefaultEnvironmentID)
	}
}

func TestLocalEnvironmentInterfaceCompliance(t *testing.T) {
	var env Environment = NewLocalEnvironment(&mockComputerBackend{})
	if env == nil {
		t.Fatal("LocalEnvironment should implement Environment")
	}
	_ = fmt.Sprintf("%s:%s", env.ID(), env.Type())
}
