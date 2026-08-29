// Package agent provides the core runtime abstractions for Ollama's agent
// tool-calling system.
//
// The environment layer adds first-class support for targeting tool execution
// at a specific execution environment (local machine, container, remote server,
// cloud VM, etc.). The agent always knows WHERE it is executing.
//
// Design goals:
//   - Environment targeting is explicit, never ambiguous.
//   - Capabilities are discovered, not assumed.
//   - Remote/cloud providers are pluggable, never hard-coded.
//   - Existing approval infrastructure is reused, not bypassed.
//   - Computer input serialization is per-environment, not global.
package agent

import (
	"context"
	"fmt"
	"strings"
	"sync"
)

// EnvironmentType classifies the execution environment.
type EnvironmentType string

const (
	EnvironmentLocal     EnvironmentType = "local"
	EnvironmentContainer EnvironmentType = "container"
	EnvironmentVM        EnvironmentType = "vm"
	EnvironmentRemote    EnvironmentType = "remote"
	EnvironmentCloud     EnvironmentType = "cloud"
)

// Capability identifies what an environment can do.
type Capability string

const (
	CapShell     Capability = "shell"
	CapFiles     Capability = "files"
	CapComputer  Capability = "computer"
	CapProcesses Capability = "processes"
)

// ComputerBackend is the interface that environment-specific computer
// implementations must satisfy. Each environment that advertises CapComputer
// must provide its own backend. This prevents a remote target from
// accidentally executing actions on the local machine.
type ComputerBackend interface {
	Screenshot(ctx context.Context) ([]byte, int, int, error)
	Click(ctx context.Context, x, y int) error
	DoubleClick(ctx context.Context, x, y int) error
	Move(ctx context.Context, x, y int) error
	Type(ctx context.Context, text string) error
	Key(ctx context.Context, key string) error
	Scroll(ctx context.Context, dx, dy int) error
}

// Environment is the interface that all execution environments must satisfy.
// An environment represents a target where the agent can execute actions.
type Environment interface {
	// ID returns a unique identifier for this environment (e.g. "local",
	// "prod-server", "dev-container").
	ID() string

	// Type returns the classification of this environment.
	Type() EnvironmentType

	// Capabilities returns the set of capabilities this environment supports.
	Capabilities() []Capability

	// SupportsCapability reports whether the environment has a given capability.
	SupportsCapability(c Capability) bool

	// ComputerBackend returns the computer execution backend for this
	// environment. Returns nil if the environment does not have a configured
	// computer backend (even if it advertises CapComputer in capabilities).
	// Callers MUST check for nil before attempting to use the backend.
	ComputerBackend() ComputerBackend
}

// EnvironmentDescriptor is a lightweight value describing an environment,
// suitable for JSON serialization and tool schema documentation.
type EnvironmentDescriptor struct {
	ID           string           `json:"id"`
	Type         EnvironmentType `json:"type"`
	Capabilities []Capability    `json:"capabilities"`
}

// Descriptor converts an Environment to its descriptor form.
func Descriptor(env Environment) EnvironmentDescriptor {
	return EnvironmentDescriptor{
		ID:           env.ID(),
		Type:         env.Type(),
		Capabilities: env.Capabilities(),
	}
}

// Descriptors converts a slice of environments to their descriptor forms.
func Descriptors(envs []Environment) []EnvironmentDescriptor {
	out := make([]EnvironmentDescriptor, len(envs))
	for i, env := range envs {
		out[i] = Descriptor(env)
	}
	return out
}

// EnvironmentRegistry manages the set of known execution environments.
// It provides lookup by ID and capability-based discovery.
type EnvironmentRegistry struct {
	mu       sync.RWMutex
	envs     map[string]Environment
	ordered  []string // insertion order for deterministic iteration
}

// NewEnvironmentRegistry creates a new, empty registry.
func NewEnvironmentRegistry() *EnvironmentRegistry {
	return &EnvironmentRegistry{
		envs: make(map[string]Environment),
	}
}

// Register adds an environment to the registry. Environments with duplicate IDs
// are silently overwritten (last wins).
func (r *EnvironmentRegistry) Register(env Environment) {
	if r == nil || env == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()

	id := env.ID()
	if _, exists := r.envs[id]; !exists {
		r.ordered = append(r.ordered, id)
	}
	r.envs[id] = env
}

// Get retrieves an environment by ID.
func (r *EnvironmentRegistry) Get(id string) (Environment, bool) {
	if r == nil {
		return nil, false
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	env, ok := r.envs[id]
	return env, ok
}

// List returns all registered environments in insertion order.
func (r *EnvironmentRegistry) List() []Environment {
	if r == nil {
		return nil
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]Environment, 0, len(r.ordered))
	for _, id := range r.ordered {
		out = append(out, r.envs[id])
	}
	return out
}

// Descriptors returns all registered environments as descriptors.
func (r *EnvironmentRegistry) Descriptors() []EnvironmentDescriptor {
	return Descriptors(r.List())
}

// WithCapability returns all environments that support the given capability.
func (r *EnvironmentRegistry) WithCapability(c Capability) []Environment {
	if r == nil {
		return nil
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	var out []Environment
	for _, id := range r.ordered {
		if r.envs[id].SupportsCapability(c) {
			out = append(out, r.envs[id])
		}
	}
	return out
}

// ResolveTarget resolves an explicit environment ID from tool arguments.
// If the ID is empty or "local", the local environment is returned.
// Returns an error if the target is specified but not found.
func (r *EnvironmentRegistry) ResolveTarget(target string) (Environment, error) {
	target = strings.TrimSpace(strings.ToLower(target))
	if target == "" || target == "local" {
		env, ok := r.Get("local")
		if !ok {
			return nil, fmt.Errorf("local environment not available")
		}
		return env, nil
	}
	env, ok := r.Get(target)
	if !ok {
		return nil, fmt.Errorf("unknown environment %q; available environments: %s", target, r.listIDs())
	}
	return env, nil
}

func (r *EnvironmentRegistry) listIDs() string {
	if r == nil {
		return ""
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	ids := make([]string, 0, len(r.ordered))
	for _, id := range r.ordered {
		ids = append(ids, fmt.Sprintf("%q", id))
	}
	return strings.Join(ids, ", ")
}

// DefaultEnvironmentID returns the ID used when no explicit target is provided.
const DefaultEnvironmentID = "local"
