package agent

import (
	"sync"
)

// LocalEnvironment represents the machine running Ollama. It is the default
// environment and supports shell, files, and computer capabilities.
type LocalEnvironment struct {
	caps    []Capability
	mu      sync.RWMutex
	backend ComputerBackend // nil if computer capability is not available
}

// NewLocalEnvironment creates a new local environment. The backend parameter
// provides the computer execution backend. Pass nil if the platform does not
// support computer-use primitives (only shell and file capabilities will be
// advertised).
func NewLocalEnvironment(backend ComputerBackend) *LocalEnvironment {
	caps := []Capability{CapShell, CapFiles}
	if backend != nil {
		caps = append(caps, CapComputer)
	}
	return &LocalEnvironment{
		caps:    caps,
		backend: backend,
	}
}

func (l *LocalEnvironment) ID() string {
	return "local"
}

func (l *LocalEnvironment) Type() EnvironmentType {
	return EnvironmentLocal
}

func (l *LocalEnvironment) Capabilities() []Capability {
	l.mu.RLock()
	defer l.mu.RUnlock()
	out := make([]Capability, len(l.caps))
	copy(out, l.caps)
	return out
}

func (l *LocalEnvironment) SupportsCapability(c Capability) bool {
	l.mu.RLock()
	defer l.mu.RUnlock()
	for _, cap := range l.caps {
		if cap == c {
			return true
		}
	}
	return false
}

func (l *LocalEnvironment) ComputerBackend() ComputerBackend {
	l.mu.RLock()
	defer l.mu.RUnlock()
	return l.backend
}

// HasComputerCapability reports whether the local environment supports
// computer-use primitives.
func (l *LocalEnvironment) HasComputerCapability() bool {
	l.mu.RLock()
	defer l.mu.RUnlock()
	return l.backend != nil
}
