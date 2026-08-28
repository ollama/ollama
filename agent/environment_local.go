package agent

import (
	"sync"
)

// LocalEnvironment represents the machine running Ollama. It is the default
// environment and supports shell, files, and computer capabilities.
type LocalEnvironment struct {
	caps        []Capability
	mu          sync.RWMutex
	computerCap bool // whether computer capability is available
}

// NewLocalEnvironment creates a new local environment. The computerCapability
// parameter indicates whether the platform supports computer-use primitives
// (screenshot, mouse, keyboard). When false, only shell and file capabilities
// are advertised.
func NewLocalEnvironment(computerCapability bool) *LocalEnvironment {
	caps := []Capability{CapShell, CapFiles}
	if computerCapability {
		caps = append(caps, CapComputer)
	}
	return &LocalEnvironment{
		caps:        caps,
		computerCap: computerCapability,
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

// HasComputerCapability reports whether the local environment supports
// computer-use primitives.
func (l *LocalEnvironment) HasComputerCapability() bool {
	l.mu.RLock()
	defer l.mu.RUnlock()
	return l.computerCap
}
