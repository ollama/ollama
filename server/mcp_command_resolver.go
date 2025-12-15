package server

import (
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"sync"
)

// =============================================================================
// Command Resolver Interface & Default Implementation
// =============================================================================
//
// SECURITY REVIEW: This component determines which executables are launched
// for MCP servers. Changes here should be reviewed carefully.

// CommandResolverInterface defines the contract for command resolution.
// Implementations resolve command names (like "npx", "python") to actual
// executable paths, with support for fallbacks and environment overrides.
//
// This interface enables dependency injection for testing MCPClient without
// requiring actual executables to be present on the system.
type CommandResolverInterface interface {
	// ResolveCommand finds the actual executable for a command name.
	// Returns the resolved path/command or an error if not found.
	ResolveCommand(command string) (string, error)

	// ResolveForEnvironment resolves a command, checking environment
	// variable overrides first (e.g., OLLAMA_NPX_COMMAND for "npx").
	// Returns the original command as fallback if resolution fails.
	ResolveForEnvironment(command string) string
}

// CommandResolver handles resolving commands to their actual executables
// with fallback detection for different system configurations.
type CommandResolver struct {
	mu       sync.RWMutex
	resolved map[string]string
}

// Ensure CommandResolver implements CommandResolverInterface
var _ CommandResolverInterface = (*CommandResolver)(nil)

// NewCommandResolver creates a new command resolver
func NewCommandResolver() *CommandResolver {
	return &CommandResolver{
		resolved: make(map[string]string),
	}
}

// DefaultCommandResolver is the shared resolver instance for production use.
// Tests should use WithCommandResolver option instead of modifying this.
var DefaultCommandResolver = NewCommandResolver()

// ResolveCommand finds the actual executable for a given command
func (cr *CommandResolver) ResolveCommand(command string) (string, error) {
	cr.mu.RLock()
	if resolved, ok := cr.resolved[command]; ok {
		cr.mu.RUnlock()
		return resolved, nil
	}
	cr.mu.RUnlock()

	// Try to resolve the command
	var resolved string
	var err error

	switch command {
	case "npx":
		resolved, err = cr.resolveNodePackageManager()
	case "python":
		resolved, err = cr.resolvePython()
	case "node":
		resolved, err = cr.resolveNode()
	default:
		// For other commands, check if they exist as-is
		resolved, err = cr.checkCommand(command)
	}

	if err != nil {
		return "", err
	}

	// Cache the resolution
	cr.mu.Lock()
	cr.resolved[command] = resolved
	cr.mu.Unlock()

	return resolved, nil
}

// resolveNodePackageManager finds an available Node.js package manager
func (cr *CommandResolver) resolveNodePackageManager() (string, error) {
	// Priority order for package managers
	managers := []struct {
		cmd  string
		args []string
	}{
		{"npx", []string{"--version"}},
		{"pnpm", []string{"dlx", "--version"}}, // pnpm equivalent of npx
		{"yarn", []string{"dlx", "--version"}}, // yarn 2+ equivalent
		{"bunx", []string{"--version"}},        // bun equivalent
	}

	for _, mgr := range managers {
		if path, err := exec.LookPath(mgr.cmd); err == nil {
			// Verify it actually works
			cmd := exec.Command(path, mgr.args...)
			if err := cmd.Run(); err == nil {
				// For pnpm/yarn, we need to return the dlx subcommand
				if mgr.cmd == "pnpm" {
					return "pnpm dlx", nil
				} else if mgr.cmd == "yarn" {
					return "yarn dlx", nil
				}
				return mgr.cmd, nil
			}
		}
	}

	// Check if npm is available and suggest installing npx
	if _, err := exec.LookPath("npm"); err == nil {
		return "", fmt.Errorf("npx not found but npm is available - install with: npm install -g npx")
	}

	return "", fmt.Errorf("no Node.js package manager found (tried npx, pnpm, yarn, bunx)")
}

// resolvePython finds an available Python interpreter
func (cr *CommandResolver) resolvePython() (string, error) {
	// Priority order for Python interpreters
	interpreters := []string{
		"python3",   // Most Unix systems
		"python",    // Windows or virtualenv
		"python3.12", // Specific versions
		"python3.11",
		"python3.10",
		"python3.9",
		"python3.8",
	}

	for _, interp := range interpreters {
		if path, err := exec.LookPath(interp); err == nil {
			// Verify it's Python 3.8+ by checking version
			cmd := exec.Command(path, "--version")
			output, err := cmd.Output()
			if err == nil && len(output) > 0 {
				// Basic check that it's Python 3
				if string(output[:7]) == "Python " && output[7] >= '3' {
					return interp, nil
				}
			}
		}
	}

	return "", fmt.Errorf("no Python 3 interpreter found (tried python3, python, and versioned variants)")
}

// resolveNode finds the Node.js executable
func (cr *CommandResolver) resolveNode() (string, error) {
	// Try different Node.js executable names
	nodes := []string{"node", "nodejs"}
	
	for _, node := range nodes {
		if path, err := exec.LookPath(node); err == nil {
			// Verify it works
			cmd := exec.Command(path, "--version")
			if err := cmd.Run(); err == nil {
				return node, nil
			}
		}
	}

	return "", fmt.Errorf("Node.js not found (tried node, nodejs)")
}

// checkCommand checks if a command exists as-is
func (cr *CommandResolver) checkCommand(command string) (string, error) {
	if _, err := exec.LookPath(command); err == nil {
		return command, nil
	}
	return "", fmt.Errorf("command not found: %s", command)
}

// ResolveForEnvironment checks environment variables for command overrides
func (cr *CommandResolver) ResolveForEnvironment(command string) string {
	// Allow environment variable overrides
	envMap := map[string]string{
		"npx":    "OLLAMA_NPX_COMMAND",
		"python": "OLLAMA_PYTHON_COMMAND",
		"node":   "OLLAMA_NODE_COMMAND",
	}

	if envVar, ok := envMap[command]; ok {
		if override := os.Getenv(envVar); override != "" {
			// Validate override against security blocklist
			if GetSecurityConfig().IsCommandAllowed(override) {
				return override
			}
			slog.Warn("Environment override blocked by security policy", "var", envVar, "command", override)
		}
	}

	// Try standard resolution
	if resolved, err := cr.ResolveCommand(command); err == nil {
		return resolved
	}

	// Return original command as fallback
	return command
}

// NOTE: A GetSystemRequirements() method could be added here for diagnostics/status endpoints