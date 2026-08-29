package tools

import (
	"context"
	"fmt"
	"strings"
	"sync"

	"github.com/ollama/ollama/agent"
	"github.com/ollama/ollama/api"
)

// Computer provides cross-platform computer interaction primitives (screenshot,
// mouse, keyboard) through the agent tool interface. It supports explicit
// environment targeting so the agent always knows WHERE it is operating.
//
// Each environment provides its own ComputerBackend. The tool resolves:
//
//	target -> environment -> that environment's computer backend
//
// This ensures that a remote target never falls back to the local machine.
type Computer struct {
	mu   sync.Mutex
	envs *agent.EnvironmentRegistry
}

// NewComputer returns a Computer backed by the environment registry.
// The registry must already contain the local environment (registered by
// NewLocalEnvironment). Returns nil if envRegistry is nil.
func NewComputer(envRegistry *agent.EnvironmentRegistry) *Computer {
	if envRegistry == nil {
		return nil
	}
	return &Computer{envs: envRegistry}
}

func (c *Computer) Name() string {
	return "computer"
}

func (c *Computer) Description() string {
	return "Interact with a computer environment through screenshots, mouse and keyboard input. " +
		"Use the optional 'target' parameter to specify an environment (defaults to 'local'). " +
		"The environment must support the 'computer' capability and have a configured backend."
}

func (c *Computer) Schema() api.ToolFunction {
	props := api.NewToolPropertiesMap()

	props.Set("action", api.ToolProperty{
		Type:        api.PropertyType{"string"},
		Description: "The computer action to perform.",
		Enum: []any{
			"screenshot",
			"click",
			"double_click",
			"move",
			"type",
			"key",
			"scroll",
		},
	})

	props.Set("target", api.ToolProperty{
		Type:        api.PropertyType{"string"},
		Description: "The environment to execute on. Defaults to 'local'. Use 'local' for the machine running Ollama.",
	})

	props.Set("x", api.ToolProperty{
		Type:        api.PropertyType{"integer"},
		Description: "X coordinate for mouse actions (screenshot pixel space). Required for click, double_click, and move.",
	})

	props.Set("y", api.ToolProperty{
		Type:        api.PropertyType{"integer"},
		Description: "Y coordinate for mouse actions (screenshot pixel space). Required for click, double_click, and move.",
	})

	props.Set("text", api.ToolProperty{
		Type:        api.PropertyType{"string"},
		Description: "Text to type. Required for the type action.",
	})

	props.Set("key", api.ToolProperty{
		Type:        api.PropertyType{"string"},
		Description: "Key name to press (e.g. \"ENTER\", \"CTRL+C\", \"ALT+TAB\"). Required for the key action.",
	})

	props.Set("dx", api.ToolProperty{
		Type:        api.PropertyType{"integer"},
		Description: "Horizontal scroll amount. Required for the scroll action. Positive scrolls right, negative scrolls left.",
	})

	props.Set("dy", api.ToolProperty{
		Type:        api.PropertyType{"integer"},
		Description: "Vertical scroll amount. Required for the scroll action. Positive scrolls down, negative scrolls up.",
	})

	return api.ToolFunction{
		Name:        c.Name(),
		Description: c.Description(),
		Parameters: api.ToolFunctionParameters{
			Type:       "object",
			Properties: props,
			Required:   []string{"action"},
		},
	}
}

// RequiresApproval returns true for all computer actions. The approval
// prompter will use the scope to distinguish observation from interaction.
func (c *Computer) RequiresApproval(map[string]any) bool {
	return true
}

// ApprovalScope returns a per-action, per-environment approval scope.
func (c *Computer) ApprovalScope(args map[string]any) string {
	action := computerActionFromArgs(args)
	target := computerTargetFromArgs(args)

	switch action {
	case "screenshot":
		return "computer:screenshot:" + target
	case "click":
		return "computer:click:" + target
	case "double_click":
		return "computer:double_click:" + target
	case "move":
		return "computer:move:" + target
	case "type":
		return "computer:type:" + target
	case "key":
		keyName := ""
		if k, ok := args["key"].(string); ok {
			keyName = strings.TrimSpace(strings.ToUpper(k))
		}
		if keyName != "" {
			return "computer:key:" + target + ":" + keyName
		}
		return "computer:key:" + target
	case "scroll":
		return "computer:scroll:" + target
	default:
		return "computer:" + target
	}
}

// Execute dispatches the requested action to the environment's computer
// backend. It resolves the target environment, obtains its backend, and
// validates that the backend is available before execution.
func (c *Computer) Execute(ctx context.Context, _ agent.ToolContext, args map[string]any) (agent.ToolResult, error) {
	action := computerActionFromArgs(args)
	if action == "" {
		return agent.ToolResult{}, fmt.Errorf("action parameter is required and must be one of: screenshot, click, double_click, move, type, key, scroll")
	}

	// Resolve environment and its computer backend
	env, backend, err := c.resolveBackend(args)
	if err != nil {
		return agent.ToolResult{}, err
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	switch action {
	case "screenshot":
		return c.executeScreenshot(ctx, env.ID(), backend)
	case "click":
		return c.executeClick(ctx, env.ID(), backend, args)
	case "double_click":
		return c.executeDoubleClick(ctx, env.ID(), backend, args)
	case "move":
		return c.executeMove(ctx, env.ID(), backend, args)
	case "type":
		return c.executeType(ctx, env.ID(), backend, args)
	case "key":
		return c.executeKey(ctx, env.ID(), backend, args)
	case "scroll":
		return c.executeScroll(ctx, env.ID(), backend, args)
	default:
		return agent.ToolResult{}, fmt.Errorf("unknown action %q; supported actions: screenshot, click, double_click, move, type, key, scroll", action)
	}
}

func (c *Computer) executeScreenshot(ctx context.Context, envID string, backend agent.ComputerBackend) (agent.ToolResult, error) {
	pngBytes, width, height, err := backend.Screenshot(ctx)
	if err != nil {
		return agent.ToolResult{}, fmt.Errorf("screen capture failed on environment %q: %w", envID, err)
	}
	if len(pngBytes) == 0 {
		return agent.ToolResult{}, fmt.Errorf("screen capture unavailable on environment %q", envID)
	}

	return agent.ToolResult{
		Content: fmt.Sprintf("Screenshot captured from environment %q (%dx%d).", envID, width, height),
		Images:  [][]byte{pngBytes},
	}, nil
}

func (c *Computer) executeClick(ctx context.Context, envID string, backend agent.ComputerBackend, args map[string]any) (agent.ToolResult, error) {
	x, y, err := requiredXY(args)
	if err != nil {
		return agent.ToolResult{}, err
	}
	if err := backend.Click(ctx, x, y); err != nil {
		return agent.ToolResult{}, err
	}
	return agent.ToolResult{Content: fmt.Sprintf("click at (%d,%d) on %q successful.", x, y, envID)}, nil
}

func (c *Computer) executeDoubleClick(ctx context.Context, envID string, backend agent.ComputerBackend, args map[string]any) (agent.ToolResult, error) {
	x, y, err := requiredXY(args)
	if err != nil {
		return agent.ToolResult{}, err
	}
	if err := backend.DoubleClick(ctx, x, y); err != nil {
		return agent.ToolResult{}, err
	}
	return agent.ToolResult{Content: fmt.Sprintf("double click at (%d,%d) on %q successful.", x, y, envID)}, nil
}

func (c *Computer) executeMove(ctx context.Context, envID string, backend agent.ComputerBackend, args map[string]any) (agent.ToolResult, error) {
	x, y, err := requiredXY(args)
	if err != nil {
		return agent.ToolResult{}, err
	}
	if err := backend.Move(ctx, x, y); err != nil {
		return agent.ToolResult{}, err
	}
	return agent.ToolResult{Content: fmt.Sprintf("move to (%d,%d) on %q successful.", x, y, envID)}, nil
}

func (c *Computer) executeType(ctx context.Context, envID string, backend agent.ComputerBackend, args map[string]any) (agent.ToolResult, error) {
	text, ok := args["text"].(string)
	if !ok || strings.TrimSpace(text) == "" {
		return agent.ToolResult{}, fmt.Errorf("text parameter is required for the type action")
	}
	if err := backend.Type(ctx, text); err != nil {
		return agent.ToolResult{}, err
	}
	return agent.ToolResult{Content: fmt.Sprintf("type on %q successful.", envID)}, nil
}

func (c *Computer) executeKey(ctx context.Context, envID string, backend agent.ComputerBackend, args map[string]any) (agent.ToolResult, error) {
	key, ok := args["key"].(string)
	if !ok || strings.TrimSpace(key) == "" {
		return agent.ToolResult{}, fmt.Errorf("key parameter is required for the key action")
	}
	if err := backend.Key(ctx, key); err != nil {
		return agent.ToolResult{}, err
	}
	return agent.ToolResult{Content: fmt.Sprintf("key %q on %q successful.", key, envID)}, nil
}

func (c *Computer) executeScroll(ctx context.Context, envID string, backend agent.ComputerBackend, args map[string]any) (agent.ToolResult, error) {
	dx, dy, err := requiredDXDY(args)
	if err != nil {
		return agent.ToolResult{}, err
	}
	if err := backend.Scroll(ctx, dx, dy); err != nil {
		return agent.ToolResult{}, err
	}
	return agent.ToolResult{Content: fmt.Sprintf("scroll (%d,%d) on %q successful.", dx, dy, envID)}, nil
}

// --- helpers ---

// resolveBackend resolves the target environment and obtains its computer
// backend. Returns an error if the target is unknown, does not support
// computer capability, or has no configured backend.
func (c *Computer) resolveBackend(args map[string]any) (agent.Environment, agent.ComputerBackend, error) {
	target := computerTargetFromArgs(args)

	env, err := c.envs.ResolveTarget(target)
	if err != nil {
		return nil, nil, fmt.Errorf("environment error: %w", err)
	}
	if !env.SupportsCapability(agent.CapComputer) {
		return nil, nil, fmt.Errorf("environment %q does not support computer capability; supported capabilities: %v", env.ID(), env.Capabilities())
	}
	backend := env.ComputerBackend()
	if backend == nil {
		return nil, nil, fmt.Errorf("computer backend unavailable for environment %q", env.ID())
	}
	return env, backend, nil
}

func computerTargetFromArgs(args map[string]any) string {
	if t, ok := args["target"].(string); ok {
		t = strings.TrimSpace(strings.ToLower(t))
		if t != "" {
			return t
		}
	}
	return "local"
}

func computerActionFromArgs(args map[string]any) string {
	if a, ok := args["action"].(string); ok {
		return strings.TrimSpace(strings.ToLower(a))
	}
	return ""
}

func requiredXY(args map[string]any) (int, int, error) {
	x, ok := args["x"]
	if !ok {
		return 0, 0, fmt.Errorf("x parameter is required for this action")
	}
	xInt, ok := x.(float64)
	if !ok {
		return 0, 0, fmt.Errorf("x parameter must be a number")
	}
	y, ok := args["y"]
	if !ok {
		return 0, 0, fmt.Errorf("y parameter is required for this action")
	}
	yInt, ok := y.(float64)
	if !ok {
		return 0, 0, fmt.Errorf("y parameter must be a number")
	}
	return int(xInt), int(yInt), nil
}

func requiredDXDY(args map[string]any) (int, int, error) {
	dx, ok := args["dx"]
	if !ok {
		return 0, 0, fmt.Errorf("dx parameter is required for the scroll action")
	}
	dxInt, ok := dx.(float64)
	if !ok {
		return 0, 0, fmt.Errorf("dx parameter must be a number")
	}
	dy, ok := args["dy"]
	if !ok {
		return 0, 0, fmt.Errorf("dy parameter is required for the scroll action")
	}
	dyInt, ok := dy.(float64)
	if !ok {
		return 0, 0, fmt.Errorf("dy parameter must be a number")
	}
	return int(dxInt), int(dyInt), nil
}
