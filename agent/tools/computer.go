package tools

import (
	"context"
	"encoding/base64"
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
// The tool exposes a single consolidated schema with an "action" discriminator
// and an optional "target" parameter. When target is omitted, the local
// environment is used by default.
type Computer struct {
	mu       sync.Mutex
	platform computerPlatform
	envs     *agent.EnvironmentRegistry
}

// NewComputer returns a Computer backed by the current OS platform
// implementation. Returns nil if the platform is not supported.
// The envRegistry parameter provides environment targeting support; pass nil
// to use local-only mode (backward compatible).
func NewComputer(envRegistry *agent.EnvironmentRegistry) *Computer {
	p := newPlatform()
	if p == nil {
		return nil
	}
	c := &Computer{
		platform: p,
		envs:     envRegistry,
	}
	// Auto-register the local environment if a registry is provided
	// and "local" is not already registered.
	if envRegistry != nil {
		if _, ok := envRegistry.Get("local"); !ok {
			envRegistry.Register(agent.NewLocalEnvironment(true))
		}
	}
	return c
}

func (c *Computer) Name() string {
	return "computer"
}

func (c *Computer) Description() string {
	return "Interact with the local computer through screenshots, mouse and keyboard input. " +
		"Use the optional 'target' parameter to specify an environment (defaults to 'local'). " +
		"The environment must support the 'computer' capability."
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
// The scope encodes: "computer:<action>:<target>" for interaction actions
// and "computer:screenshot:<target>" for observation.
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

// Execute dispatches the requested action to the platform implementation.
// It validates the environment target and capability before execution.
// Only one computer action can execute at a time per Computer instance to
// prevent interleaving of mouse and keyboard events.
func (c *Computer) Execute(ctx context.Context, _ agent.ToolContext, args map[string]any) (agent.ToolResult, error) {
	action := computerActionFromArgs(args)
	if action == "" {
		return agent.ToolResult{}, fmt.Errorf("action parameter is required and must be one of: screenshot, click, double_click, move, type, key, scroll")
	}

	// Resolve and validate environment target
	env, err := c.resolveEnvironment(args)
	if err != nil {
		return agent.ToolResult{}, err
	}
	if !env.SupportsCapability(agent.CapComputer) {
		return agent.ToolResult{}, fmt.Errorf("environment %q does not support computer capability; supported capabilities: %v", env.ID(), env.Capabilities())
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	switch action {
	case "screenshot":
		return c.executeScreenshot(ctx, env)
	case "click":
		return c.executeClick(ctx, env, args)
	case "double_click":
		return c.executeDoubleClick(ctx, env, args)
	case "move":
		return c.executeMove(ctx, env, args)
	case "type":
		return c.executeType(ctx, env, args)
	case "key":
		return c.executeKey(ctx, env, args)
	case "scroll":
		return c.executeScroll(ctx, env, args)
	default:
		return agent.ToolResult{}, fmt.Errorf("unknown action %q; supported actions: screenshot, click, double_click, move, type, key, scroll", action)
	}
}

func (c *Computer) executeScreenshot(ctx context.Context, env agent.Environment) (agent.ToolResult, error) {
	img, err := c.platform.Screenshot(ctx)
	if err != nil {
		return agent.ToolResult{}, fmt.Errorf("screen capture failed on environment %q: %w", env.ID(), err)
	}
	if img == nil {
		return agent.ToolResult{}, fmt.Errorf("screen capture unavailable on environment %q", env.ID())
	}

	b64 := base64.StdEncoding.EncodeToString(img.Pixels)
	return agent.ToolResult{
		Content: fmt.Sprintf(`{"ok":true,"environment":"%s","width":%d,"height":%d,"image":"data:image/png;base64,%s"}`, env.ID(), img.Width, img.Height, b64),
	}, nil
}

func (c *Computer) executeClick(ctx context.Context, env agent.Environment, args map[string]any) (agent.ToolResult, error) {
	x, y, err := requiredXY(args)
	if err != nil {
		return agent.ToolResult{}, err
	}
	if err := c.platform.Click(ctx, x, y); err != nil {
		return agent.ToolResult{}, err
	}
	return agent.ToolResult{Content: fmt.Sprintf(`{"ok":true,"environment":"%s","action":"click","x":%d,"y":%d}`, env.ID(), x, y)}, nil
}

func (c *Computer) executeDoubleClick(ctx context.Context, env agent.Environment, args map[string]any) (agent.ToolResult, error) {
	x, y, err := requiredXY(args)
	if err != nil {
		return agent.ToolResult{}, err
	}
	if err := c.platform.DoubleClick(ctx, x, y); err != nil {
		return agent.ToolResult{}, err
	}
	return agent.ToolResult{Content: fmt.Sprintf(`{"ok":true,"environment":"%s","action":"double_click","x":%d,"y":%d}`, env.ID(), x, y)}, nil
}

func (c *Computer) executeMove(ctx context.Context, env agent.Environment, args map[string]any) (agent.ToolResult, error) {
	x, y, err := requiredXY(args)
	if err != nil {
		return agent.ToolResult{}, err
	}
	if err := c.platform.Move(ctx, x, y); err != nil {
		return agent.ToolResult{}, err
	}
	return agent.ToolResult{Content: fmt.Sprintf(`{"ok":true,"environment":"%s","action":"move","x":%d,"y":%d}`, env.ID(), x, y)}, nil
}

func (c *Computer) executeType(ctx context.Context, env agent.Environment, args map[string]any) (agent.ToolResult, error) {
	text, ok := args["text"].(string)
	if !ok || strings.TrimSpace(text) == "" {
		return agent.ToolResult{}, fmt.Errorf("text parameter is required for the type action")
	}
	if err := c.platform.Type(ctx, text); err != nil {
		return agent.ToolResult{}, err
	}
	return agent.ToolResult{Content: fmt.Sprintf(`{"ok":true,"environment":"%s","action":"type"}`, env.ID())}, nil
}

func (c *Computer) executeKey(ctx context.Context, env agent.Environment, args map[string]any) (agent.ToolResult, error) {
	key, ok := args["key"].(string)
	if !ok || strings.TrimSpace(key) == "" {
		return agent.ToolResult{}, fmt.Errorf("key parameter is required for the key action")
	}
	if err := c.platform.Key(ctx, key); err != nil {
		return agent.ToolResult{}, err
	}
	return agent.ToolResult{Content: fmt.Sprintf(`{"ok":true,"environment":"%s","action":"key","key":"%s"}`, env.ID(), key)}, nil
}

func (c *Computer) executeScroll(ctx context.Context, env agent.Environment, args map[string]any) (agent.ToolResult, error) {
	dx, dy, err := requiredDXDY(args)
	if err != nil {
		return agent.ToolResult{}, err
	}
	if err := c.platform.Scroll(ctx, dx, dy); err != nil {
		return agent.ToolResult{}, err
	}
	return agent.ToolResult{Content: fmt.Sprintf(`{"ok":true,"environment":"%s","action":"scroll","dx":%d,"dy":%d}`, env.ID(), dx, dy)}, nil
}

// --- helpers ---

func (c *Computer) resolveEnvironment(args map[string]any) (agent.Environment, error) {
	target := computerTargetFromArgs(args)

	if c.envs != nil {
		env, err := c.envs.ResolveTarget(target)
		if err != nil {
			return nil, fmt.Errorf("environment error: %w", err)
		}
		return env, nil
	}

	// Fallback: no registry, only "local" is supported
	if target != "" && target != "local" {
		return nil, fmt.Errorf("environment targeting is not configured; only 'local' is available")
	}
	return agent.NewLocalEnvironment(true), nil
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

// --- platform abstraction ---

// ScreenImage holds a screenshot as raw RGBA pixels.
type ScreenImage struct {
	Pixels []byte
	Width  int
	Height int
}

// computerPlatform is the interface that OS-specific implementations must
// satisfy. The agent layer never calls platform methods directly; it always
// goes through Computer.Execute which holds the session lock.
type computerPlatform interface {
	Screenshot(ctx context.Context) (*ScreenImage, error)
	Click(ctx context.Context, x, y int) error
	DoubleClick(ctx context.Context, x, y int) error
	Move(ctx context.Context, x, y int) error
	Type(ctx context.Context, text string) error
	Key(ctx context.Context, key string) error
	Scroll(ctx context.Context, dx, dy int) error
}
