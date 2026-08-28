package tools

import (
	"context"
	"strings"
	"testing"

	"github.com/ollama/ollama/agent"
)

func TestComputerName(t *testing.T) {
	c := &Computer{}
	if got := c.Name(); got != "computer" {
		t.Fatalf("Name() = %q, want %q", got, "computer")
	}
}

func TestComputerDescription(t *testing.T) {
	c := &Computer{}
	if got := c.Description(); got == "" {
		t.Fatal("Description() should not be empty")
	}
}

func TestComputerRequiresApproval(t *testing.T) {
	c := &Computer{}
	// All actions require approval (observation included)
	if !c.RequiresApproval(map[string]any{}) {
		t.Fatal("computer tool should always require approval")
	}
	if !c.RequiresApproval(map[string]any{"action": "screenshot"}) {
		t.Fatal("computer tool should require approval for screenshot")
	}
	if !c.RequiresApproval(map[string]any{"action": "click", "x": 100, "y": 200}) {
		t.Fatal("computer tool should require approval for click")
	}
}

func TestComputerApprovalScope(t *testing.T) {
	c := &Computer{}

	tests := []struct {
		name string
		args map[string]any
		want string
	}{
		{"screenshot local", map[string]any{"action": "screenshot"}, "computer:screenshot:local"},
		{"screenshot explicit local", map[string]any{"action": "screenshot", "target": "local"}, "computer:screenshot:local"},
		{"click local", map[string]any{"action": "click"}, "computer:click:local"},
		{"double_click", map[string]any{"action": "double_click"}, "computer:double_click:local"},
		{"move", map[string]any{"action": "move"}, "computer:move:local"},
		{"type", map[string]any{"action": "type"}, "computer:type:local"},
		{"key", map[string]any{"action": "key", "key": "ENTER"}, "computer:key:local:ENTER"},
		{"key ctrl+c", map[string]any{"action": "key", "key": "CTRL+C"}, "computer:key:local:CTRL+C"},
		{"key lowercase", map[string]any{"action": "key", "key": "enter"}, "computer:key:local:ENTER"},
		{"key no value", map[string]any{"action": "key"}, "computer:key:local"},
		{"scroll", map[string]any{"action": "scroll"}, "computer:scroll:local"},
		{"scroll remote", map[string]any{"action": "scroll", "target": "prod"}, "computer:scroll:prod"},
		{"unknown", map[string]any{"action": "unknown"}, "computer:local"},
		{"no action", map[string]any{}, "computer:local"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := c.ApprovalScope(tt.args); got != tt.want {
				t.Fatalf("ApprovalScope(%v) = %q, want %q", tt.args, got, tt.want)
			}
		})
	}
}

func TestComputerSchema(t *testing.T) {
	c := &Computer{}
	schema := c.Schema()

	if schema.Name != "computer" {
		t.Fatalf("schema.Name = %q, want %q", schema.Name, "computer")
	}
	if schema.Parameters.Type != "object" {
		t.Fatalf("schema.Parameters.Type = %q, want %q", schema.Parameters.Type, "object")
	}
	if len(schema.Parameters.Required) != 1 || schema.Parameters.Required[0] != "action" {
		t.Fatalf("schema.Parameters.Required = %v, want [action]", schema.Parameters.Required)
	}

	// Verify action enum exists
	actionProp, ok := schema.Parameters.Properties.Get("action")
	if !ok {
		t.Fatal("schema missing action property")
	}
	if len(actionProp.Enum) != 7 {
		t.Fatalf("action.Enum length = %d, want 7", len(actionProp.Enum))
	}
	expectedActions := []string{"screenshot", "click", "double_click", "move", "type", "key", "scroll"}
	for i, ea := range expectedActions {
		if actionProp.Enum[i] != ea {
			t.Fatalf("action.Enum[%d] = %q, want %q", i, actionProp.Enum[i], ea)
		}
	}

	// Verify all parameter properties exist
	for _, name := range []string{"action", "target", "x", "y", "text", "key", "dx", "dy"} {
		if _, ok := schema.Parameters.Properties.Get(name); !ok {
			t.Fatalf("schema missing %q property", name)
		}
	}
}

func TestComputerRequiresApprovalInterface(t *testing.T) {
	c := &Computer{}
	// Verify it implements ApprovalRequired
	var _ agent.ApprovalRequired = c
	// Verify it implements ScopedTool
	var _ agent.ScopedTool = c
}

func TestComputerExecuteMissingAction(t *testing.T) {
	c := NewComputer(nil)
	if c == nil {
		t.Skip("platform not supported on this OS")
	}

	_, err := c.Execute(context.Background(), agent.ToolContext{}, map[string]any{})
	if err == nil {
		t.Fatal("expected error for missing action")
	}
	if !strings.Contains(err.Error(), "action parameter is required") {
		t.Fatalf("error = %q, want 'action parameter is required'", err.Error())
	}
}

func TestComputerExecuteUnknownAction(t *testing.T) {
	c := NewComputer(nil)
	if c == nil {
		t.Skip("platform not supported on this OS")
	}

	_, err := c.Execute(context.Background(), agent.ToolContext{}, map[string]any{"action": "fly"})
	if err == nil {
		t.Fatal("expected error for unknown action")
	}
	if !strings.Contains(err.Error(), "unknown action") {
		t.Fatalf("error = %q, want 'unknown action'", err.Error())
	}
}

func TestComputerExecuteClickMissingCoords(t *testing.T) {
	c := NewComputer(nil)
	if c == nil {
		t.Skip("platform not supported on this OS")
	}

	_, err := c.Execute(context.Background(), agent.ToolContext{}, map[string]any{"action": "click"})
	if err == nil {
		t.Fatal("expected error for missing x/y")
	}
	if !strings.Contains(err.Error(), "x parameter is required") {
		t.Fatalf("error = %q, want 'x parameter is required'", err.Error())
	}
}

func TestComputerExecuteTypeMissingText(t *testing.T) {
	c := NewComputer(nil)
	if c == nil {
		t.Skip("platform not supported on this OS")
	}

	_, err := c.Execute(context.Background(), agent.ToolContext{}, map[string]any{"action": "type"})
	if err == nil {
		t.Fatal("expected error for missing text")
	}
	if !strings.Contains(err.Error(), "text parameter is required") {
		t.Fatalf("error = %q, want 'text parameter is required'", err.Error())
	}
}

func TestComputerExecuteTypeEmptyText(t *testing.T) {
	c := NewComputer(nil)
	if c == nil {
		t.Skip("platform not supported on this OS")
	}

	_, err := c.Execute(context.Background(), agent.ToolContext{}, map[string]any{"action": "type", "text": "  "})
	if err == nil {
		t.Fatal("expected error for empty text")
	}
	if !strings.Contains(err.Error(), "text parameter is required") {
		t.Fatalf("error = %q, want 'text parameter is required'", err.Error())
	}
}

func TestComputerExecuteKeyMissingKey(t *testing.T) {
	c := NewComputer(nil)
	if c == nil {
		t.Skip("platform not supported on this OS")
	}

	_, err := c.Execute(context.Background(), agent.ToolContext{}, map[string]any{"action": "key"})
	if err == nil {
		t.Fatal("expected error for missing key")
	}
	if !strings.Contains(err.Error(), "key parameter is required") {
		t.Fatalf("error = %q, want 'key parameter is required'", err.Error())
	}
}

func TestComputerExecuteScrollMissingDxDy(t *testing.T) {
	c := NewComputer(nil)
	if c == nil {
		t.Skip("platform not supported on this OS")
	}

	_, err := c.Execute(context.Background(), agent.ToolContext{}, map[string]any{"action": "scroll"})
	if err == nil {
		t.Fatal("expected error for missing dx/dy")
	}
	if !strings.Contains(err.Error(), "dx parameter is required") {
		t.Fatalf("error = %q, want 'dx parameter is required'", err.Error())
	}
}

func TestComputerExecuteMoveMissingCoords(t *testing.T) {
	c := NewComputer(nil)
	if c == nil {
		t.Skip("platform not supported on this OS")
	}

	_, err := c.Execute(context.Background(), agent.ToolContext{}, map[string]any{"action": "move"})
	if err == nil {
		t.Fatal("expected error for missing x/y")
	}
	if !strings.Contains(err.Error(), "x parameter is required") {
		t.Fatalf("error = %q, want 'x parameter is required'", err.Error())
	}
}

func TestComputerExecuteDoubleClickMissingCoords(t *testing.T) {
	c := NewComputer(nil)
	if c == nil {
		t.Skip("platform not supported on this OS")
	}

	_, err := c.Execute(context.Background(), agent.ToolContext{}, map[string]any{"action": "double_click"})
	if err == nil {
		t.Fatal("expected error for missing x/y")
	}
	if !strings.Contains(err.Error(), "x parameter is required") {
		t.Fatalf("error = %q, want 'x parameter is required'", err.Error())
	}
}

// --- Environment targeting tests ---

func TestComputerEnvironmentTargetingUnknown(t *testing.T) {
	r := agent.NewEnvironmentRegistry()
	r.Register(agent.NewLocalEnvironment(true))

	c := NewComputer(r)
	if c == nil {
		t.Skip("platform not supported on this OS")
	}

	_, err := c.Execute(context.Background(), agent.ToolContext{}, map[string]any{
		"action": "screenshot",
		"target": "nonexistent",
	})
	if err == nil {
		t.Fatal("expected error for unknown environment")
	}
	if !strings.Contains(err.Error(), "unknown environment") {
		t.Fatalf("error = %q, want 'unknown environment'", err.Error())
	}
}

func TestComputerAutoRegistersLocalEnvironment(t *testing.T) {
	r := agent.NewEnvironmentRegistry()

	c := NewComputer(r)
	if c == nil {
		t.Skip("platform not supported on this OS")
	}

	// Local should be auto-registered
	env, ok := r.Get("local")
	if !ok {
		t.Fatal("expected local environment to be auto-registered")
	}
	if !env.SupportsCapability(agent.CapComputer) {
		t.Fatal("auto-registered local environment should support computer")
	}
}

func TestComputerDoesNotOverwriteExistingLocal(t *testing.T) {
	r := agent.NewEnvironmentRegistry()
	// Register a custom local environment first
	r.Register(agent.NewLocalEnvironment(false)) // no computer

	c := NewComputer(r)
	if c == nil {
		t.Skip("platform not supported on this OS")
	}

	// Should NOT overwrite the existing local
	env, ok := r.Get("local")
	if !ok {
		t.Fatal("local environment should still exist")
	}
	if env.SupportsCapability(agent.CapComputer) {
		t.Fatal("existing local environment should not be overwritten")
	}
}

func TestComputerApprovalScopeIncludesTarget(t *testing.T) {
	c := &Computer{}

	// Screenshot on remote
	scope := c.ApprovalScope(map[string]any{"action": "screenshot", "target": "prod"})
	if scope != "computer:screenshot:prod" {
		t.Fatalf("scope = %q, want computer:screenshot:prod", scope)
	}

	// Click on local
	scope = c.ApprovalScope(map[string]any{"action": "click", "target": "local"})
	if scope != "computer:click:local" {
		t.Fatalf("scope = %q, want computer:click:local", scope)
	}

	// Key combo on remote with key name
	scope = c.ApprovalScope(map[string]any{"action": "key", "target": "dev", "key": "CTRL+S"})
	if scope != "computer:key:dev:CTRL+S" {
		t.Fatalf("scope = %q, want computer:key:dev:CTRL+S", scope)
	}
}

func TestComputerNoRegistryLocalFallback(t *testing.T) {
	c := NewComputer(nil)
	if c == nil {
		t.Skip("platform not supported on this OS")
	}

	// Without registry, "local" should work
	env, err := c.resolveEnvironment(map[string]any{"target": "local"})
	if err != nil {
		t.Fatal(err)
	}
	if env.ID() != "local" {
		t.Fatalf("env.ID() = %q, want %q", env.ID(), "local")
	}
}

func TestComputerNoRegistryRejectsRemote(t *testing.T) {
	c := NewComputer(nil)
	if c == nil {
		t.Skip("platform not supported on this OS")
	}

	_, err := c.resolveEnvironment(map[string]any{"target": "remote-server"})
	if err == nil {
		t.Fatal("expected error for remote target without registry")
	}
	if !strings.Contains(err.Error(), "environment targeting is not configured") {
		t.Fatalf("error = %q, want 'environment targeting is not configured'", err.Error())
	}
}

// --- Fake computer backend tests ---

// fakePlatform is a test double that records all actions without touching the
// real OS. It satisfies computerPlatform and is used to test the Computer
// tool's orchestration without requiring a real display.
type fakePlatform struct {
	screenshotWidth  int
	screenshotHeight int
	actions          []fakeAction
}

type fakeAction struct {
	Name string
	Args map[string]any
}

func newFakePlatform() *fakePlatform {
	return &fakePlatform{
		screenshotWidth:  1920,
		screenshotHeight: 1080,
	}
}

func (f *fakePlatform) Screenshot(_ context.Context) (*ScreenImage, error) {
	f.actions = append(f.actions, fakeAction{Name: "screenshot"})
	return &ScreenImage{
		Pixels: make([]byte, 100), // dummy PNG data
		Width:  f.screenshotWidth,
		Height: f.screenshotHeight,
	}, nil
}

func (f *fakePlatform) Click(_ context.Context, x, y int) error {
	f.actions = append(f.actions, fakeAction{
		Name: "click",
		Args: map[string]any{"x": x, "y": y},
	})
	return nil
}

func (f *fakePlatform) DoubleClick(_ context.Context, x, y int) error {
	f.actions = append(f.actions, fakeAction{
		Name: "double_click",
		Args: map[string]any{"x": x, "y": y},
	})
	return nil
}

func (f *fakePlatform) Move(_ context.Context, x, y int) error {
	f.actions = append(f.actions, fakeAction{
		Name: "move",
		Args: map[string]any{"x": x, "y": y},
	})
	return nil
}

func (f *fakePlatform) Type(_ context.Context, text string) error {
	f.actions = append(f.actions, fakeAction{
		Name: "type",
		Args: map[string]any{"text": text},
	})
	return nil
}

func (f *fakePlatform) Key(_ context.Context, key string) error {
	f.actions = append(f.actions, fakeAction{
		Name: "key",
		Args: map[string]any{"key": key},
	})
	return nil
}

func (f *fakePlatform) Scroll(_ context.Context, dx, dy int) error {
	f.actions = append(f.actions, fakeAction{
		Name: "scroll",
		Args: map[string]any{"dx": dx, "dy": dy},
	})
	return nil
}

func newFakeComputer() *Computer {
	r := agent.NewEnvironmentRegistry()
	r.Register(agent.NewLocalEnvironment(true))
	return &Computer{platform: newFakePlatform(), envs: r}
}

func TestFakeComputerScreenshot(t *testing.T) {
	fake := newFakeComputer()
	fp := fake.platform.(*fakePlatform)

	result, err := fake.Execute(context.Background(), agent.ToolContext{}, map[string]any{
		"action": "screenshot",
	})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(result.Content, "\"ok\":true") {
		t.Fatalf("screenshot result should contain ok:true, got: %q", result.Content)
	}
	if !strings.Contains(result.Content, "\"width\":1920") {
		t.Fatalf("screenshot result should contain width, got: %q", result.Content)
	}
	if !strings.Contains(result.Content, "\"height\":1080") {
		t.Fatalf("screenshot result should contain height, got: %q", result.Content)
	}
	if !strings.Contains(result.Content, "\"environment\":\"local\"") {
		t.Fatalf("screenshot result should contain environment, got: %q", result.Content)
	}
	if len(fp.actions) != 1 || fp.actions[0].Name != "screenshot" {
		t.Fatalf("expected 1 screenshot action, got: %v", fp.actions)
	}
}

func TestFakeComputerClick(t *testing.T) {
	fake := newFakeComputer()
	fp := fake.platform.(*fakePlatform)

	result, err := fake.Execute(context.Background(), agent.ToolContext{}, map[string]any{
		"action": "click",
		"x":     float64(100),
		"y":     float64(200),
	})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(result.Content, "\"ok\":true") {
		t.Fatalf("result = %q, want ok:true", result.Content)
	}
	if !strings.Contains(result.Content, "\"environment\":\"local\"") {
		t.Fatalf("result should contain environment, got: %q", result.Content)
	}
	if len(fp.actions) != 1 || fp.actions[0].Name != "click" {
		t.Fatalf("expected 1 click action, got: %v", fp.actions)
	}
	if fp.actions[0].Args["x"] != 100 || fp.actions[0].Args["y"] != 200 {
		t.Fatalf("click args = %v, want x=100 y=200", fp.actions[0].Args)
	}
}

func TestFakeComputerDoubleClick(t *testing.T) {
	fake := newFakeComputer()
	fp := fake.platform.(*fakePlatform)

	result, err := fake.Execute(context.Background(), agent.ToolContext{}, map[string]any{
		"action": "double_click",
		"x":     float64(50),
		"y":     float64(75),
	})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(result.Content, "\"ok\":true") {
		t.Fatalf("result = %q, want ok:true", result.Content)
	}
	if len(fp.actions) != 1 || fp.actions[0].Name != "double_click" {
		t.Fatalf("expected 1 double_click action, got: %v", fp.actions)
	}
}

func TestFakeComputerMove(t *testing.T) {
	fake := newFakeComputer()
	fp := fake.platform.(*fakePlatform)

	result, err := fake.Execute(context.Background(), agent.ToolContext{}, map[string]any{
		"action": "move",
		"x":     float64(300),
		"y":     float64(400),
	})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(result.Content, "\"ok\":true") {
		t.Fatalf("result = %q, want ok:true", result.Content)
	}
	if len(fp.actions) != 1 || fp.actions[0].Name != "move" {
		t.Fatalf("expected 1 move action, got: %v", fp.actions)
	}
}

func TestFakeComputerType(t *testing.T) {
	fake := newFakeComputer()
	fp := fake.platform.(*fakePlatform)

	result, err := fake.Execute(context.Background(), agent.ToolContext{}, map[string]any{
		"action": "type",
		"text":   "hello world",
	})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(result.Content, "\"ok\":true") {
		t.Fatalf("result = %q, want ok:true", result.Content)
	}
	if len(fp.actions) != 1 || fp.actions[0].Name != "type" {
		t.Fatalf("expected 1 type action, got: %v", fp.actions)
	}
	if fp.actions[0].Args["text"] != "hello world" {
		t.Fatalf("type args = %v, want text='hello world'", fp.actions[0].Args)
	}
}

func TestFakeComputerKey(t *testing.T) {
	fake := newFakeComputer()
	fp := fake.platform.(*fakePlatform)

	result, err := fake.Execute(context.Background(), agent.ToolContext{}, map[string]any{
		"action": "key",
		"key":    "ENTER",
	})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(result.Content, "\"ok\":true") {
		t.Fatalf("result = %q, want ok:true", result.Content)
	}
	if len(fp.actions) != 1 || fp.actions[0].Name != "key" {
		t.Fatalf("expected 1 key action, got: %v", fp.actions)
	}
	if fp.actions[0].Args["key"] != "ENTER" {
		t.Fatalf("key args = %v, want key='ENTER'", fp.actions[0].Args)
	}
}

func TestFakeComputerScroll(t *testing.T) {
	fake := newFakeComputer()
	fp := fake.platform.(*fakePlatform)

	result, err := fake.Execute(context.Background(), agent.ToolContext{}, map[string]any{
		"action": "scroll",
		"dx":     float64(0),
		"dy":     float64(3),
	})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(result.Content, "\"ok\":true") {
		t.Fatalf("result = %q, want ok:true", result.Content)
	}
	if len(fp.actions) != 1 || fp.actions[0].Name != "scroll" {
		t.Fatalf("expected 1 scroll action, got: %v", fp.actions)
	}
}

func TestFakeComputerMultiStepSequence(t *testing.T) {
	// Simulate a typical agent workflow: screenshot → click → type → key → screenshot
	fake := newFakeComputer()
	fp := fake.platform.(*fakePlatform)
	ctx := context.Background()
	tc := agent.ToolContext{}

	// Step 1: screenshot
	r, err := fake.Execute(ctx, tc, map[string]any{"action": "screenshot"})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(r.Content, "width") {
		t.Fatal("screenshot should return width")
	}

	// Step 2: click on address bar
	r, err = fake.Execute(ctx, tc, map[string]any{"action": "click", "x": float64(640), "y": float64(52)})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(r.Content, "\"ok\":true") {
		t.Fatalf("click result = %q", r.Content)
	}

	// Step 3: type URL
	r, err = fake.Execute(ctx, tc, map[string]any{"action": "type", "text": "https://github.com/ollama/ollama"})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(r.Content, "\"ok\":true") {
		t.Fatalf("type result = %q", r.Content)
	}

	// Step 4: press Enter
	r, err = fake.Execute(ctx, tc, map[string]any{"action": "key", "key": "ENTER"})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(r.Content, "\"ok\":true") {
		t.Fatalf("key result = %q", r.Content)
	}

	// Step 5: final screenshot
	r, err = fake.Execute(ctx, tc, map[string]any{"action": "screenshot"})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(r.Content, "width") {
		t.Fatal("final screenshot should return width")
	}

	// Verify the action sequence
	expectedActions := []string{"screenshot", "click", "type", "key", "screenshot"}
	if len(fp.actions) != len(expectedActions) {
		t.Fatalf("expected %d actions, got %d: %v", len(expectedActions), len(fp.actions), fp.actions)
	}
	for i, ea := range expectedActions {
		if fp.actions[i].Name != ea {
			t.Fatalf("action[%d] = %q, want %q", i, fp.actions[i].Name, ea)
		}
	}
}

func TestFakeComputerContextCancellation(t *testing.T) {
	fake := newFakeComputer()
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel immediately

	_, err := fake.Execute(ctx, agent.ToolContext{}, map[string]any{"action": "screenshot"})
	if err == nil {
		t.Fatal("expected error from cancelled context")
	}
}

func TestFakeComputerEnvironmentInResult(t *testing.T) {
	fake := newFakeComputer()
	fp := fake.platform.(*fakePlatform)

	// All results should include the environment
	actions := []map[string]any{
		{"action": "screenshot"},
		{"action": "click", "x": float64(1), "y": float64(1)},
		{"action": "double_click", "x": float64(1), "y": float64(1)},
		{"action": "move", "x": float64(1), "y": float64(1)},
		{"action": "type", "text": "x"},
		{"action": "key", "key": "A"},
		{"action": "scroll", "dx": float64(0), "dy": float64(1)},
	}

	for _, args := range actions {
		fp.actions = nil
		result, err := fake.Execute(context.Background(), agent.ToolContext{}, args)
		if err != nil {
			t.Fatalf("action %v error: %v", args["action"], err)
		}
		if !strings.Contains(result.Content, "\"environment\":\"local\"") {
			t.Fatalf("action %v result missing environment, got: %q", args["action"], result.Content)
		}
	}
}

func TestFakeComputerEnvTargeting(t *testing.T) {
	r := agent.NewEnvironmentRegistry()
	r.Register(agent.NewLocalEnvironment(true))
	r.Register(&fakeRemoteEnv{id: "remote-desktop", caps: []agent.Capability{agent.CapComputer}})

	fake := &Computer{platform: newFakePlatform(), envs: r}
	fp := fake.platform.(*fakePlatform)

	// Execute on remote
	result, err := fake.Execute(context.Background(), agent.ToolContext{}, map[string]any{
		"action": "screenshot",
		"target": "remote-desktop",
	})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(result.Content, "\"environment\":\"remote-desktop\"") {
		t.Fatalf("result should target remote-desktop, got: %q", result.Content)
	}
	_ = fp
}

func TestFakeComputerEnvCapabilityCheck(t *testing.T) {
	r := agent.NewEnvironmentRegistry()
	r.Register(&fakeRemoteEnv{id: "no-computer", caps: []agent.Capability{agent.CapShell}})

	fake := &Computer{platform: newFakePlatform(), envs: r}

	_, err := fake.Execute(context.Background(), agent.ToolContext{}, map[string]any{
		"action": "screenshot",
		"target": "no-computer",
	})
	if err == nil {
		t.Fatal("expected error for environment without computer capability")
	}
	if !strings.Contains(err.Error(), "does not support computer capability") {
		t.Fatalf("error = %q, want 'does not support computer capability'", err.Error())
	}
}

// fakeRemoteEnv is a minimal mock environment for cross-environment tests.
type fakeRemoteEnv struct {
	id   string
	caps []agent.Capability
}

func (f *fakeRemoteEnv) ID() string              { return f.id }
func (f *fakeRemoteEnv) Type() agent.EnvironmentType { return agent.EnvironmentRemote }
func (f *fakeRemoteEnv) Capabilities() []agent.Capability {
	out := make([]agent.Capability, len(f.caps))
	copy(out, f.caps)
	return out
}
func (f *fakeRemoteEnv) SupportsCapability(c agent.Capability) bool {
	for _, cap := range f.caps {
		if cap == c {
			return true
		}
	}
	return false
}

// Verify that the fake platform satisfies the interface at compile time.
var _ computerPlatform = (*fakePlatform)(nil)
