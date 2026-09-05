package tools

import (
	"context"
	"strings"
	"testing"

	"github.com/ollama/ollama/agent"
)

// --- interface compliance ---

var _ agent.ComputerBackend = (*fakeComputerBackend)(nil)

// --- Tool metadata tests ---

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
	if !c.RequiresApproval(map[string]any{}) {
		t.Fatal("computer tool should always require approval")
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
		{"click local", map[string]any{"action": "click"}, "computer:click:local"},
		{"key local", map[string]any{"action": "key", "key": "ENTER"}, "computer:key:local:ENTER"},
		{"scroll remote", map[string]any{"action": "scroll", "target": "prod"}, "computer:scroll:prod"},
		{"click remote", map[string]any{"action": "click", "target": "remote-desktop"}, "computer:click:remote-desktop"},
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
	for _, name := range []string{"action", "target", "x", "y", "text", "key", "dx", "dy"} {
		if _, ok := schema.Parameters.Properties.Get(name); !ok {
			t.Fatalf("schema missing %q property", name)
		}
	}
}

// --- Fake computer backend ---

type fakeComputerBackend struct {
	actions     []fakeAction
	screenshotW int
	screenshotH int
}

type fakeAction struct {
	Name string
	Args map[string]any
}

func newFakeBackend() *fakeComputerBackend {
	return &fakeComputerBackend{screenshotW: 1920, screenshotH: 1080}
}

func (f *fakeComputerBackend) Screenshot(_ context.Context) ([]byte, int, int, error) {
	f.actions = append(f.actions, fakeAction{Name: "screenshot"})
	// Return minimal valid PNG (1x1 white pixel)
	png := []byte{
		0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, // PNG signature
		0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52, // IHDR
		0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53, 0xde,
		0x00, 0x00, 0x00, 0x0c, 0x49, 0x44, 0x41, 0x54, // IDAT
		0x08, 0xd7, 0x63, 0xf8, 0xcf, 0xc0, 0x00, 0x00, 0x00, 0x02, 0x00, 0x01, 0xe2, 0x21, 0xbc, 0x33,
		0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4e, 0x44, // IEND
		0xae, 0x42, 0x60, 0x82,
	}
	return png, f.screenshotW, f.screenshotH, nil
}

func (f *fakeComputerBackend) Click(_ context.Context, x, y int) error {
	f.actions = append(f.actions, fakeAction{Name: "click", Args: map[string]any{"x": x, "y": y}})
	return nil
}

func (f *fakeComputerBackend) DoubleClick(_ context.Context, x, y int) error {
	f.actions = append(f.actions, fakeAction{Name: "double_click", Args: map[string]any{"x": x, "y": y}})
	return nil
}

func (f *fakeComputerBackend) Move(_ context.Context, x, y int) error {
	f.actions = append(f.actions, fakeAction{Name: "move", Args: map[string]any{"x": x, "y": y}})
	return nil
}

func (f *fakeComputerBackend) Type(_ context.Context, text string) error {
	f.actions = append(f.actions, fakeAction{Name: "type", Args: map[string]any{"text": text}})
	return nil
}

func (f *fakeComputerBackend) Key(_ context.Context, key string) error {
	f.actions = append(f.actions, fakeAction{Name: "key", Args: map[string]any{"key": key}})
	return nil
}

func (f *fakeComputerBackend) Scroll(_ context.Context, dx, dy int) error {
	f.actions = append(f.actions, fakeAction{Name: "scroll", Args: map[string]any{"dx": dx, "dy": dy}})
	return nil
}

// --- Environment-aware registry helpers ---

func newTestRegistry(localBackend agent.ComputerBackend) *agent.EnvironmentRegistry {
	r := agent.NewEnvironmentRegistry()
	r.Register(agent.NewLocalEnvironment(localBackend))
	return r
}

func newTestRegistryWithRemote(localBackend agent.ComputerBackend) *agent.EnvironmentRegistry {
	r := newTestRegistry(localBackend)
	// Register a remote environment that has CapComputer but NO backend
	remote := &fakeEnvironment{
		id:   "remote-desktop",
		env:  agent.EnvironmentRemote,
		caps: []agent.Capability{agent.CapShell, agent.CapComputer},
	}
	r.Register(remote)
	return r
}

// fakeEnvironment is a minimal mock for cross-environment tests.
type fakeEnvironment struct {
	id     string
	env    agent.EnvironmentType
	caps   []agent.Capability
	backend agent.ComputerBackend
}

func (f *fakeEnvironment) ID() string              { return f.id }
func (f *fakeEnvironment) Type() agent.EnvironmentType { return f.env }
func (f *fakeEnvironment) Capabilities() []agent.Capability {
	out := make([]agent.Capability, len(f.caps))
	copy(out, f.caps)
	return out
}
func (f *fakeEnvironment) SupportsCapability(c agent.Capability) bool {
	for _, cap := range f.caps {
		if cap == c {
			return true
		}
	}
	return false
}
func (f *fakeEnvironment) ComputerBackend() agent.ComputerBackend {
	return f.backend
}

// --- Core execution tests ---

func TestFakeComputerScreenshotReturnsImages(t *testing.T) {
	backend := newFakeBackend()
	r := newTestRegistry(backend)
	c := NewComputer(r)
	if c == nil {
		t.Skip("platform not supported")
	}

	result, err := c.Execute(context.Background(), agent.ToolContext{}, map[string]any{"action": "screenshot"})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Images) != 1 {
		t.Fatalf("expected 1 image, got %d", len(result.Images))
	}
	if len(result.Images[0]) == 0 {
		t.Fatal("expected non-empty image data")
	}
	if !strings.Contains(result.Content, "Screenshot captured") {
		t.Fatalf("content should describe screenshot, got: %q", result.Content)
	}
	// Content must NOT contain base64
	if strings.Contains(result.Content, "base64") {
		t.Fatalf("content must not contain base64, got: %q", result.Content)
	}
}

func TestFakeComputerClick(t *testing.T) {
	backend := newFakeBackend()
	r := newTestRegistry(backend)
	c := NewComputer(r)
	if c == nil {
		t.Skip("platform not supported")
	}

	result, err := c.Execute(context.Background(), agent.ToolContext{}, map[string]any{
		"action": "click", "x": float64(100), "y": float64(200),
	})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(result.Content, "click") {
		t.Fatalf("content = %q", result.Content)
	}
	if len(backend.actions) != 1 || backend.actions[0].Name != "click" {
		t.Fatalf("expected click action, got: %v", backend.actions)
	}
}

func TestFakeComputerDoubleclick(t *testing.T) {
	backend := newFakeBackend()
	r := newTestRegistry(backend)
	c := NewComputer(r)
	if c == nil {
		t.Skip("platform not supported")
	}

	_, err := c.Execute(context.Background(), agent.ToolContext{}, map[string]any{
		"action": "double_click", "x": float64(50), "y": float64(75),
	})
	if err != nil {
		t.Fatal(err)
	}
	if backend.actions[0].Name != "double_click" {
		t.Fatalf("expected double_click, got: %v", backend.actions[0].Name)
	}
}

func TestFakeComputerMove(t *testing.T) {
	backend := newFakeBackend()
	r := newTestRegistry(backend)
	c := NewComputer(r)
	if c == nil {
		t.Skip("platform not supported")
	}

	_, err := c.Execute(context.Background(), agent.ToolContext{}, map[string]any{
		"action": "move", "x": float64(300), "y": float64(400),
	})
	if err != nil {
		t.Fatal(err)
	}
	if backend.actions[0].Name != "move" {
		t.Fatalf("expected move, got: %v", backend.actions[0].Name)
	}
}

func TestFakeComputerType(t *testing.T) {
	backend := newFakeBackend()
	r := newTestRegistry(backend)
	c := NewComputer(r)
	if c == nil {
		t.Skip("platform not supported")
	}

	_, err := c.Execute(context.Background(), agent.ToolContext{}, map[string]any{
		"action": "type", "text": "hello world",
	})
	if err != nil {
		t.Fatal(err)
	}
	if backend.actions[0].Args["text"] != "hello world" {
		t.Fatalf("type args = %v", backend.actions[0].Args)
	}
}

func TestFakeComputerKey(t *testing.T) {
	backend := newFakeBackend()
	r := newTestRegistry(backend)
	c := NewComputer(r)
	if c == nil {
		t.Skip("platform not supported")
	}

	_, err := c.Execute(context.Background(), agent.ToolContext{}, map[string]any{
		"action": "key", "key": "ENTER",
	})
	if err != nil {
		t.Fatal(err)
	}
	if backend.actions[0].Args["key"] != "ENTER" {
		t.Fatalf("key args = %v", backend.actions[0].Args)
	}
}

func TestFakeComputerScroll(t *testing.T) {
	backend := newFakeBackend()
	r := newTestRegistry(backend)
	c := NewComputer(r)
	if c == nil {
		t.Skip("platform not supported")
	}

	_, err := c.Execute(context.Background(), agent.ToolContext{}, map[string]any{
		"action": "scroll", "dx": float64(0), "dy": float64(3),
	})
	if err != nil {
		t.Fatal(err)
	}
	if backend.actions[0].Name != "scroll" {
		t.Fatalf("expected scroll, got: %v", backend.actions[0].Name)
	}
}

// --- Validation tests ---

func TestComputerExecuteMissingAction(t *testing.T) {
	backend := newFakeBackend()
	r := newTestRegistry(backend)
	c := NewComputer(r)
	if c == nil {
		t.Skip("platform not supported")
	}

	_, err := c.Execute(context.Background(), agent.ToolContext{}, map[string]any{})
	if err == nil {
		t.Fatal("expected error for missing action")
	}
	if !strings.Contains(err.Error(), "action parameter is required") {
		t.Fatalf("error = %q", err.Error())
	}
}

func TestComputerExecuteUnknownAction(t *testing.T) {
	backend := newFakeBackend()
	r := newTestRegistry(backend)
	c := NewComputer(r)
	if c == nil {
		t.Skip("platform not supported")
	}

	_, err := c.Execute(context.Background(), agent.ToolContext{}, map[string]any{"action": "fly"})
	if err == nil {
		t.Fatal("expected error for unknown action")
	}
}

func TestComputerExecuteClickMissingCoords(t *testing.T) {
	backend := newFakeBackend()
	r := newTestRegistry(backend)
	c := NewComputer(r)
	if c == nil {
		t.Skip("platform not supported")
	}

	_, err := c.Execute(context.Background(), agent.ToolContext{}, map[string]any{"action": "click"})
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "x parameter is required") {
		t.Fatalf("error = %q", err.Error())
	}
}

func TestComputerExecuteTypeMissingText(t *testing.T) {
	backend := newFakeBackend()
	r := newTestRegistry(backend)
	c := NewComputer(r)
	if c == nil {
		t.Skip("platform not supported")
	}

	_, err := c.Execute(context.Background(), agent.ToolContext{}, map[string]any{"action": "type"})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestComputerExecuteKeyMissingKey(t *testing.T) {
	backend := newFakeBackend()
	r := newTestRegistry(backend)
	c := NewComputer(r)
	if c == nil {
		t.Skip("platform not supported")
	}

	_, err := c.Execute(context.Background(), agent.ToolContext{}, map[string]any{"action": "key"})
	if err == nil {
		t.Fatal("expected error")
	}
}

func TestComputerExecuteScrollMissingDxDy(t *testing.T) {
	backend := newFakeBackend()
	r := newTestRegistry(backend)
	c := NewComputer(r)
	if c == nil {
		t.Skip("platform not supported")
	}

	_, err := c.Execute(context.Background(), agent.ToolContext{}, map[string]any{"action": "scroll"})
	if err == nil {
		t.Fatal("expected error")
	}
}

// --- Environment routing tests (P0-1) ---

func TestEnvironmentBackendRoutingLocal(t *testing.T) {
	backend := newFakeBackend()
	r := newTestRegistry(backend)
	c := NewComputer(r)
	if c == nil {
		t.Skip("platform not supported")
	}

	_, err := c.Execute(context.Background(), agent.ToolContext{}, map[string]any{
		"action": "click", "x": float64(1), "y": float64(1), "target": "local",
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(backend.actions) != 1 {
		t.Fatalf("expected 1 action on local backend, got %d", len(backend.actions))
	}
}

func TestEnvironmentBackendRoutingRemoteWithBackend(t *testing.T) {
	localBackend := newFakeBackend()
	remoteBackend := newFakeBackend()
	r := newTestRegistry(localBackend)
	r.Register(&fakeEnvironment{
		id:      "remote-desktop",
		env:     agent.EnvironmentRemote,
		caps:    []agent.Capability{agent.CapComputer},
		backend: remoteBackend,
	})
	c := NewComputer(r)
	if c == nil {
		t.Skip("platform not supported")
	}

	_, err := c.Execute(context.Background(), agent.ToolContext{}, map[string]any{
		"action": "click", "x": float64(10), "y": float64(20), "target": "remote-desktop",
	})
	if err != nil {
		t.Fatal(err)
	}
	// Must execute on remote, not local
	if len(localBackend.actions) != 0 {
		t.Fatal("local backend should NOT have been called")
	}
	if len(remoteBackend.actions) != 1 {
		t.Fatalf("remote backend should have 1 action, got %d", len(remoteBackend.actions))
	}
}

func TestEnvironmentBackendRoutingRemoteNoBackend(t *testing.T) {
	localBackend := newFakeBackend()
	r := newTestRegistryWithRemote(localBackend)
	c := NewComputer(r)
	if c == nil {
		t.Skip("platform not supported")
	}

	_, err := c.Execute(context.Background(), agent.ToolContext{}, map[string]any{
		"action": "screenshot", "target": "remote-desktop",
	})
	if err == nil {
		t.Fatal("expected error for remote without backend")
	}
	if !strings.Contains(err.Error(), "computer backend unavailable") {
		t.Fatalf("error = %q, want 'computer backend unavailable'", err.Error())
	}
	// Local backend must NOT have been called
	if len(localBackend.actions) != 0 {
		t.Fatal("local backend should NOT have been called")
	}
}

func TestEnvironmentUnknownTarget(t *testing.T) {
	backend := newFakeBackend()
	r := newTestRegistry(backend)
	c := NewComputer(r)
	if c == nil {
		t.Skip("platform not supported")
	}

	_, err := c.Execute(context.Background(), agent.ToolContext{}, map[string]any{
		"action": "screenshot", "target": "nonexistent",
	})
	if err == nil {
		t.Fatal("expected error for unknown target")
	}
	if !strings.Contains(err.Error(), "unknown environment") {
		t.Fatalf("error = %q", err.Error())
	}
}

func TestEnvironmentNoCapability(t *testing.T) {
	r := agent.NewEnvironmentRegistry()
	r.Register(&fakeEnvironment{
		id:   "shell-only",
		env:  agent.EnvironmentRemote,
		caps: []agent.Capability{agent.CapShell},
	})
	c := NewComputer(r)
	if c == nil {
		t.Skip("platform not supported")
	}

	_, err := c.Execute(context.Background(), agent.ToolContext{}, map[string]any{
		"action": "screenshot", "target": "shell-only",
	})
	if err == nil {
		t.Fatal("expected error for environment without computer capability")
	}
	if !strings.Contains(err.Error(), "does not support computer capability") {
		t.Fatalf("error = %q", err.Error())
	}
}

func TestEnvironmentCrossBoundarySafety(t *testing.T) {
	localBackend := newFakeBackend()
	remoteBackend := newFakeBackend()
	r := newTestRegistry(localBackend)
	r.Register(&fakeEnvironment{
		id:      "prod",
		env:     agent.EnvironmentRemote,
		caps:    []agent.Capability{agent.CapComputer},
		backend: remoteBackend,
	})
	c := NewComputer(r)
	if c == nil {
		t.Skip("platform not supported")
	}

	// Execute on remote
	_, err := c.Execute(context.Background(), agent.ToolContext{}, map[string]any{
		"action": "type", "text": "rm -rf /", "target": "prod",
	})
	if err != nil {
		t.Fatal(err)
	}
	// Verify it went to remote, not local
	if len(localBackend.actions) != 0 {
		t.Fatal("CRITICAL: local backend was called for remote target!")
	}
	if len(remoteBackend.actions) != 1 || remoteBackend.actions[0].Args["text"] != "rm -rf /" {
		t.Fatalf("remote backend did not receive correct action: %v", remoteBackend.actions)
	}
}

// --- Multi-step workflow test ---

func TestFakeComputerMultiStepSequence(t *testing.T) {
	backend := newFakeBackend()
	r := newTestRegistry(backend)
	c := NewComputer(r)
	if c == nil {
		t.Skip("platform not supported")
	}
	ctx := context.Background()
	tc := agent.ToolContext{}

	steps := []map[string]any{
		{"action": "screenshot"},
		{"action": "click", "x": float64(640), "y": float64(52)},
		{"action": "type", "text": "https://github.com/ollama/ollama"},
		{"action": "key", "key": "ENTER"},
		{"action": "screenshot"},
	}
	expectedActions := []string{"screenshot", "click", "type", "key", "screenshot"}

	for i, args := range steps {
		result, err := c.Execute(ctx, tc, args)
		if err != nil {
			t.Fatalf("step %d (%v) error: %v", i, args["action"], err)
		}
		if backend.actions[i].Name != expectedActions[i] {
			t.Fatalf("step %d: got action %q, want %q", i, backend.actions[i].Name, expectedActions[i])
		}
		// Screenshot should have images
		if args["action"] == "screenshot" {
			if len(result.Images) != 1 {
				t.Fatalf("step %d: screenshot should return 1 image, got %d", i, len(result.Images))
			}
		}
	}
}

// --- Nil/edge case tests ---

func TestNewComputerNilRegistry(t *testing.T) {
	c := NewComputer(nil)
	if c != nil {
		t.Fatal("NewComputer(nil) should return nil")
	}
}

func TestComputerContextCancellation(t *testing.T) {
	backend := newFakeBackend()
	r := newTestRegistry(backend)
	c := NewComputer(r)
	if c == nil {
		t.Skip("platform not supported")
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := c.Execute(ctx, agent.ToolContext{}, map[string]any{"action": "screenshot"})
	if err == nil {
		t.Fatal("expected error from cancelled context")
	}
}

func TestComputerResultContentFormat(t *testing.T) {
	backend := newFakeBackend()
	r := newTestRegistry(backend)
	c := NewComputer(r)
	if c == nil {
		t.Skip("platform not supported")
	}

	result, err := c.Execute(context.Background(), agent.ToolContext{}, map[string]any{
		"action": "click", "x": float64(10), "y": float64(20),
	})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(result.Content, "(10,20)") {
		t.Fatalf("content should include coordinates, got: %q", result.Content)
	}
	if !strings.Contains(result.Content, "local") {
		t.Fatalf("content should include environment ID, got: %q", result.Content)
	}
}

func TestComputerScreenshotEnvironmentInContent(t *testing.T) {
	backend := newFakeBackend()
	r := newTestRegistry(backend)
	c := NewComputer(r)
	if c == nil {
		t.Skip("platform not supported")
	}

	result, err := c.Execute(context.Background(), agent.ToolContext{}, map[string]any{
		"action": "screenshot",
	})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(result.Content, "local") {
		t.Fatalf("screenshot content should mention environment, got: %q", result.Content)
	}
	if !strings.Contains(result.Content, "1920") {
		t.Fatalf("screenshot content should mention width, got: %q", result.Content)
	}
}

func TestComputerApprovalScopeIncludesTarget(t *testing.T) {
	c := &Computer{}
	scope := c.ApprovalScope(map[string]any{"action": "screenshot", "target": "prod"})
	if scope != "computer:screenshot:prod" {
		t.Fatalf("scope = %q, want computer:screenshot:prod", scope)
	}
	scope = c.ApprovalScope(map[string]any{"action": "key", "target": "dev", "key": "CTRL+S"})
	if scope != "computer:key:dev:CTRL+S" {
		t.Fatalf("scope = %q, want computer:key:dev:CTRL+S", scope)
	}
}
