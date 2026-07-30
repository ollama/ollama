package tools

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ollama/ollama/agent"
	"github.com/ollama/ollama/api"
)

func TestSkillLoadsCoreCatalogWithApproval(t *testing.T) {
	catalog := testSkillCatalog(t)
	tool := &Skill{Catalog: catalog}
	if !agent.ToolRequiresApproval(tool, map[string]any{"name": "release-notes"}) {
		t.Fatal("model-initiated skill loading should require approval")
	}
	result, err := tool.Execute(context.Background(), agent.ToolContext{}, map[string]any{"name": "release-notes"})
	if err != nil || !strings.Contains(result.Content, "Use concise bullets.") {
		t.Fatalf("tool result = %#v, %v", result, err)
	}
}

func TestModelSkillLoadRequiresApproval(t *testing.T) {
	for _, tt := range []struct {
		name        string
		approval    agent.Approval
		prompt      bool
		wantCalls   int
		wantPrompts int
		wantResult  string
	}{
		{name: "rejected", approval: agent.Approval{Reason: "Skill loading denied."}, prompt: true, wantCalls: 1, wantPrompts: 1, wantResult: "Skill loading denied."},
		{name: "approved", approval: agent.Approval{Allow: true}, prompt: true, wantCalls: 2, wantPrompts: 1, wantResult: "Use concise bullets."},
		{name: "headless denied", wantCalls: 1, wantResult: "Tool execution requires approval"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			catalog := testSkillCatalog(t)
			args := api.NewToolCallFunctionArguments()
			args.Set("name", "release-notes")
			client := &skillTestClient{responses: [][]api.ChatResponse{
				{{Message: api.Message{Role: "assistant", ToolCalls: []api.ToolCall{{
					ID:       "call_skill_1",
					Function: api.ToolCallFunction{Name: "skill", Arguments: args},
				}}}}},
				{{Message: api.Message{Role: "assistant", Content: "done"}}},
			}}
			var prompter *skillApprovalPrompter
			var approvalPrompter agent.ApprovalPrompter
			if tt.prompt {
				prompter = &skillApprovalPrompter{result: tt.approval}
				approvalPrompter = prompter
			}
			registry := &agent.Registry{}
			registry.Register(&Skill{Catalog: catalog})

			result, err := (&agent.Session{
				Client:           client,
				Tools:            registry,
				ApprovalPrompter: approvalPrompter,
			}).Run(context.Background(), agent.RunOptions{
				Model:       "test",
				NewMessages: []api.Message{{Role: "user", Content: "load the release-notes skill"}},
			})
			if err != nil {
				t.Fatal(err)
			}
			if tt.prompt {
				if got := len(prompter.requests); got != tt.wantPrompts {
					t.Fatalf("approval prompts = %d, want %d", got, tt.wantPrompts)
				}
				request := prompter.requests[0]
				if len(request.Calls) != 1 || request.Calls[0].ToolName != "skill" || request.Calls[0].ApprovalScope != "skill" || request.Calls[0].Args["name"] != "release-notes" {
					t.Fatalf("approval request = %#v", request)
				}
			}
			if got := client.calls; got != tt.wantCalls {
				t.Fatalf("model calls = %d, want %d", got, tt.wantCalls)
			}
			var toolResult string
			for _, message := range result.Messages {
				if message.Role == "tool" && message.ToolCallID == "call_skill_1" {
					toolResult = message.Content
					break
				}
			}
			if !strings.Contains(toolResult, tt.wantResult) {
				t.Fatalf("skill tool result = %q, want it to contain %q", toolResult, tt.wantResult)
			}
		})
	}
}

func TestExplicitSkillActivationBypassesApproval(t *testing.T) {
	catalog := testSkillCatalog(t)
	client := &skillTestClient{responses: [][]api.ChatResponse{{{Message: api.Message{Role: "assistant", Content: "done"}}}}}
	prompter := &skillApprovalPrompter{result: agent.Approval{}}
	result, err := (&agent.Session{
		Client:           client,
		Skills:           catalog,
		ApprovalPrompter: prompter,
	}).Run(context.Background(), agent.RunOptions{
		Model:       "test",
		NewMessages: []api.Message{{Role: "user", Content: "draft release notes"}},
		SkillName:   "release-notes",
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(prompter.requests) != 0 {
		t.Fatalf("explicit activation prompted for approval: %#v", prompter.requests)
	}
	if len(result.Messages) != 4 || result.Messages[2].ToolName != "skill" || !strings.Contains(result.Messages[2].Content, "Use concise bullets.") {
		t.Fatalf("synthetic skill activation = %#v", result.Messages)
	}
}

func testSkillCatalog(t *testing.T) *agent.SkillCatalog {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "release-notes")
	if err := os.Mkdir(path, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(path, "SKILL.md"), []byte("---\nname: release-notes\ndescription: Draft release notes.\n---\nUse concise bullets."), 0o644); err != nil {
		t.Fatal(err)
	}
	catalog, err := agent.DiscoverSkills(dir)
	if err != nil {
		t.Fatal(err)
	}
	return catalog
}

type skillTestClient struct {
	responses [][]api.ChatResponse
	calls     int
}

func (c *skillTestClient) Chat(_ context.Context, _ *api.ChatRequest, fn api.ChatResponseFunc) error {
	if c.calls >= len(c.responses) {
		return nil
	}
	for _, response := range c.responses[c.calls] {
		if err := fn(response); err != nil {
			return err
		}
	}
	c.calls++
	return nil
}

type skillApprovalPrompter struct {
	requests []agent.ApprovalRequest
	result   agent.Approval
}

func (p *skillApprovalPrompter) PromptApproval(_ context.Context, request agent.ApprovalRequest) (agent.Approval, error) {
	p.requests = append(p.requests, request)
	return p.result, nil
}
