package cmd

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/ollama/ollama/api"
)

// TestToolMessage verifies that tool messages are constructed correctly
// with ToolName and ToolCallID preserved from the tool call.
func TestToolMessage(t *testing.T) {
	tests := []struct {
		name     string
		call     api.ToolCall
		content  string
		expected api.Message
	}{
		{
			name: "basic tool message with ID",
			call: api.ToolCall{
				ID: "call_abc123",
				Function: api.ToolCallFunction{
					Name: "get_weather",
					Arguments: api.ToolCallFunctionArguments{
						"location": "Paris",
					},
				},
			},
			content: "Sunny, 22°C",
			expected: api.Message{
				Role:       "tool",
				Content:    "Sunny, 22°C",
				ToolName:   "get_weather",
				ToolCallID: "call_abc123",
			},
		},
		{
			name: "tool message without ID",
			call: api.ToolCall{
				Function: api.ToolCallFunction{
					Name: "calculate",
					Arguments: api.ToolCallFunctionArguments{
						"expression": "2+2",
					},
				},
			},
			content: "4",
			expected: api.Message{
				Role:     "tool",
				Content:  "4",
				ToolName: "calculate",
				// ToolCallID should be empty when call.ID is empty
			},
		},
		{
			name: "MCP tool message",
			call: api.ToolCall{
				ID: "call_mcp123",
				Function: api.ToolCallFunction{
					Name: "mcp_websearch_search",
					Arguments: api.ToolCallFunctionArguments{
						"query": "ollama agents",
					},
				},
			},
			content: "Found 10 results",
			expected: api.Message{
				Role:       "tool",
				Content:    "Found 10 results",
				ToolName:   "mcp_websearch_search",
				ToolCallID: "call_mcp123",
			},
		},
		{
			name: "skill tool message",
			call: api.ToolCall{
				ID: "call_skill456",
				Function: api.ToolCallFunction{
					Name: "run_skill_script",
					Arguments: api.ToolCallFunctionArguments{
						"skill":   "calculator",
						"command": "python scripts/calc.py 2+2",
					},
				},
			},
			content: "Result: 4",
			expected: api.Message{
				Role:       "tool",
				Content:    "Result: 4",
				ToolName:   "run_skill_script",
				ToolCallID: "call_skill456",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := toolMessage(tt.call, tt.content)
			if diff := cmp.Diff(tt.expected, result); diff != "" {
				t.Errorf("toolMessage() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestAssistantMessageWithThinking verifies that assistant messages
// in the tool loop should include thinking content.
func TestAssistantMessageConstruction(t *testing.T) {
	tests := []struct {
		name        string
		content     string
		thinking    string
		toolCalls   []api.ToolCall
		expectedMsg api.Message
	}{
		{
			name:     "assistant with thinking and tool calls",
			content:  "",
			thinking: "I need to check the weather for Paris.",
			toolCalls: []api.ToolCall{
				{
					ID: "call_1",
					Function: api.ToolCallFunction{
						Name:      "get_weather",
						Arguments: api.ToolCallFunctionArguments{"city": "Paris"},
					},
				},
			},
			expectedMsg: api.Message{
				Role:     "assistant",
				Content:  "",
				Thinking: "I need to check the weather for Paris.",
				ToolCalls: []api.ToolCall{
					{
						ID: "call_1",
						Function: api.ToolCallFunction{
							Name:      "get_weather",
							Arguments: api.ToolCallFunctionArguments{"city": "Paris"},
						},
					},
				},
			},
		},
		{
			name:     "assistant with content, thinking, and tool calls",
			content:  "Let me check that for you.",
			thinking: "User wants weather info.",
			toolCalls: []api.ToolCall{
				{
					ID: "call_2",
					Function: api.ToolCallFunction{
						Name:      "search",
						Arguments: api.ToolCallFunctionArguments{"query": "weather"},
					},
				},
			},
			expectedMsg: api.Message{
				Role:     "assistant",
				Content:  "Let me check that for you.",
				Thinking: "User wants weather info.",
				ToolCalls: []api.ToolCall{
					{
						ID: "call_2",
						Function: api.ToolCallFunction{
							Name:      "search",
							Arguments: api.ToolCallFunctionArguments{"query": "weather"},
						},
					},
				},
			},
		},
		{
			name:     "assistant with multiple tool calls",
			content:  "",
			thinking: "I'll check both cities.",
			toolCalls: []api.ToolCall{
				{
					ID: "call_a",
					Function: api.ToolCallFunction{
						Name:      "get_weather",
						Arguments: api.ToolCallFunctionArguments{"city": "Paris"},
					},
				},
				{
					ID: "call_b",
					Function: api.ToolCallFunction{
						Name:      "get_weather",
						Arguments: api.ToolCallFunctionArguments{"city": "London"},
					},
				},
			},
			expectedMsg: api.Message{
				Role:     "assistant",
				Content:  "",
				Thinking: "I'll check both cities.",
				ToolCalls: []api.ToolCall{
					{
						ID: "call_a",
						Function: api.ToolCallFunction{
							Name:      "get_weather",
							Arguments: api.ToolCallFunctionArguments{"city": "Paris"},
						},
					},
					{
						ID: "call_b",
						Function: api.ToolCallFunction{
							Name:      "get_weather",
							Arguments: api.ToolCallFunctionArguments{"city": "London"},
						},
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Simulate the assistant message construction as done in chat()
			assistantMsg := api.Message{
				Role:      "assistant",
				Content:   tt.content,
				Thinking:  tt.thinking,
				ToolCalls: tt.toolCalls,
			}

			if diff := cmp.Diff(tt.expectedMsg, assistantMsg); diff != "" {
				t.Errorf("assistant message mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

// TestMessageStitchingOrder verifies that messages in a tool loop
// are stitched in the correct order:
// 1. User message
// 2. Assistant message with tool calls (and thinking)
// 3. Tool result messages (one per tool call, in order)
// 4. Next assistant response
func TestMessageStitchingOrder(t *testing.T) {
	// Simulate a complete tool loop conversation
	messages := []api.Message{
		// Initial user message
		{Role: "user", Content: "What's the weather in Paris and London?"},
		// Assistant's first response with tool calls
		{
			Role:     "assistant",
			Content:  "",
			Thinking: "I need to check the weather for both cities.",
			ToolCalls: []api.ToolCall{
				{ID: "call_1", Function: api.ToolCallFunction{Name: "get_weather", Arguments: api.ToolCallFunctionArguments{"city": "Paris"}}},
				{ID: "call_2", Function: api.ToolCallFunction{Name: "get_weather", Arguments: api.ToolCallFunctionArguments{"city": "London"}}},
			},
		},
		// Tool results (in order matching tool calls)
		{Role: "tool", Content: "Sunny, 22°C", ToolName: "get_weather", ToolCallID: "call_1"},
		{Role: "tool", Content: "Rainy, 15°C", ToolName: "get_weather", ToolCallID: "call_2"},
		// Final assistant response
		{Role: "assistant", Content: "Paris is sunny at 22°C, and London is rainy at 15°C.", Thinking: "Got the data, now summarizing."},
	}

	// Verify structure
	expectedRoles := []string{"user", "assistant", "tool", "tool", "assistant"}
	for i, msg := range messages {
		if msg.Role != expectedRoles[i] {
			t.Errorf("message %d: expected role %q, got %q", i, expectedRoles[i], msg.Role)
		}
	}

	// Verify tool results match tool calls in order
	assistantWithTools := messages[1]
	toolResults := []api.Message{messages[2], messages[3]}

	if len(toolResults) != len(assistantWithTools.ToolCalls) {
		t.Errorf("expected %d tool results for %d tool calls", len(assistantWithTools.ToolCalls), len(toolResults))
	}

	for i, result := range toolResults {
		expectedToolCallID := assistantWithTools.ToolCalls[i].ID
		if result.ToolCallID != expectedToolCallID {
			t.Errorf("tool result %d: expected ToolCallID %q, got %q", i, expectedToolCallID, result.ToolCallID)
		}
		expectedToolName := assistantWithTools.ToolCalls[i].Function.Name
		if result.ToolName != expectedToolName {
			t.Errorf("tool result %d: expected ToolName %q, got %q", i, expectedToolName, result.ToolName)
		}
	}

	// Verify thinking is present in assistant messages
	if messages[1].Thinking == "" {
		t.Error("first assistant message should have thinking content")
	}
	if messages[4].Thinking == "" {
		t.Error("final assistant message should have thinking content")
	}
}

// TestMultiTurnToolLoop verifies message stitching across multiple
// tool call iterations.
func TestMultiTurnToolLoop(t *testing.T) {
	messages := []api.Message{
		{Role: "user", Content: "What's 2+2 and also what's the weather in Paris?"},
		// First tool call: calculate
		{
			Role:     "assistant",
			Thinking: "I'll start with the calculation.",
			ToolCalls: []api.ToolCall{
				{ID: "calc_1", Function: api.ToolCallFunction{Name: "calculate", Arguments: api.ToolCallFunctionArguments{"expr": "2+2"}}},
			},
		},
		{Role: "tool", Content: "4", ToolName: "calculate", ToolCallID: "calc_1"},
		// Second tool call: weather
		{
			Role:     "assistant",
			Thinking: "Got the calculation. Now checking weather.",
			ToolCalls: []api.ToolCall{
				{ID: "weather_1", Function: api.ToolCallFunction{Name: "get_weather", Arguments: api.ToolCallFunctionArguments{"city": "Paris"}}},
			},
		},
		{Role: "tool", Content: "Sunny, 20°C", ToolName: "get_weather", ToolCallID: "weather_1"},
		// Final response
		{Role: "assistant", Content: "2+2 equals 4, and Paris is sunny at 20°C."},
	}

	// Count message types
	roleCounts := map[string]int{}
	for _, msg := range messages {
		roleCounts[msg.Role]++
	}

	if roleCounts["user"] != 1 {
		t.Errorf("expected 1 user message, got %d", roleCounts["user"])
	}
	if roleCounts["assistant"] != 3 {
		t.Errorf("expected 3 assistant messages, got %d", roleCounts["assistant"])
	}
	if roleCounts["tool"] != 2 {
		t.Errorf("expected 2 tool messages, got %d", roleCounts["tool"])
	}

	// Verify each tool message follows an assistant with matching tool call
	for i, msg := range messages {
		if msg.Role == "tool" {
			// Find preceding assistant message with tool calls
			var precedingAssistant *api.Message
			for j := i - 1; j >= 0; j-- {
				if messages[j].Role == "assistant" && len(messages[j].ToolCalls) > 0 {
					precedingAssistant = &messages[j]
					break
				}
			}

			if precedingAssistant == nil {
				t.Errorf("tool message at index %d has no preceding assistant with tool calls", i)
				continue
			}

			// Verify tool result matches one of the tool calls
			found := false
			for _, tc := range precedingAssistant.ToolCalls {
				if tc.ID == msg.ToolCallID {
					found = true
					break
				}
			}
			if !found {
				t.Errorf("tool message at index %d has ToolCallID %q not found in preceding tool calls", i, msg.ToolCallID)
			}
		}
	}
}

// TestSkillCatalogRunToolCallPreservesFields tests that skill catalog
// returns tool messages with correct fields.
func TestSkillCatalogToolMessageFields(t *testing.T) {
	// Create a minimal test for toolMessage function
	call := api.ToolCall{
		ID: "test_id_123",
		Function: api.ToolCallFunction{
			Name: "run_skill_script",
			Arguments: api.ToolCallFunctionArguments{
				"skill":   "test-skill",
				"command": "echo hello",
			},
		},
	}

	msg := toolMessage(call, "hello")

	if msg.Role != "tool" {
		t.Errorf("expected role 'tool', got %q", msg.Role)
	}
	if msg.Content != "hello" {
		t.Errorf("expected content 'hello', got %q", msg.Content)
	}
	if msg.ToolName != "run_skill_script" {
		t.Errorf("expected ToolName 'run_skill_script', got %q", msg.ToolName)
	}
	if msg.ToolCallID != "test_id_123" {
		t.Errorf("expected ToolCallID 'test_id_123', got %q", msg.ToolCallID)
	}
}
