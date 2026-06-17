package agent

import (
	"context"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

type compactionStore struct {
	chatID        string
	keepUserTurns int
	summary       string
}

func (s *compactionStore) ArchiveForCompaction(_ context.Context, chatID string, keepUserTurns int, summary string) error {
	s.chatID = chatID
	s.keepUserTurns = keepUserTurns
	s.summary = summary
	return nil
}

func assertCompactionSummaryPair(t *testing.T, messages []api.Message) {
	t.Helper()
	if len(messages) != 2 {
		t.Fatalf("compaction summary pair len = %d, want 2: %#v", len(messages), messages)
	}
	if messages[0].Role != "assistant" || len(messages[0].ToolCalls) != 1 || messages[0].ToolCalls[0].Function.Name != compactionToolName {
		t.Fatalf("compaction assistant message = %#v", messages[0])
	}
	if messages[1].Role != "tool" || messages[1].ToolName != compactionToolName || messages[1].ToolCallID != messages[0].ToolCalls[0].ID {
		t.Fatalf("compaction tool result = %#v", messages[1])
	}
	if !strings.HasPrefix(messages[1].Content, compactionSummaryMessagePrefix) {
		t.Fatalf("compaction tool result missing summary prefix: %#v", messages[1])
	}
}

func TestSimpleCompactorSummarizesOldMessages(t *testing.T) {
	client := &fakeClient{
		responses: [][]api.ChatResponse{{
			{Message: api.Message{Role: "assistant", Content: "summary"}},
		}},
	}
	store := &compactionStore{}
	compactor := NewSimpleCompactor(client, store, CompactionOptions{
		ContextWindowTokens: 100,
		KeepUserTurns:       2,
		Threshold:           0.5,
	})

	messages := []api.Message{
		{Role: "system", Content: "stay pinned"},
		{Role: "user", Content: "old request"},
		{Role: "assistant", Content: "old answer", Thinking: "hidden"},
		{Role: "user", Content: "recent one"},
		{Role: "assistant", Content: "recent answer"},
		{Role: "user", Content: "recent two"},
	}

	result, err := compactor.MaybeCompact(context.Background(), CompactionRequest{
		ChatID:   "chat-1",
		Model:    "model",
		Messages: messages,
		Latest:   api.ChatResponse{Metrics: api.Metrics{PromptEvalCount: 75}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !result.Compacted {
		t.Fatal("expected compaction")
	}
	compacted := result.Messages
	if len(compacted) != 6 {
		t.Fatalf("compacted messages = %d, want 6", len(compacted))
	}
	if compacted[0].Content != "stay pinned" {
		t.Fatalf("first message = %#v", compacted[0])
	}
	if result.Summary != "summary" {
		t.Fatalf("result summary = %q", result.Summary)
	}
	if compacted[1].Content != "recent one" || compacted[3].Content != "recent two" {
		t.Fatalf("recent turns were not kept: %#v", compacted)
	}
	assertCompactionSummaryPair(t, compacted[4:])
	if store.chatID != "chat-1" || store.keepUserTurns != 2 || store.summary != "summary" {
		t.Fatalf("archive call = %#v", store)
	}
	if len(client.requests) != 1 {
		t.Fatalf("summary requests = %d, want 1", len(client.requests))
	}
	if strings.Contains(client.requests[0].Messages[1].Content, "hidden") {
		t.Fatal("compaction prompt should omit thinking")
	}
}

func TestSimpleCompactorTruncatesOversizedSummary(t *testing.T) {
	longSummary := strings.Repeat("x", maxCompactionSummaryBytes+1024)
	client := &fakeClient{
		responses: [][]api.ChatResponse{{
			{Message: api.Message{Role: "assistant", Content: longSummary}},
		}},
	}
	store := &compactionStore{}
	compactor := NewSimpleCompactor(client, store, CompactionOptions{
		ContextWindowTokens: 100,
		KeepUserTurns:       1,
		Threshold:           0.5,
	})

	result, err := compactor.MaybeCompact(context.Background(), CompactionRequest{
		ChatID: "chat-1",
		Model:  "model",
		Messages: []api.Message{
			{Role: "user", Content: "old one"},
			{Role: "assistant", Content: "old answer"},
			{Role: "user", Content: "recent one"},
		},
		Latest: api.ChatResponse{Metrics: api.Metrics{PromptEvalCount: 75}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !result.Compacted {
		t.Fatal("expected compaction")
	}
	if len(result.Summary) > maxCompactionSummaryBytes {
		t.Fatalf("summary bytes = %d, want <= %d", len(result.Summary), maxCompactionSummaryBytes)
	}
	if !strings.HasSuffix(result.Summary, compactionSummaryTruncated) {
		t.Fatalf("summary missing truncation marker")
	}
	if store.summary != result.Summary {
		t.Fatalf("stored summary mismatch")
	}
	if !strings.Contains(result.Messages[len(result.Messages)-1].Content, compactionSummaryTruncated) {
		t.Fatalf("compacted message missing truncation marker: %#v", result.Messages)
	}
}

func TestSimpleCompactorKeepsFewerTurnsForShortChats(t *testing.T) {
	client := &fakeClient{
		responses: [][]api.ChatResponse{{
			{Message: api.Message{Role: "assistant", Content: "short summary"}},
		}},
	}
	store := &compactionStore{}
	compactor := NewSimpleCompactor(client, store, CompactionOptions{
		ContextWindowTokens: 100,
		KeepUserTurns:       3,
		Threshold:           0.5,
	})

	result, err := compactor.MaybeCompact(context.Background(), CompactionRequest{
		ChatID: "chat-1",
		Model:  "model",
		Messages: []api.Message{
			{Role: "user", Content: "old request"},
			{Role: "assistant", Content: "old answer"},
			{Role: "user", Content: "latest request"},
		},
		Latest: api.ChatResponse{Metrics: api.Metrics{PromptEvalCount: 75}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !result.Compacted {
		t.Fatal("expected compaction")
	}
	if store.keepUserTurns != 1 {
		t.Fatalf("kept user turns = %d, want 1", store.keepUserTurns)
	}
	if len(result.Messages) != 3 {
		t.Fatalf("messages = %#v, want latest request plus compaction tool pair", result.Messages)
	}
	if result.Messages[0].Content != "latest request" {
		t.Fatalf("latest turn was not kept: %#v", result.Messages)
	}
	assertCompactionSummaryPair(t, result.Messages[1:])
}

func TestSimpleCompactorCanArchiveWholeShortChat(t *testing.T) {
	client := &fakeClient{
		responses: [][]api.ChatResponse{{
			{Message: api.Message{Role: "assistant", Content: "whole summary"}},
		}},
	}
	store := &compactionStore{}
	compactor := NewSimpleCompactor(client, store, CompactionOptions{
		ContextWindowTokens: 100,
		KeepUserTurns:       3,
		Threshold:           0.5,
	})

	result, err := compactor.MaybeCompact(context.Background(), CompactionRequest{
		ChatID: "chat-1",
		Model:  "model",
		Messages: []api.Message{
			{Role: "user", Content: "only request"},
			{Role: "assistant", Content: "only answer"},
		},
		Latest: api.ChatResponse{Metrics: api.Metrics{PromptEvalCount: 75}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !result.Compacted {
		t.Fatal("expected compaction")
	}
	if store.keepUserTurns != 0 {
		t.Fatalf("kept user turns = %d, want 0", store.keepUserTurns)
	}
	if len(result.Messages) != 2 {
		t.Fatalf("messages = %#v, want only compaction tool pair", result.Messages)
	}
	assertCompactionSummaryPair(t, result.Messages)
}

func TestSimpleCompactorSkipsBelowThreshold(t *testing.T) {
	client := &fakeClient{}
	compactor := NewSimpleCompactor(client, nil, CompactionOptions{
		ContextWindowTokens: 100,
		Threshold:           0.8,
	})

	messages := []api.Message{
		{Role: "user", Content: "one"},
		{Role: "user", Content: "two"},
		{Role: "user", Content: "three"},
		{Role: "user", Content: "four"},
		{Role: "user", Content: "five"},
	}
	result, err := compactor.MaybeCompact(context.Background(), CompactionRequest{
		Model:    "model",
		Messages: messages,
		Latest:   api.ChatResponse{Metrics: api.Metrics{PromptEvalCount: 50}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Compacted {
		t.Fatal("did not expect compaction")
	}
	if result.Due {
		t.Fatal("below-threshold compaction should not be due")
	}
	if len(result.Messages) != len(messages) {
		t.Fatalf("messages changed below threshold: %#v", result.Messages)
	}
	if len(client.requests) != 0 {
		t.Fatalf("summary requests = %d, want 0", len(client.requests))
	}
}

func TestSimpleCompactorUsesEstimatedMessagesWhenPromptEvalMissing(t *testing.T) {
	client := &fakeClient{
		responses: [][]api.ChatResponse{{
			{Message: api.Message{Role: "assistant", Content: "estimated summary"}},
		}},
	}
	compactor := NewSimpleCompactor(client, nil, CompactionOptions{
		ContextWindowTokens: 100,
		KeepUserTurns:       1,
		Threshold:           0.8,
	})

	result, err := compactor.MaybeCompact(context.Background(), CompactionRequest{
		Model: "model",
		Messages: []api.Message{
			{Role: "user", Content: "old request"},
			{Role: "assistant", Content: "old answer"},
			{Role: "user", Content: "read large output"},
			{Role: "assistant", ToolCalls: []api.ToolCall{{
				ID: "call-1",
				Function: api.ToolCallFunction{
					Name: "read",
				},
			}}},
			{Role: "tool", ToolName: "read", ToolCallID: "call-1", Content: strings.Repeat("x", 360)},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !result.Due || !result.Compacted {
		t.Fatalf("expected estimate-driven compaction, got %#v", result)
	}
	if result.Summary != "estimated summary" {
		t.Fatalf("summary = %q", result.Summary)
	}
}

func TestSimpleCompactorEstimateIncludesRequestPreamble(t *testing.T) {
	compactor := NewSimpleCompactor(nil, nil, CompactionOptions{
		ContextWindowTokens: 100,
		Threshold:           0.8,
	})

	if !compactor.shouldCompact(CompactionRequest{
		SystemPrompt: strings.Repeat("system ", 360),
		Messages:     []api.Message{{Role: "user", Content: "tiny"}},
	}) {
		t.Fatal("system prompt should count toward compaction estimate")
	}

	if !compactor.shouldCompact(CompactionRequest{
		Messages: []api.Message{{Role: "user", Content: "tiny"}},
		Tools: api.Tools{{
			Type: "function",
			Function: api.ToolFunction{
				Name:        "verbose_tool",
				Description: strings.Repeat("description ", 360),
			},
		}},
	}) {
		t.Fatal("tool definitions should count toward compaction estimate")
	}
}

func TestCompactionPromptFitsBudgetByTruncatingLargeToolOutput(t *testing.T) {
	largeToolOutput := strings.Repeat("x", 10_000)
	body, err := compactionPrompt("", []api.Message{
		{Role: "user", Content: "what changed?"},
		{Role: "assistant", ToolCalls: []api.ToolCall{{
			ID: "call-1",
			Function: api.ToolCallFunction{
				Name: "bash",
			},
		}}},
		{Role: "tool", ToolName: "bash", ToolCallID: "call-1", Content: largeToolOutput},
	}, 300)
	if err != nil {
		t.Fatal(err)
	}
	if estimateCompactionTokens(body) > 300 {
		t.Fatalf("compaction prompt tokens = %d, want <= 300", estimateCompactionTokens(body))
	}
	if strings.Count(body, "x") >= len(largeToolOutput) {
		t.Fatal("large tool output was not truncated")
	}
	if !strings.Contains(body, "[tool output truncated: omitted ") {
		t.Fatalf("truncation marker missing from compaction prompt: %q", body)
	}
}

func TestCompactionSummaryTextStripsPrefix(t *testing.T) {
	content := compactionSummaryMessage("worked on branch changes")
	if got := compactionSummaryText(content); got != "worked on branch changes" {
		t.Fatalf("summary text = %q", got)
	}
}

func TestResolveContextWindowTokensUsesSmallerKnownWindow(t *testing.T) {
	tests := []struct {
		name       string
		options    map[string]any
		configured int
		want       int
	}{
		{
			name:       "explicit smaller num ctx",
			options:    map[string]any{"num_ctx": 4096},
			configured: 8192,
			want:       4096,
		},
		{
			name:       "server effective smaller than requested num ctx",
			options:    map[string]any{"num_ctx": 131072},
			configured: 8192,
			want:       8192,
		},
		{
			name:       "metadata without explicit num ctx",
			configured: 32768,
			want:       32768,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := ResolveContextWindowTokens(tt.options, tt.configured); got != tt.want {
				t.Fatalf("ResolveContextWindowTokens() = %d, want %d", got, tt.want)
			}
		})
	}
}

func TestSimpleCompactorForceCompactsWithoutPromptEvalCount(t *testing.T) {
	client := &fakeClient{
		responses: [][]api.ChatResponse{{
			{Message: api.Message{Role: "assistant", Content: "forced summary"}},
		}},
	}
	compactor := NewSimpleCompactor(client, nil, CompactionOptions{
		ContextWindowTokens: 100,
		KeepUserTurns:       1,
		Threshold:           0.8,
	})

	result, err := compactor.MaybeCompact(context.Background(), CompactionRequest{
		Model: "model",
		Messages: []api.Message{
			{Role: "user", Content: "old"},
			{Role: "assistant", Content: "old answer"},
			{Role: "user", Content: "recent"},
		},
		Force: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !result.Due || !result.Compacted {
		t.Fatalf("forced compaction result = %#v", result)
	}
	if result.Summary != "forced summary" {
		t.Fatalf("summary = %q", result.Summary)
	}
}

func TestSimpleCompactorDefaultsToKeepingThreeUserTurns(t *testing.T) {
	client := &fakeClient{
		responses: [][]api.ChatResponse{{
			{Message: api.Message{Role: "assistant", Content: "summary"}},
		}},
	}
	store := &compactionStore{}
	compactor := NewSimpleCompactor(client, store, CompactionOptions{
		ContextWindowTokens: 100,
		Threshold:           0.5,
	})

	result, err := compactor.MaybeCompact(context.Background(), CompactionRequest{
		ChatID: "chat-1",
		Model:  "model",
		Messages: []api.Message{
			{Role: "user", Content: "old"},
			{Role: "assistant", Content: "old answer"},
			{Role: "user", Content: "one"},
			{Role: "assistant", Content: "one answer"},
			{Role: "user", Content: "two"},
			{Role: "assistant", Content: "two answer"},
			{Role: "user", Content: "three"},
		},
		Latest: api.ChatResponse{Metrics: api.Metrics{PromptEvalCount: 75}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !result.Compacted {
		t.Fatal("expected compaction")
	}
	if store.keepUserTurns != 3 {
		t.Fatalf("keepUserTurns = %d, want 3", store.keepUserTurns)
	}
	if got := result.Messages[0].Content; got != "one" {
		t.Fatalf("first kept turn = %q, want one", got)
	}
	assertCompactionSummaryPair(t, result.Messages[len(result.Messages)-2:])
}

func TestSimpleCompactorCarriesPreviousSummary(t *testing.T) {
	client := &fakeClient{
		responses: [][]api.ChatResponse{{
			{Message: api.Message{Role: "assistant", Content: "new summary"}},
		}},
	}
	compactor := NewSimpleCompactor(client, nil, CompactionOptions{
		ContextWindowTokens: 10,
		KeepUserTurns:       1,
		Threshold:           0.5,
	})

	result, err := compactor.MaybeCompact(context.Background(), CompactionRequest{
		Model: "model",
		Messages: []api.Message{
			{Role: "system", Content: compactionSummaryMessagePrefix + "old summary"},
			{Role: "user", Content: "old"},
			{Role: "assistant", Content: "old answer"},
			{Role: "user", Content: "recent"},
		},
		Latest: api.ChatResponse{Metrics: api.Metrics{PromptEvalCount: 9}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !result.Compacted {
		t.Fatal("expected compaction")
	}
	if !strings.Contains(client.requests[0].Messages[1].Content, "Previous summary:\nold summary") {
		t.Fatalf("previous summary missing from request: %q", client.requests[0].Messages[1].Content)
	}
}

func TestSimpleCompactorCarriesPreviousToolSummaryAndKeepsNewSummaryLatest(t *testing.T) {
	client := &fakeClient{
		responses: [][]api.ChatResponse{{
			{Message: api.Message{Role: "assistant", Content: "new summary"}},
		}},
	}
	compactor := NewSimpleCompactor(client, nil, CompactionOptions{
		ContextWindowTokens: 10,
		KeepUserTurns:       1,
		Threshold:           0.5,
	})

	messages := []api.Message{
		{Role: "user", Content: "kept before old summary"},
		compactionSummaryMessages("old summary")[0],
		compactionSummaryMessages("old summary")[1],
		{Role: "user", Content: "latest request"},
	}
	result, err := compactor.MaybeCompact(context.Background(), CompactionRequest{
		Model:    "model",
		Messages: messages,
		Latest:   api.ChatResponse{Metrics: api.Metrics{PromptEvalCount: 9}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if !result.Compacted {
		t.Fatal("expected compaction")
	}
	if !strings.Contains(client.requests[0].Messages[1].Content, "Previous summary:\nold summary") {
		t.Fatalf("previous summary missing from request: %q", client.requests[0].Messages[1].Content)
	}
	if len(result.Messages) != 3 {
		t.Fatalf("messages = %#v, want latest request plus compaction pair", result.Messages)
	}
	if result.Messages[0].Content != "latest request" {
		t.Fatalf("kept suffix = %#v", result.Messages)
	}
	assertCompactionSummaryPair(t, result.Messages[1:])
}
