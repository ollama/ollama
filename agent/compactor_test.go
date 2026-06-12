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
	if len(compacted) != 5 {
		t.Fatalf("compacted messages = %d, want 5", len(compacted))
	}
	if compacted[0].Content != "stay pinned" {
		t.Fatalf("first message = %#v", compacted[0])
	}
	if compacted[1].Role != "user" {
		t.Fatalf("summary role = %q, want user", compacted[1].Role)
	}
	if !strings.HasPrefix(compacted[1].Content, compactionSummaryMessagePrefix) {
		t.Fatalf("summary message = %#v", compacted[1])
	}
	if result.Summary != "summary" {
		t.Fatalf("result summary = %q", result.Summary)
	}
	if compacted[2].Content != "recent one" || compacted[4].Content != "recent two" {
		t.Fatalf("recent turns were not kept: %#v", compacted)
	}
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
	if got := result.Messages[1].Content; got != "one" {
		t.Fatalf("first kept turn = %q, want one", got)
	}
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
