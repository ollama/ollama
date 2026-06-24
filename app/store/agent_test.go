package store

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/agent"
	"github.com/ollama/ollama/api"
)

func newTestAgentStore(t *testing.T) *Store {
	t.Helper()
	setTestHome(t, t.TempDir())
	store, err := New(filepath.Join(t.TempDir(), "db.sqlite"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := store.Close(); err != nil {
			t.Fatal(err)
		}
	})
	return store
}

func TestAgentStoreWritesAppCompatibleChat(t *testing.T) {
	store := newTestAgentStore(t)
	ctx := context.Background()

	if err := store.EnsureChat(ctx, "chat-1", ""); err != nil {
		t.Fatal(err)
	}
	if err := store.AppendAgentMessage(ctx, "chat-1", api.Message{Role: "user", Content: "hello from cli"}, ""); err != nil {
		t.Fatal(err)
	}

	args := api.NewToolCallFunctionArguments()
	args.Set("command", "pwd")
	if err := store.AppendAgentMessage(ctx, "chat-1", api.Message{
		Role:    "assistant",
		Content: "I'll check.",
		ToolCalls: []api.ToolCall{{
			ID: "call-1",
			Function: api.ToolCallFunction{
				Name:      "bash",
				Arguments: args,
			},
		}},
	}, "llama3.2"); err != nil {
		t.Fatal(err)
	}
	if err := store.AppendAgentMessage(ctx, "chat-1", api.Message{
		Role:       "tool",
		Content:    "cwd",
		ToolName:   "bash",
		ToolCallID: "call-1",
	}, ""); err != nil {
		t.Fatal(err)
	}

	agentChat, err := store.AgentChat(ctx, "chat-1")
	if err != nil {
		t.Fatal(err)
	}
	if len(agentChat.Messages) != 3 {
		t.Fatalf("messages = %d, want 3", len(agentChat.Messages))
	}
	if agentChat.Title != "hello from cli" {
		t.Fatalf("title = %q, want %q", agentChat.Title, "hello from cli")
	}
	if got := agentChat.Messages[1].ToolCalls[0].Function.Name; got != "bash" {
		t.Fatalf("tool name = %q, want bash", got)
	}
	if got := agentChat.Messages[1].ToolCalls[0].ID; got != "call-1" {
		t.Fatalf("tool call id = %q, want call-1", got)
	}
	if agentChat.Messages[2].Role != "tool" || agentChat.Messages[2].ToolCallID != "call-1" {
		t.Fatalf("tool result = %#v", agentChat.Messages[2])
	}

	appChat, err := store.Chat("chat-1")
	if err != nil {
		t.Fatal(err)
	}
	if len(appChat.Messages) != 3 || appChat.Messages[0].Content != "hello from cli" {
		t.Fatalf("app chat = %#v", appChat)
	}
}

func TestAgentStoreRepairsDanglingToolCallsOnResume(t *testing.T) {
	store := newTestAgentStore(t)
	ctx := context.Background()

	args := api.NewToolCallFunctionArguments()
	args.Set("command", "pwd")
	if err := store.AppendAgentMessage(ctx, "chat-1", api.Message{Role: "user", Content: "start"}, ""); err != nil {
		t.Fatal(err)
	}
	if err := store.AppendAgentMessage(ctx, "chat-1", api.Message{
		Role: "assistant",
		ToolCalls: []api.ToolCall{{
			ID: "call-1",
			Function: api.ToolCallFunction{
				Name:      "bash",
				Arguments: args,
			},
		}},
	}, "llama3.2"); err != nil {
		t.Fatal(err)
	}
	if err := store.AppendAgentMessage(ctx, "chat-1", api.Message{Role: "user", Content: "after restart"}, ""); err != nil {
		t.Fatal(err)
	}

	agentChat, err := store.AgentChat(ctx, "chat-1")
	if err != nil {
		t.Fatal(err)
	}
	if len(agentChat.Messages) != 4 {
		t.Fatalf("messages = %#v, want synthetic tool result inserted", agentChat.Messages)
	}
	repair := agentChat.Messages[2]
	if repair.Role != "tool" || repair.ToolName != "bash" || repair.ToolCallID != "call-1" || !strings.Contains(repair.Content, "interrupted") {
		t.Fatalf("repair message = %#v", repair)
	}
	if agentChat.Messages[3].Role != "user" || agentChat.Messages[3].Content != "after restart" {
		t.Fatalf("message after repair = %#v", agentChat.Messages[3])
	}
}

func TestAgentStoreRoundTripsToolMetadataAndImages(t *testing.T) {
	store := newTestAgentStore(t)
	ctx := context.Background()

	first := api.ImageData([]byte("first image"))
	second := api.ImageData([]byte{0, 1, 2, 3})
	if err := store.AppendAgentMessage(ctx, "chat-1", api.Message{
		Role:       "tool",
		Content:    "tool output",
		Images:     []api.ImageData{first, second},
		ToolName:   "bash",
		ToolCallID: "call-1",
	}, ""); err != nil {
		t.Fatal(err)
	}

	chat, err := store.AgentChat(ctx, "chat-1")
	if err != nil {
		t.Fatal(err)
	}
	if len(chat.Messages) != 1 {
		t.Fatalf("messages = %d, want 1", len(chat.Messages))
	}
	msg := chat.Messages[0]
	if msg.ToolName != "bash" || msg.ToolCallID != "call-1" {
		t.Fatalf("tool metadata = %#v", msg)
	}
	if len(msg.Images) != 2 || !bytes.Equal(msg.Images[0], first) || !bytes.Equal(msg.Images[1], second) {
		t.Fatalf("images = %#v, want %#v", msg.Images, []api.ImageData{first, second})
	}

	updated := api.ImageData([]byte("updated image"))
	if err := store.UpdateLastAgentMessage(ctx, "chat-1", api.Message{
		Role:    "user",
		Content: "updated",
		Images:  []api.ImageData{updated},
	}, ""); err != nil {
		t.Fatal(err)
	}
	chat, err = store.AgentChat(ctx, "chat-1")
	if err != nil {
		t.Fatal(err)
	}
	if len(chat.Messages) != 1 || len(chat.Messages[0].Images) != 1 || !bytes.Equal(chat.Messages[0].Images[0], updated) {
		t.Fatalf("updated images = %#v", chat.Messages)
	}
}

func TestAgentStoreLatestAndListChats(t *testing.T) {
	store := newTestAgentStore(t)
	ctx := context.Background()

	if err := store.AppendAgentMessage(ctx, "chat-old", api.Message{Role: "user", Content: "old topic"}, ""); err != nil {
		t.Fatal(err)
	}
	if err := store.AppendAgentMessage(ctx, "chat-old", api.Message{Role: "assistant", Content: "old answer"}, "llama3.2"); err != nil {
		t.Fatal(err)
	}
	if err := store.AppendAgentMessage(ctx, "chat-new", api.Message{Role: "user", Content: "new topic"}, ""); err != nil {
		t.Fatal(err)
	}
	if err := store.AppendAgentMessage(ctx, "chat-new", api.Message{Role: "assistant", Content: "new answer"}, "qwen3"); err != nil {
		t.Fatal(err)
	}

	chat, err := store.LatestChat(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if chat.ID != "chat-new" || chat.Model != "qwen3" {
		t.Fatalf("latest chat = %#v, want chat-new with qwen3", chat)
	}

	chat, err = store.LatestChatForModel(ctx, "llama3.2")
	if err != nil {
		t.Fatal(err)
	}
	if chat.ID != "chat-old" {
		t.Fatalf("llama latest chat = %q, want chat-old", chat.ID)
	}
	if _, err := store.LatestChatForModel(ctx, "missing"); !errors.Is(err, sql.ErrNoRows) {
		t.Fatalf("missing model err = %v, want sql.ErrNoRows", err)
	}

	summaries, err := store.ListChats(ctx, 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(summaries) != 2 {
		t.Fatalf("summaries = %d, want 2", len(summaries))
	}
	if summaries[0].ID != "chat-new" || summaries[0].Title != "new topic" || summaries[0].Model != "qwen3" {
		t.Fatalf("newest summary = %#v", summaries[0])
	}
	if summaries[1].ID != "chat-old" || summaries[1].Model != "llama3.2" {
		t.Fatalf("older summary = %#v", summaries[1])
	}

	future := time.Date(2099, 1, 1, 12, 0, 0, 0, time.UTC)
	if _, err := store.db.conn.ExecContext(ctx, `
		INSERT INTO chats (id, title, created_at)
		VALUES (?, ?, ?)
	`, "chat-archived", "archived topic", future); err != nil {
		t.Fatal(err)
	}
	if _, err := store.db.conn.ExecContext(ctx, `
		INSERT INTO messages (chat_id, role, content, model_name, created_at, updated_at, archived)
		VALUES (?, ?, ?, ?, ?, ?, 1)
	`, "chat-archived", "assistant", "archived answer", "ghost-model", future, future); err != nil {
		t.Fatal(err)
	}

	chat, err = store.LatestChat(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if chat.ID != "chat-new" {
		t.Fatalf("latest chat = %q, want chat-new after archived future row", chat.ID)
	}
	if _, err := store.LatestChatForModel(ctx, "ghost-model"); !errors.Is(err, sql.ErrNoRows) {
		t.Fatalf("archived model err = %v, want sql.ErrNoRows", err)
	}
	summaries, err = store.ListChats(ctx, 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(summaries) != 2 {
		t.Fatalf("summaries = %d, want archived-only chat hidden", len(summaries))
	}
}

func TestAgentStoreUpdateLastMessageIgnoresArchivedRows(t *testing.T) {
	store := newTestAgentStore(t)
	ctx := context.Background()

	if err := store.AppendAgentMessage(ctx, "chat-1", api.Message{Role: "assistant", Content: "active"}, "llama3.2"); err != nil {
		t.Fatal(err)
	}

	future := time.Date(2099, 1, 1, 12, 0, 0, 0, time.UTC)
	if _, err := store.db.conn.ExecContext(ctx, `
		INSERT INTO messages (chat_id, role, content, created_at, updated_at, archived)
		VALUES (?, ?, ?, ?, ?, 1)
	`, "chat-1", "assistant", "archived", future, future); err != nil {
		t.Fatal(err)
	}

	if err := store.UpdateLastAgentMessage(ctx, "chat-1", api.Message{Role: "assistant", Content: "active updated"}, "llama3.2"); err != nil {
		t.Fatal(err)
	}

	chat, err := store.AgentChat(ctx, "chat-1")
	if err != nil {
		t.Fatal(err)
	}
	if len(chat.Messages) != 1 || chat.Messages[0].Content != "active updated" {
		t.Fatalf("active messages = %#v, want updated active message only", chat.Messages)
	}

	var archivedContent string
	if err := store.db.conn.QueryRowContext(ctx, `
		SELECT content
		FROM messages
		WHERE chat_id = ? AND archived = 1
		ORDER BY id DESC
		LIMIT 1
	`, "chat-1").Scan(&archivedContent); err != nil {
		t.Fatal(err)
	}
	if archivedContent != "archived" {
		t.Fatalf("archived content = %q, want archived", archivedContent)
	}
}

func TestAgentStoreListUserMessages(t *testing.T) {
	store := newTestAgentStore(t)
	ctx := context.Background()

	for _, msg := range []api.Message{
		{Role: "user", Content: "old prompt"},
		{Role: "assistant", Content: "not user"},
		{Role: "user", Content: "middle prompt"},
		{Role: "user", Content: "   "},
		{Role: "user", Content: agent.CompactionSummaryMessagePrefix + "old context"},
		{Role: "user", Content: "new prompt"},
	} {
		if err := store.AppendAgentMessage(ctx, "chat-1", msg, ""); err != nil {
			t.Fatal(err)
		}
	}

	if _, err := store.db.conn.ExecContext(ctx, `UPDATE messages SET archived = 1 WHERE content = ?`, "middle prompt"); err != nil {
		t.Fatal(err)
	}

	messages, err := store.ListUserMessages(ctx, 2)
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"old prompt", "new prompt"}
	if !slices.Equal(messages, want) {
		t.Fatalf("messages = %#v, want %#v", messages, want)
	}
}

func TestAgentStoreArchivesCompactedMessages(t *testing.T) {
	store := newTestAgentStore(t)
	ctx := context.Background()

	for _, msg := range []api.Message{
		{Role: "user", Content: "old request"},
		{Role: "assistant", Content: "old answer"},
		{Role: "user", Content: "recent request"},
	} {
		if err := store.AppendAgentMessage(ctx, "chat-1", msg, ""); err != nil {
			t.Fatal(err)
		}
	}

	if err := store.ArchiveForCompaction(ctx, "chat-1", 1, "summary", true); err != nil {
		t.Fatal(err)
	}

	agentChat, err := store.AgentChat(ctx, "chat-1")
	if err != nil {
		t.Fatal(err)
	}
	if len(agentChat.Messages) != 3 {
		t.Fatalf("active messages = %#v, want compaction pair plus latest request", agentChat.Messages)
	}
	if agentChat.Messages[0].Role != "assistant" || len(agentChat.Messages[0].ToolCalls) != 1 || agentChat.Messages[0].ToolCalls[0].Function.Name != agent.CompactionToolName {
		t.Fatalf("summary tool call = %#v", agentChat.Messages[0])
	}
	content := agentChat.Messages[1].Content
	if !strings.Contains(content, agent.CompactionContinueInstruction) {
		t.Fatalf("summary tool result missing continuation instruction: %q", content)
	}
	if agentChat.Messages[2].Content != "recent request" {
		t.Fatalf("kept message = %#v, want recent request", agentChat.Messages[2])
	}

	appChat, err := store.Chat("chat-1")
	if err != nil {
		t.Fatal(err)
	}
	if len(appChat.Messages) != 3 {
		t.Fatalf("app chat should hide archived messages too: %#v", appChat.Messages)
	}

	var idsJSON string
	if err := store.db.conn.QueryRowContext(ctx, `
		SELECT archived_message_ids FROM compactions WHERE chat_id = ?
	`, "chat-1").Scan(&idsJSON); err != nil {
		t.Fatal(err)
	}
	var ids []int64
	if err := json.Unmarshal([]byte(idsJSON), &ids); err != nil {
		t.Fatal(err)
	}
	if len(ids) != 2 {
		t.Fatalf("archived ids = %v, want 2 ids", ids)
	}
}
