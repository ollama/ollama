package chatstore

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

	"github.com/ollama/ollama/api"
)

func TestStoreWritesAppCompatibleChat(t *testing.T) {
	store, err := New(filepath.Join(t.TempDir(), "db.sqlite"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	ctx := context.Background()
	if err := store.EnsureChat(ctx, "chat-1", ""); err != nil {
		t.Fatal(err)
	}
	if err := store.AppendMessage(ctx, "chat-1", api.Message{Role: "user", Content: "hello from cli"}, ""); err != nil {
		t.Fatal(err)
	}

	args := api.NewToolCallFunctionArguments()
	args.Set("command", "pwd")
	if err := store.AppendMessage(ctx, "chat-1", api.Message{
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

	chat, err := store.Chat(ctx, "chat-1")
	if err != nil {
		t.Fatal(err)
	}
	if len(chat.Messages) != 2 {
		t.Fatalf("messages = %d, want 2", len(chat.Messages))
	}
	if chat.Title != "hello from cli" {
		t.Fatalf("title = %q, want %q", chat.Title, "hello from cli")
	}
	if got := chat.Messages[1].ToolCalls[0].Function.Name; got != "bash" {
		t.Fatalf("tool name = %q, want bash", got)
	}
	if got := chat.Messages[1].ToolCalls[0].ID; got != "call-1" {
		t.Fatalf("tool call id = %q, want call-1", got)
	}

	var modelName sql.NullString
	if err := store.db.QueryRowContext(ctx, `
		SELECT model_name FROM messages WHERE chat_id = ? AND role = 'assistant'
	`, "chat-1").Scan(&modelName); err != nil {
		t.Fatal(err)
	}
	if !modelName.Valid || modelName.String != "llama3.2" {
		t.Fatalf("model_name = %#v, want llama3.2", modelName)
	}
}

func TestStoreRestoresToolMessageMetadata(t *testing.T) {
	store, err := New(filepath.Join(t.TempDir(), "db.sqlite"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	ctx := context.Background()
	if err := store.AppendMessage(ctx, "chat-1", api.Message{
		Role:       "tool",
		Content:    "tool output",
		ToolName:   "bash",
		ToolCallID: "call-1",
	}, ""); err != nil {
		t.Fatal(err)
	}

	chat, err := store.Chat(ctx, "chat-1")
	if err != nil {
		t.Fatal(err)
	}
	if len(chat.Messages) != 1 {
		t.Fatalf("messages = %d, want 1", len(chat.Messages))
	}
	if chat.Messages[0].ToolName != "bash" || chat.Messages[0].ToolCallID != "call-1" {
		t.Fatalf("tool metadata = %#v", chat.Messages[0])
	}
}

func TestStoreRoundTripsMessageImages(t *testing.T) {
	store, err := New(filepath.Join(t.TempDir(), "db.sqlite"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	ctx := context.Background()
	first := api.ImageData([]byte("first image"))
	second := api.ImageData([]byte{0, 1, 2, 3})
	if err := store.AppendMessage(ctx, "chat-1", api.Message{
		Role:    "user",
		Content: "look",
		Images:  []api.ImageData{first, second},
	}, ""); err != nil {
		t.Fatal(err)
	}

	chat, err := store.Chat(ctx, "chat-1")
	if err != nil {
		t.Fatal(err)
	}
	if len(chat.Messages) != 1 || len(chat.Messages[0].Images) != 2 {
		t.Fatalf("messages = %#v", chat.Messages)
	}
	if !bytes.Equal(chat.Messages[0].Images[0], first) || !bytes.Equal(chat.Messages[0].Images[1], second) {
		t.Fatalf("images = %#v, want %#v", chat.Messages[0].Images, []api.ImageData{first, second})
	}

	updated := api.ImageData([]byte("updated image"))
	if err := store.UpdateLastMessage(ctx, "chat-1", api.Message{
		Role:    "user",
		Content: "updated",
		Images:  []api.ImageData{updated},
	}, ""); err != nil {
		t.Fatal(err)
	}
	chat, err = store.Chat(ctx, "chat-1")
	if err != nil {
		t.Fatal(err)
	}
	if len(chat.Messages) != 1 || len(chat.Messages[0].Images) != 1 || !bytes.Equal(chat.Messages[0].Images[0], updated) {
		t.Fatalf("updated images = %#v", chat.Messages)
	}
}

func TestStoreMigratesOldMessagesSchemaForImages(t *testing.T) {
	path := filepath.Join(t.TempDir(), "db.sqlite")
	db, err := sql.Open("sqlite3", path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(`
		CREATE TABLE chats (
			id TEXT PRIMARY KEY,
			title TEXT NOT NULL DEFAULT '',
			created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
			browser_state TEXT
		);

		CREATE TABLE messages (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			chat_id TEXT NOT NULL,
			role TEXT NOT NULL,
			content TEXT NOT NULL DEFAULT '',
			thinking TEXT NOT NULL DEFAULT '',
			stream BOOLEAN NOT NULL DEFAULT 0,
			model_name TEXT,
			model_cloud BOOLEAN,
			model_ollama_host BOOLEAN,
			created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
			updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
			thinking_time_start TIMESTAMP,
			thinking_time_end TIMESTAMP,
			tool_result TEXT,
			tool_name TEXT NOT NULL DEFAULT '',
			tool_call_id TEXT NOT NULL DEFAULT '',
			archived BOOLEAN NOT NULL DEFAULT 0,
			FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
		);
	`); err != nil {
		db.Close()
		t.Fatal(err)
	}
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}

	store, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	var found bool
	rows, err := store.db.Query(`PRAGMA table_info(messages)`)
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()
	for rows.Next() {
		var cid int
		var name string
		var typ string
		var notNull int
		var defaultValue any
		var pk int
		if err := rows.Scan(&cid, &name, &typ, &notNull, &defaultValue, &pk); err != nil {
			t.Fatal(err)
		}
		if name == "images" {
			found = true
		}
	}
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}
	if !found {
		t.Fatal("images column was not added")
	}
	found = false
	rows, err = store.db.Query(`PRAGMA table_info(chats)`)
	if err != nil {
		t.Fatal(err)
	}
	defer rows.Close()
	for rows.Next() {
		var cid int
		var name string
		var typ string
		var notNull int
		var defaultValue any
		var pk int
		if err := rows.Scan(&cid, &name, &typ, &notNull, &defaultValue, &pk); err != nil {
			t.Fatal(err)
		}
		if name == "model_name" {
			found = true
		}
	}
	if err := rows.Err(); err != nil {
		t.Fatal(err)
	}
	if !found {
		t.Fatal("chat model_name column was not added")
	}

	ctx := context.Background()
	image := api.ImageData([]byte("after migration"))
	if err := store.AppendMessage(ctx, "chat-1", api.Message{Role: "user", Content: "image", Images: []api.ImageData{image}}, ""); err != nil {
		t.Fatal(err)
	}
	chat, err := store.Chat(ctx, "chat-1")
	if err != nil {
		t.Fatal(err)
	}
	if len(chat.Messages) != 1 || len(chat.Messages[0].Images) != 1 || !bytes.Equal(chat.Messages[0].Images[0], image) {
		t.Fatalf("migrated images = %#v", chat.Messages)
	}
}

func TestStoreLatestChatForModel(t *testing.T) {
	store, err := New(filepath.Join(t.TempDir(), "db.sqlite"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	ctx := context.Background()
	if err := store.AppendMessage(ctx, "chat-old", api.Message{Role: "assistant", Content: "old"}, "llama3.2"); err != nil {
		t.Fatal(err)
	}
	if err := store.AppendMessage(ctx, "chat-other", api.Message{Role: "assistant", Content: "other"}, "qwen3"); err != nil {
		t.Fatal(err)
	}
	if err := store.AppendMessage(ctx, "chat-new", api.Message{Role: "assistant", Content: "new"}, "llama3.2"); err != nil {
		t.Fatal(err)
	}

	chat, err := store.LatestChatForModel(ctx, "llama3.2")
	if err != nil {
		t.Fatal(err)
	}
	if chat.ID != "chat-new" {
		t.Fatalf("chat ID = %q, want chat-new", chat.ID)
	}
	if chat.Model != "llama3.2" {
		t.Fatalf("chat model = %q, want llama3.2", chat.Model)
	}

	if _, err := store.LatestChatForModel(ctx, "missing"); !errors.Is(err, sql.ErrNoRows) {
		t.Fatalf("missing model err = %v, want sql.ErrNoRows", err)
	}
}

func TestStoreLatestChat(t *testing.T) {
	store, err := New(filepath.Join(t.TempDir(), "db.sqlite"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	ctx := context.Background()
	if err := store.AppendMessage(ctx, "chat-old", api.Message{Role: "assistant", Content: "old"}, "llama3.2"); err != nil {
		t.Fatal(err)
	}
	if err := store.AppendMessage(ctx, "chat-new", api.Message{Role: "assistant", Content: "new"}, "qwen3:8b"); err != nil {
		t.Fatal(err)
	}

	chat, err := store.LatestChat(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if chat.ID != "chat-new" {
		t.Fatalf("chat ID = %q, want chat-new", chat.ID)
	}
	if chat.Model != "qwen3:8b" {
		t.Fatalf("chat model = %q, want qwen3:8b", chat.Model)
	}
}

func TestStoreLatestChatSkipsChatsWithoutModel(t *testing.T) {
	store, err := New(filepath.Join(t.TempDir(), "db.sqlite"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	ctx := context.Background()
	if err := store.AppendMessage(ctx, "chat-1", api.Message{Role: "user", Content: "orphaned user turn"}, ""); err != nil {
		t.Fatal(err)
	}

	if _, err := store.LatestChat(ctx); !errors.Is(err, sql.ErrNoRows) {
		t.Fatalf("latest chat err = %v, want sql.ErrNoRows", err)
	}
}

func TestStoreSetChatModel(t *testing.T) {
	store, err := New(filepath.Join(t.TempDir(), "db.sqlite"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	ctx := context.Background()
	if err := store.EnsureChat(ctx, "empty", ""); err != nil {
		t.Fatal(err)
	}
	if err := store.SetChatModel(ctx, "empty", "qwen3"); err != nil {
		t.Fatal(err)
	}
	summaries, err := store.ListChats(ctx, 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(summaries) != 0 {
		t.Fatalf("empty chat summaries = %#v, want none", summaries)
	}
	if _, err := store.LatestChat(ctx); !errors.Is(err, sql.ErrNoRows) {
		t.Fatalf("empty latest chat err = %v, want sql.ErrNoRows", err)
	}

	if err := store.AppendMessage(ctx, "chat-1", api.Message{Role: "user", Content: "hello"}, ""); err != nil {
		t.Fatal(err)
	}
	if err := store.AppendMessage(ctx, "chat-1", api.Message{Role: "assistant", Content: "hi"}, "llama3.2"); err != nil {
		t.Fatal(err)
	}
	if err := store.SetChatModel(ctx, "chat-1", "qwen3"); err != nil {
		t.Fatal(err)
	}

	chat, err := store.Chat(ctx, "chat-1")
	if err != nil {
		t.Fatal(err)
	}
	if chat.Model != "qwen3" {
		t.Fatalf("chat model = %q, want qwen3", chat.Model)
	}
	if len(chat.Messages) != 2 || chat.Messages[0].Content != "hello" || chat.Messages[1].Content != "hi" {
		t.Fatalf("chat messages changed: %#v", chat.Messages)
	}

	chat, err = store.LatestChat(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if chat.ID != "chat-1" || chat.Model != "qwen3" {
		t.Fatalf("latest chat = %#v, want chat-1 with qwen3", chat)
	}

	chat, err = store.LatestChatForModel(ctx, "qwen3")
	if err != nil {
		t.Fatal(err)
	}
	if chat.ID != "chat-1" {
		t.Fatalf("latest qwen chat = %q, want chat-1", chat.ID)
	}
	if _, err := store.LatestChatForModel(ctx, "llama3.2"); !errors.Is(err, sql.ErrNoRows) {
		t.Fatalf("old message model err = %v, want sql.ErrNoRows", err)
	}

	summaries, err = store.ListChats(ctx, 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(summaries) != 1 || summaries[0].ID != "chat-1" || summaries[0].Model != "qwen3" {
		t.Fatalf("summaries = %#v, want chat-1 with qwen3", summaries)
	}
}

func TestStoreListChats(t *testing.T) {
	store, err := New(filepath.Join(t.TempDir(), "db.sqlite"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	ctx := context.Background()
	if err := store.AppendMessage(ctx, "chat-old", api.Message{Role: "user", Content: "old topic"}, ""); err != nil {
		t.Fatal(err)
	}
	if err := store.AppendMessage(ctx, "chat-old", api.Message{Role: "assistant", Content: "old answer"}, "llama3.2"); err != nil {
		t.Fatal(err)
	}
	if err := store.AppendMessage(ctx, "chat-new", api.Message{Role: "user", Content: "new topic"}, ""); err != nil {
		t.Fatal(err)
	}
	if err := store.AppendMessage(ctx, "chat-new", api.Message{Role: "assistant", Content: "new answer"}, "qwen3"); err != nil {
		t.Fatal(err)
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
	if summaries[0].MessageCount != 2 {
		t.Fatalf("message count = %d, want 2", summaries[0].MessageCount)
	}
	if summaries[0].ApproxBytes <= 0 {
		t.Fatalf("approx bytes = %d, want positive", summaries[0].ApproxBytes)
	}
	if summaries[1].ID != "chat-old" || summaries[1].Model != "llama3.2" {
		t.Fatalf("older summary = %#v", summaries[1])
	}
}

func TestStoreListUserMessages(t *testing.T) {
	store, err := New(filepath.Join(t.TempDir(), "db.sqlite"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	ctx := context.Background()
	for _, msg := range []api.Message{
		{Role: "user", Content: "old prompt"},
		{Role: "assistant", Content: "not user"},
		{Role: "user", Content: "middle prompt"},
		{Role: "user", Content: "   "},
		{Role: "user", Content: compactionSummaryMessagePrefix + "old context"},
		{Role: "user", Content: "new prompt"},
	} {
		if err := store.AppendMessage(ctx, "chat-1", msg, ""); err != nil {
			t.Fatal(err)
		}
	}

	messages, err := store.ListUserMessages(ctx, 2)
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"middle prompt", "new prompt"}
	if !slices.Equal(messages, want) {
		t.Fatalf("messages = %#v, want %#v", messages, want)
	}
}

func TestStoreArchivesCompactedMessages(t *testing.T) {
	store, err := New(filepath.Join(t.TempDir(), "db.sqlite"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	ctx := context.Background()
	for _, msg := range []api.Message{
		{Role: "user", Content: "old one"},
		{Role: "assistant", Content: "old answer"},
		{Role: "user", Content: "recent one"},
		{Role: "assistant", Content: "recent answer"},
		{Role: "user", Content: "recent two"},
	} {
		if err := store.AppendMessage(ctx, "chat-1", msg, ""); err != nil {
			t.Fatal(err)
		}
	}

	if err := store.ArchiveForCompaction(ctx, "chat-1", 2, "summary"); err != nil {
		t.Fatal(err)
	}

	chat, err := store.Chat(ctx, "chat-1")
	if err != nil {
		t.Fatal(err)
	}
	if len(chat.Messages) != 5 {
		t.Fatalf("active messages = %d, want 5", len(chat.Messages))
	}
	if chat.Messages[0].Role != "assistant" || len(chat.Messages[0].ToolCalls) != 1 || chat.Messages[0].ToolCalls[0].Function.Name != compactionToolName {
		t.Fatalf("summary tool call = %#v", chat.Messages[0])
	}
	if chat.Messages[1].Role != "tool" || chat.Messages[1].ToolName != compactionToolName || chat.Messages[1].Content != compactionSummaryMessagePrefix+"summary" {
		t.Fatalf("summary tool result = %#v", chat.Messages[1])
	}
	if chat.Messages[2].Content != "recent one" {
		t.Fatalf("first kept message = %#v", chat.Messages[2])
	}

	var idsJSON string
	if err := store.db.QueryRowContext(ctx, `
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

func TestStoreArchivesWholeChatWhenKeepingZeroTurns(t *testing.T) {
	store, err := New(filepath.Join(t.TempDir(), "db.sqlite"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	ctx := context.Background()
	for _, msg := range []api.Message{
		{Role: "user", Content: "only request"},
		{Role: "assistant", Content: "only answer"},
	} {
		if err := store.AppendMessage(ctx, "chat-1", msg, ""); err != nil {
			t.Fatal(err)
		}
	}

	if err := store.ArchiveForCompaction(ctx, "chat-1", 0, "summary"); err != nil {
		t.Fatal(err)
	}

	chat, err := store.Chat(ctx, "chat-1")
	if err != nil {
		t.Fatal(err)
	}
	if len(chat.Messages) != 2 {
		t.Fatalf("active messages = %d, want compaction tool pair only", len(chat.Messages))
	}
	if chat.Messages[0].Role != "assistant" || len(chat.Messages[0].ToolCalls) != 1 || chat.Messages[0].ToolCalls[0].Function.Name != compactionToolName {
		t.Fatalf("summary tool call = %#v", chat.Messages[0])
	}
	if chat.Messages[1].Role != "tool" || chat.Messages[1].Content != compactionSummaryMessagePrefix+"summary" {
		t.Fatalf("summary tool result = %#v", chat.Messages[1])
	}
}

func TestStoreArchivesCompactionWithContinuation(t *testing.T) {
	store, err := New(filepath.Join(t.TempDir(), "db.sqlite"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	ctx := context.Background()
	for _, msg := range []api.Message{
		{Role: "user", Content: "old request"},
		{Role: "assistant", Content: "old answer"},
		{Role: "user", Content: "recent request"},
	} {
		if err := store.AppendMessage(ctx, "chat-1", msg, ""); err != nil {
			t.Fatal(err)
		}
	}

	if err := store.ArchiveForCompactionWithContinuation(ctx, "chat-1", 1, "summary", true); err != nil {
		t.Fatal(err)
	}

	chat, err := store.Chat(ctx, "chat-1")
	if err != nil {
		t.Fatal(err)
	}
	if len(chat.Messages) != 3 {
		t.Fatalf("active messages = %#v, want compaction pair plus latest request", chat.Messages)
	}
	content := chat.Messages[1].Content
	if !strings.Contains(content, compactionContinueInstruction) {
		t.Fatalf("summary tool result missing continuation instruction: %q", content)
	}
	if chat.Messages[2].Content != "recent request" {
		t.Fatalf("kept message = %#v, want recent request", chat.Messages[2])
	}

	var storedSummary string
	if err := store.db.QueryRowContext(ctx, `
		SELECT summary FROM compactions WHERE chat_id = ?
	`, "chat-1").Scan(&storedSummary); err != nil {
		t.Fatal(err)
	}
	if storedSummary != "summary" {
		t.Fatalf("stored summary = %q, want raw summary", storedSummary)
	}
}

func TestStoreReplacesPreviousCompactionMessages(t *testing.T) {
	store, err := New(filepath.Join(t.TempDir(), "db.sqlite"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	ctx := context.Background()
	for _, msg := range []api.Message{
		{Role: "user", Content: "old"},
		{Role: "assistant", Content: "old answer"},
		{Role: "user", Content: "recent"},
	} {
		if err := store.AppendMessage(ctx, "chat-1", msg, ""); err != nil {
			t.Fatal(err)
		}
	}
	if err := store.ArchiveForCompaction(ctx, "chat-1", 1, "first summary"); err != nil {
		t.Fatal(err)
	}
	if err := store.AppendMessage(ctx, "chat-1", api.Message{Role: "user", Content: "next"}, ""); err != nil {
		t.Fatal(err)
	}
	if err := store.ArchiveForCompaction(ctx, "chat-1", 1, "second summary"); err != nil {
		t.Fatal(err)
	}

	chat, err := store.Chat(ctx, "chat-1")
	if err != nil {
		t.Fatal(err)
	}
	if len(chat.Messages) != 3 {
		t.Fatalf("active messages = %#v, want compaction pair plus latest user", chat.Messages)
	}
	if chat.Messages[2].Content != "next" {
		t.Fatalf("kept message = %#v, want next user prompt", chat.Messages[2])
	}
	if chat.Messages[1].Content != compactionSummaryMessagePrefix+"second summary" {
		t.Fatalf("summary = %#v", chat.Messages[1])
	}
	for _, msg := range chat.Messages {
		if strings.Contains(msg.Content, "first summary") {
			t.Fatalf("old summary should have been archived: %#v", chat.Messages)
		}
	}
}
