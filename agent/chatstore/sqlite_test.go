package chatstore

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"path/filepath"
	"slices"
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
	if len(chat.Messages) != 4 {
		t.Fatalf("active messages = %d, want 4", len(chat.Messages))
	}
	if chat.Messages[0].Role != "user" {
		t.Fatalf("summary role = %q, want user", chat.Messages[0].Role)
	}
	if chat.Messages[0].Content != compactionSummaryMessagePrefix+"summary" {
		t.Fatalf("summary message = %#v", chat.Messages[0])
	}
	if chat.Messages[1].Content != "recent one" {
		t.Fatalf("first active message = %#v", chat.Messages[1])
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
