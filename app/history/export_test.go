//go:build windows || darwin

package history

import (
	"archive/zip"
	"bytes"
	"crypto/sha256"
	"database/sql"
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/app/store"
)

func TestExportPreservesConversationAndFiles(t *testing.T) {
	now := time.Now().UTC()
	toolResult := json.RawMessage(`{"answer":"saved result"}`)
	chat := store.Chat{
		ID: "saved-chat", Title: "Garden <script>alert(1)</script>", CreatedAt: now,
		BrowserState: json.RawMessage(`{"page_stack":["https://example.com"]}`),
		Messages: []store.Message{
			{Role: "user", Content: "Question\n```\n<script>alert(1)</script>", CreatedAt: now, UpdatedAt: now, Attachments: []store.File{
				{Filename: "../../photo.png", Data: []byte{0x89, 'P', 'N', 'G', 0, 255}},
				{Filename: `C:\notes\photo.png`, Data: []byte("different file")},
				{Filename: "empty.txt", Data: []byte{}},
			}},
			{Role: "assistant", Content: "Answer", Thinking: "Saved thinking", Model: "gpt-oss:120b-cloud", CreatedAt: now, UpdatedAt: now, ThinkingTimeStart: &now, ThinkingTimeEnd: &now, ToolCalls: []store.ToolCall{
				{Type: "function", Function: store.ToolFunction{Name: "web_search", Arguments: `{"query":"plants"}`, Result: toolResult}},
			}},
			{Role: "tool", Content: "Tool output", ToolName: "web_search", ToolResult: &toolResult, CreatedAt: now, UpdatedAt: now},
		},
	}
	parent := t.TempDir()
	result, err := Export(chat, parent)
	if err != nil {
		t.Fatal(err)
	}
	read := func(name string) []byte {
		t.Helper()
		data, err := os.ReadFile(filepath.Join(result.Path, filepath.FromSlash(name)))
		if err != nil {
			t.Fatal(err)
		}
		return data
	}
	entries, err := os.ReadDir(result.Path)
	if err != nil || len(entries) != 2 || entries[0].Name() != "attachments" || !entries[0].IsDir() || entries[1].Name() != "conversation.md" {
		t.Fatalf("export should contain only Markdown and attachments: %v, %v", entries, err)
	}
	metadata := readArchiveDetails(t, result.Path)
	if !metadata.Complete || len(result.Warnings) != 0 || metadata.MessageCount != 3 || len(metadata.Attachments) != 3 {
		t.Fatalf("wrong manifest: %+v", metadata)
	}
	if metadata.ChatID != chat.ID || metadata.Title != chat.Title || !metadata.CreatedAt.Equal(now) {
		t.Fatal("transcript did not preserve conversation metadata")
	}
	for i, file := range metadata.Attachments {
		if !filepath.IsLocal(file.Path) || !strings.HasPrefix(file.Path, "attachments/") || strings.Contains(file.Path, "..") {
			t.Fatalf("unsafe attachment path: %s", file.Path)
		}
		data := read(file.Path)
		if !bytes.Equal(data, chat.Messages[0].Attachments[i].Data) || file.SHA256 != fmt.Sprintf("%x", sha256.Sum256(data)) || file.Status != "saved" {
			t.Fatalf("attachment %d changed", i)
		}
	}
	markdown := string(read("conversation.md"))
	for _, content := range []string{chat.Messages[0].Content, "Saved thinking", "gpt-oss:120b-cloud", "web_search", "saved result", "page_stack", "````text"} {
		if !strings.Contains(markdown, content) {
			t.Errorf("transcript is missing %q", content)
		}
	}
	if !(strings.Index(markdown, "## Message 1") < strings.Index(markdown, "## Message 2") && strings.Index(markdown, "## Message 2") < strings.Index(markdown, "## Message 3")) {
		t.Fatal("message order changed")
	}
	if !strings.Contains(markdown, "attach this file and any needed files from attachments/") {
		t.Fatal("transcript is missing the usage note")
	}
	second, err := Export(chat, parent)
	if err != nil || second.Path == result.Path {
		t.Fatalf("second export overwrote the first: %+v, %v", second, err)
	}
	if !bytes.Equal(read("conversation.md"), mustRead(t, filepath.Join(second.Path, "conversation.md"))) {
		t.Fatal("repeated export changed the original")
	}
}

func TestExportReportsIncompleteHistory(t *testing.T) {
	result, err := Export(store.Chat{Messages: []store.Message{{Stream: true, Attachments: []store.File{{Filename: "missing.txt"}}}}}, t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Warnings) != 2 {
		t.Fatalf("missing warnings: %+v", result)
	}
	metadata := readArchiveDetails(t, result.Path)
	if metadata.Complete || metadata.Attachments[0].Status != "missing" || len(metadata.Warnings) != 2 {
		t.Fatal("incomplete archive reported as complete")
	}
}

func TestExportRejectsCorruptHistory(t *testing.T) {
	parent := t.TempDir()
	if _, err := Export(store.Chat{BrowserState: json.RawMessage(`{broken`)}, parent); err == nil {
		t.Fatal("export silently discarded corrupt browser state")
	}
	entries, err := os.ReadDir(parent)
	if err != nil || len(entries) != 0 {
		t.Fatalf("failed export left files behind: %v", err)
	}
}

func TestExportAllPreservesChatsAndPreviousArchiveOnFailure(t *testing.T) {
	st := &store.Store{DBPath: filepath.Join(t.TempDir(), "app.sqlite")}
	defer st.Close()
	if _, err := st.Settings(); err != nil {
		t.Fatal(err)
	}
	db, err := sql.Open("sqlite3", st.DBPath)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	for i := 1; i <= 2; i++ {
		id := fmt.Sprintf("chat-%d", i)
		if _, err := db.Exec(`INSERT INTO chats (id, title) VALUES (?, 'Same title');
			INSERT INTO messages (chat_id, role, content, model_name, stream) VALUES (?, 'assistant', ?, 'gpt-oss:120b-cloud', ?);
			INSERT INTO attachments (message_id, filename, data) VALUES (last_insert_rowid(), '../notes.txt', ?);`,
			id, id, "Saved answer for "+id, i == 2, []byte{0, byte(i), 255}); err != nil {
			t.Fatal(err)
		}
	}
	path := filepath.Join(t.TempDir(), "chats.zip")
	result, err := ExportAll(st, path)
	if err != nil {
		t.Fatal(err)
	}
	if result.Path != path || len(result.Warnings) != 1 {
		t.Fatalf("missing saved path or unfinished-message warning: %+v", result)
	}
	data := mustRead(t, path)
	archive, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		t.Fatal(err)
	}
	if len(archive.File) != 4 {
		t.Fatalf("expected two transcripts and two attachments, got %d entries", len(archive.File))
	}
	seen := map[string]bool{}
	for _, file := range archive.File {
		if !fs.ValidPath(file.Name) || strings.Contains(file.Name, "..") {
			t.Fatalf("unsafe ZIP path: %s", file.Name)
		}
		if !strings.HasSuffix(file.Name, "/conversation.md") {
			continue
		}
		markdown, err := fs.ReadFile(archive, file.Name)
		if err != nil {
			t.Fatal(err)
		}
		var metadata manifest
		start := bytes.IndexByte(markdown, '{')
		if start < 0 {
			t.Fatal("missing archive details")
		}
		if err := json.NewDecoder(bytes.NewReader(markdown[start:])).Decode(&metadata); err != nil {
			t.Fatal(err)
		}
		if metadata.Title != "Same title" || metadata.MessageCount != 1 || len(metadata.Attachments) != 1 || !bytes.Contains(markdown, []byte("Saved answer for "+metadata.ChatID)) || !bytes.Contains(markdown, []byte("gpt-oss:120b-cloud")) {
			t.Fatalf("incomplete conversation: %s", markdown)
		}
		folder := strings.TrimSuffix(file.Name, "conversation.md")
		attachment, err := fs.ReadFile(archive, folder+metadata.Attachments[0].Path)
		if err != nil {
			t.Fatal(err)
		}
		if metadata.ChatID != "chat-1" && metadata.ChatID != "chat-2" || !bytes.Equal(attachment, []byte{0, metadata.ChatID[len(metadata.ChatID)-1] - '0', 255}) {
			t.Fatalf("attachment changed or belongs to another chat: %s", metadata.ChatID)
		}
		seen[metadata.ChatID] = true
	}
	if len(seen) != 2 {
		t.Fatal("same-titled conversations were not kept separately")
	}
	if _, err := db.Exec(`UPDATE chats SET browser_state = '{broken' WHERE id = 'chat-2'`); err != nil {
		t.Fatal(err)
	}
	if _, err := ExportAll(st, path); err == nil {
		t.Fatal("export silently discarded corrupt history")
	}
	if !bytes.Equal(data, mustRead(t, path)) {
		t.Fatal("failed export replaced the previous ZIP")
	}
	entries, err := os.ReadDir(filepath.Dir(path))
	if err != nil || len(entries) != 1 {
		t.Fatalf("failed export left temporary files behind: %v, %v", entries, err)
	}
}

func readArchiveDetails(t *testing.T, directory string) manifest {
	t.Helper()
	markdown := string(mustRead(t, filepath.Join(directory, "conversation.md")))
	start := strings.Index(markdown, "{")
	if start < 0 {
		t.Fatal("transcript is missing archive details")
	}
	var metadata manifest
	if err := json.NewDecoder(strings.NewReader(markdown[start:])).Decode(&metadata); err != nil {
		t.Fatal(err)
	}
	return metadata
}

func mustRead(t *testing.T, path string) []byte {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return data
}
