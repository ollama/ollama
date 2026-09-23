//go:build windows || darwin

package history

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
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
		data, err := os.ReadFile(filepath.Join(result.Directory, filepath.FromSlash(name)))
		if err != nil {
			t.Fatal(err)
		}
		return data
	}
	original, _ := json.Marshal(chat)
	var restored store.Chat
	if err := json.Unmarshal(read("source-chat.json"), &restored); err != nil {
		t.Fatal(err)
	}
	reencoded, _ := json.Marshal(restored)
	if !bytes.Equal(original, reencoded) {
		t.Fatal("source archive did not preserve every conversation field")
	}
	var metadata manifest
	if err := json.Unmarshal(read("manifest.json"), &metadata); err != nil {
		t.Fatal(err)
	}
	if !metadata.Complete || len(result.Warnings) != 0 || metadata.MessageCount != 3 || len(metadata.Attachments) != 3 {
		t.Fatalf("wrong manifest: %+v", metadata)
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
	if strings.Contains(string(read("conversation.html")), "<script>") || !strings.Contains(string(read("continue.txt")), "My next question:") {
		t.Fatal("unsafe preview or missing continuation prompt")
	}
	second, err := Export(chat, parent)
	if err != nil || second.Directory == result.Directory {
		t.Fatalf("second export overwrote the first: %+v, %v", second, err)
	}
	if !bytes.Equal(read("source-chat.json"), mustRead(t, filepath.Join(second.Directory, "source-chat.json"))) {
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
	var metadata manifest
	if err := json.Unmarshal(mustRead(t, filepath.Join(result.Directory, "manifest.json")), &metadata); err != nil {
		t.Fatal(err)
	}
	if metadata.Complete || metadata.Attachments[0].Status != "missing" {
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

func mustRead(t *testing.T, path string) []byte {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return data
}
