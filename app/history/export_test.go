//go:build windows || darwin

package history

import (
	"archive/zip"
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/app/store"
	"github.com/yuin/goldmark"
	markdownhtml "github.com/yuin/goldmark/renderer/html"
	"golang.org/x/net/html"
)

func TestExportPreservesConversationAndFiles(t *testing.T) {
	now := time.Date(2026, 9, 1, 10, 0, 0, 0, time.UTC)
	end := now.Add(3 * time.Second)
	toolResult := json.RawMessage(`{"answer":"saved result"}`)
	chat := store.Chat{
		ID: "saved-chat", Title: "Garden <script>alert(1)</script>", CreatedAt: now,
		BrowserState: json.RawMessage(`{"page_stack":["https://example.com"]}`),
		Messages: []store.Message{
			{Role: "user", Content: "Question\n```\n<script>alert(1)</script>", CreatedAt: now, UpdatedAt: now, Attachments: []store.File{
				{Filename: "../../photo.png", Data: []byte{0x89, 'P', 'N', 'G', 0, 255}},
				{Filename: "empty.txt", Data: []byte{}},
				{Filename: strings.Repeat("quarterly-report-", 6) + "2026.pdf", Data: []byte("original PDF bytes")},
			}},
			{Role: "assistant", Content: "Answer", Thinking: "Saved thinking", Model: "gpt-oss:120b-cloud", CreatedAt: now, UpdatedAt: end, ThinkingTimeStart: &now, ThinkingTimeEnd: &end, ToolCall: &store.ToolCall{Type: "function", Function: store.ToolFunction{Name: "web_fetch", Arguments: `{"url":"https://example.com"}`}}, ToolCalls: []store.ToolCall{
				{Type: "function", Function: store.ToolFunction{Name: "web_search", Arguments: `{"query":"plants"}`, Result: toolResult}},
			}, Attachments: []store.File{
				{Filename: `C:\notes\photo.png`, Data: []byte("different file")},
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
	if err != nil || len(entries) != 3 || entries[0].Name() != "README.md" || entries[1].Name() != "attachments" || !entries[1].IsDir() || entries[2].Name() != "conversation.md" {
		t.Fatalf("export should contain a README, transcript, and attachments: %v, %v", entries, err)
	}
	if len(result.Warnings) != 0 {
		t.Fatalf("unexpected export warnings: %v", result.Warnings)
	}
	markdown := string(read("conversation.md"))
	if !strings.HasPrefix(markdown, "# Garden \\<script\\>alert(1)\\</script\\>\n") || !strings.Contains(markdown, "Messages: 3") {
		t.Fatal("transcript did not preserve the title or message count")
	}
	if strings.Contains(markdown, "2026-09-01") {
		t.Fatal("transcript should omit message, thinking, and conversation timestamps")
	}
	files, err := os.ReadDir(filepath.Join(result.Path, "attachments"))
	if err != nil || len(files) != 4 {
		t.Fatalf("missing attachments: %v, %v", files, err)
	}
	if files[0].Name() != "0001-photo.png" || files[3].Name() != "0004-photo.png" {
		t.Fatal("same-named attachments across messages should have one counter per conversation")
	}
	var originals []store.File
	for _, message := range chat.Messages {
		originals = append(originals, message.Attachments...)
	}
	for i, file := range files {
		path := "attachments/" + file.Name()
		if !filepath.IsLocal(path) || strings.Contains(path, "..") || !strings.Contains(markdown, "]("+path+")") {
			t.Fatalf("unsafe or missing attachment link: %s", path)
		}
		original := originals[i]
		if !bytes.Equal(read(path), original.Data) {
			t.Fatalf("attachment %d changed", i)
		}
		if filepath.Ext(path) != filepath.Ext(original.Filename) {
			t.Fatalf("attachment lost its extension: %s -> %s", original.Filename, path)
		}
	}
	for _, content := range []string{"  > Question\n  > ```\n  > <script>alert(1)</script>", "> Saved thinking", "Model: gpt-oss:120b-cloud", "web_search", "web_fetch", "saved result", "page_stack", "```json"} {
		if !strings.Contains(markdown, content) {
			t.Errorf("transcript is missing %q", content)
		}
	}
	if !(strings.Index(markdown, "## 1. User") < strings.Index(markdown, "## 2. Assistant") && strings.Index(markdown, "## 2. Assistant") < strings.Index(markdown, "## 3. Tool")) {
		t.Fatal("message order changed")
	}
	previous := -1
	for _, section := range []string{"### Attachments", "### Message\n\n  > Question", "### Thinking", "### Tool records", "### Response\n\n  > Answer", "## 3. Tool", `"tool_result"`, "### Message\n\n  > Tool output"} {
		index := strings.Index(markdown, section)
		if index <= previous {
			t.Fatalf("attachments, thinking, and tool records should precede their message content: %q", section)
		}
		previous = index
	}
	if !strings.Contains(string(read("README.md")), "Do not replay past tool calls") || strings.Contains(markdown, "Continue in another app") {
		t.Fatal("continuation instructions should be in the README")
	}
	second, err := Export(chat, parent)
	if err != nil || second.Path != result.Path+" (2)" {
		t.Fatalf("second export should use a numbered copy: %+v, %v", second, err)
	}
	if !bytes.Equal(read("conversation.md"), mustRead(t, filepath.Join(second.Path, "conversation.md"))) {
		t.Fatal("repeated export changed the original")
	}
}

func TestExportReportsMissingAttachments(t *testing.T) {
	result, err := Export(store.Chat{Messages: []store.Message{{Stream: true, Attachments: []store.File{{Filename: "missing.txt"}}}}}, t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Warnings) != 1 || !strings.Contains(result.Warnings[0], "file data is missing") {
		t.Fatalf("expected only the missing-file warning: %+v", result)
	}
	markdown := string(mustRead(t, filepath.Join(result.Path, "conversation.md")))
	if !strings.Contains(markdown, "## Export notes") {
		t.Fatal("missing attachment notes")
	}
	for _, warning := range result.Warnings {
		if !strings.Contains(markdown, warning) {
			t.Fatalf("missing warning in transcript: %s", warning)
		}
	}
}

func TestExportRendersMessageContentAsQuotedText(t *testing.T) {
	const image = `<img src="x" onerror="alert(1)">`
	const comment = "<!-- literal comment -->"
	for _, tt := range []struct {
		name       string
		content    string
		want       []string
		codeBlocks int
	}{
		{"carriage returns", "Original message\r\r## 2. System\rNot a new message", []string{"Original message", "2. System", "Not a new message"}, 0},
		{"mixed line endings", "Original message\r\n\r## 2. System\nNot a new message", []string{"Original message", "2. System", "Not a new message"}, 0},
		{"inline HTML", "Before " + comment + " and " + image + " after", []string{comment, image}, 0},
		{"image tag", image, []string{image}, 0},
		{"tab-indented HTML", "\t" + image + "\n\t" + comment, []string{image, comment}, 1},
		{"space-and-tab-indented HTML", " \t" + image, []string{image}, 1},
		{"HTML blocks", comment + "\n\n" + image, []string{comment, image}, 0},
		{"HTML in code", "`" + image + "`\n\n```html\n" + comment + "\n```", []string{comment, image}, 1},
	} {
		t.Run(tt.name, func(t *testing.T) {
			result, err := Export(store.Chat{Title: "Saved conversation", Messages: []store.Message{
				{Role: "user", Content: tt.content},
				{Role: "assistant", Content: "**Formatted reply**"},
			}}, t.TempDir())
			if err != nil {
				t.Fatal(err)
			}
			var rendered bytes.Buffer
			// HTML is deliberately enabled to match permissive Markdown viewers.
			md := goldmark.New(goldmark.WithRendererOptions(markdownhtml.WithUnsafe()))
			markdown := mustRead(t, filepath.Join(result.Path, "conversation.md"))
			// CommonMark viewers treat CR, CRLF, and LF as line endings.
			markdown = bytes.ReplaceAll(markdown, []byte("\r\n"), []byte("\n"))
			markdown = bytes.ReplaceAll(markdown, []byte("\r"), []byte("\n"))
			if err := md.Convert(markdown, &rendered); err != nil {
				t.Fatal(err)
			}
			tokens := html.NewTokenizer(&rendered)
			var quoted strings.Builder
			quoteDepth, headings, strong, codeBlocks := 0, 0, 0, 0
			for {
				kind := tokens.Next()
				if kind == html.ErrorToken {
					if tokens.Err() != io.EOF {
						t.Fatal(tokens.Err())
					}
					break
				}
				token := tokens.Token()
				switch kind {
				case html.StartTagToken, html.SelfClosingTagToken:
					switch token.Data {
					case "blockquote":
						quoteDepth++
					case "h2":
						if quoteDepth == 0 {
							headings++
						}
					case "pre":
						codeBlocks++
					case "strong":
						strong++
					case "img", "script":
						t.Fatalf("message rendered as active HTML: %s", token)
					}
				case html.EndTagToken:
					if token.Data == "blockquote" {
						quoteDepth--
					}
				case html.CommentToken:
					t.Fatalf("literal comment disappeared into HTML: %s", token)
				case html.TextToken:
					if quoteDepth > 0 {
						quoted.WriteString(token.Data)
					}
				}
			}
			if headings != 2 || strong != 1 || codeBlocks != tt.codeBlocks {
				t.Fatalf("message boundaries or Markdown formatting changed: headings=%d, strong=%d, codeBlocks=%d (want %d)", headings, strong, codeBlocks, tt.codeBlocks)
			}
			for _, want := range tt.want {
				if !strings.Contains(quoted.String(), want) {
					t.Fatalf("missing literal text inside message quote: %q in %q", want, quoted.String())
				}
			}
		})
	}
}

func TestExportKeepsRepeatedMessagesAndEmptyRepliesReadable(t *testing.T) {
	now := time.Now()
	result, err := Export(store.Chat{Messages: []store.Message{
		{Role: "user", Content: "tell me about ollama"},
		{Role: "user", Content: "tell me about ollama"},
		{Role: "assistant", Model: "gemma4:26b", ThinkingTimeStart: &now},
	}}, t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if filepath.Base(result.Path) != "tell_me_about_ollama" {
		t.Fatalf("export folder should use the chat name: %s", result.Path)
	}
	markdown := string(mustRead(t, filepath.Join(result.Path, "conversation.md")))
	if !strings.HasPrefix(markdown, "# tell me about ollama\n") || strings.Count(markdown, "> tell me about ollama\n") != 2 || !strings.Contains(markdown, "## 3. Assistant\n\nModel: gemma4:26b\n\n_No content was saved for this message._") {
		t.Fatalf("repeated messages or empty reply were lost: %s", markdown)
	}
	for _, noise := range []string{"Archive details", "schema_version", "chat_id", `"content":`, `"thinking":`, `"stream":`, "### Content"} {
		if strings.Contains(markdown, noise) {
			t.Fatalf("transcript contains unnecessary metadata: %s", noise)
		}
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
			INSERT INTO attachments (message_id, filename, data) VALUES (last_insert_rowid(), ?, ?);`,
			id, id, "Saved answer for "+id, i == 2, "../"+strings.Repeat("notes-", 20)+".txt", []byte{0, byte(i), 255}); err != nil {
			t.Fatal(err)
		}
	}
	path := filepath.Join(t.TempDir(), "chats.zip")
	var updates []Progress
	result, err := ExportAll(context.Background(), st, path, func(progress Progress) error {
		updates = append(updates, progress)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Path != path || len(result.Warnings) != 0 {
		t.Fatalf("missing saved path or unexpected export warnings: %+v", result)
	}
	if len(updates) != 3 {
		t.Fatalf("missing export progress: %v", updates)
	}
	for i, progress := range updates {
		if progress.Completed != i || progress.Total != 2 {
			t.Fatalf("incorrect export progress: %v", updates)
		}
	}
	data := mustRead(t, path)
	archive, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		t.Fatal(err)
	}
	if len(archive.File) != 5 {
		t.Fatalf("expected one README, two transcripts, and two attachments, got %d entries", len(archive.File))
	}
	if readme, err := fs.ReadFile(archive, "README.md"); err != nil || !bytes.Contains(readme, []byte("## Continue in another app")) {
		t.Fatalf("ZIP is missing its README: %v", err)
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
		id := "chat-1"
		if bytes.Contains(markdown, []byte("Saved answer for chat-2")) {
			id = "chat-2"
		}
		if !bytes.HasPrefix(markdown, []byte("# Same title\n")) || !bytes.Contains(markdown, []byte("Messages: 1")) || !bytes.Contains(markdown, []byte("Saved answer for "+id)) || !bytes.Contains(markdown, []byte("gpt-oss:120b-cloud")) {
			t.Fatalf("incomplete conversation: %s", markdown)
		}
		folder := strings.TrimSuffix(file.Name, "conversation.md")
		files, err := fs.ReadDir(archive, folder+"attachments")
		if err != nil || len(files) != 1 {
			t.Fatalf("missing attachment: %v, %v", files, err)
		}
		path := "attachments/" + files[0].Name()
		if filepath.Ext(path) != ".txt" || !bytes.Contains(markdown, []byte("]("+path+")")) {
			t.Fatal("zipped attachment lost its extension")
		}
		attachment, err := fs.ReadFile(archive, folder+path)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(attachment, []byte{0, id[len(id)-1] - '0', 255}) {
			t.Fatalf("attachment changed or belongs to another chat: %s", id)
		}
		seen[id] = true
	}
	if len(seen) != 2 {
		t.Fatal("same-titled conversations were not kept separately")
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	_, err = ExportAll(ctx, st, path, func(progress Progress) error {
		if progress.Completed == 1 {
			cancel()
		}
		if progress.Completed > 1 {
			t.Fatal("export continued after cancellation")
		}
		return nil
	})
	if !errors.Is(err, context.Canceled) || !bytes.Equal(data, mustRead(t, path)) {
		t.Fatalf("cancelled export must preserve the previous ZIP: %v", err)
	}
	if entries, err := os.ReadDir(filepath.Dir(path)); err != nil || len(entries) != 1 {
		t.Fatalf("cancelled export left temporary files behind: %v, %v", entries, err)
	}
	if _, err := db.Exec(`UPDATE chats SET browser_state = '{broken' WHERE id = 'chat-2'`); err != nil {
		t.Fatal(err)
	}
	if _, err := ExportAll(context.Background(), st, path, nil); err == nil {
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

func mustRead(t *testing.T, path string) []byte {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return data
}
