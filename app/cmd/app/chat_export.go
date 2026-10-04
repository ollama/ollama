//go:build windows || darwin

package main

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/ollama/ollama/app/store"
)

// maxChatFileNameLength keeps exported file names readable in Finder and
// well under file system limits.
const maxChatFileNameLength = 80

// exportChats writes each saved chat to dir as a Markdown file and returns how
// many it wrote. A chat's attachments are saved in a folder next to its file.
// Exporting into the same folder again replaces the files it wrote before.
func exportChats(st *store.Store, dir string) (int, error) {
	summaries, err := st.Chats()
	if err != nil {
		return 0, fmt.Errorf("read chats: %w", err)
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return 0, fmt.Errorf("create export folder: %w", err)
	}

	names := make(map[string]bool, len(summaries))
	exported := 0
	var errs []error
	for _, summary := range summaries {
		chat, err := st.Chat(summary.ID)
		if err != nil {
			errs = append(errs, fmt.Errorf("read chat %q: %w", summary.Title, err))
			continue
		}
		if len(chat.Messages) == 0 {
			continue
		}
		name := uniqueFileName(chatFileName(chat), names)
		if err := writeChatMarkdown(dir, name, chat); err != nil {
			errs = append(errs, fmt.Errorf("export chat %q: %w", name, err))
			continue
		}
		exported++
	}
	return exported, errors.Join(errs...)
}

// writeChatMarkdown writes chat to dir/name.md, and its attachments to
// dir/"name attachments".
func writeChatMarkdown(dir, name string, chat *store.Chat) error {
	attachmentDir := name + " attachments"
	attachmentNames := make(map[string]bool)
	saveAttachment := func(file store.File) (string, error) {
		fileName := uniqueFileName(sanitizeFileName(filepath.Base(file.Filename), "Attachment"), attachmentNames)
		if err := os.MkdirAll(filepath.Join(dir, attachmentDir), 0o755); err != nil {
			return "", err
		}
		if err := os.WriteFile(filepath.Join(dir, attachmentDir, fileName), file.Data, 0o644); err != nil {
			return "", err
		}
		return attachmentDir + "/" + fileName, nil
	}

	markdown, err := chatMarkdown(chat, saveAttachment)
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, name+".md"), []byte(markdown), 0o644)
}

// chatMarkdown renders chat as Markdown. saveAttachment stores an attachment
// and returns the relative path to link to it.
func chatMarkdown(chat *store.Chat, saveAttachment func(store.File) (string, error)) (string, error) {
	var b strings.Builder
	fmt.Fprintf(&b, "# %s\n\n", chatTitle(chat))
	if !chat.CreatedAt.IsZero() {
		fmt.Fprintf(&b, "%s\n", chat.CreatedAt.Local().Format("January 2, 2006 at 3:04 PM"))
	}

	// Tool results follow the assistant message that asked for them and
	// arrive in the same order as its tool calls.
	var pendingTools []string
	speaker := ""
	for _, message := range chat.Messages {
		switch message.Role {
		case "user":
			speaker = "user"
			b.WriteString("\n## You\n\n")
			writeMarkdownText(&b, message.Content)
			for _, file := range message.Attachments {
				path, err := saveAttachment(file)
				if err != nil {
					return "", fmt.Errorf("save attachment %q: %w", file.Filename, err)
				}
				link := "[" + file.Filename + "](<" + path + ">)"
				if isImageAttachment(file.Filename) {
					link = "!" + link
				}
				b.WriteString(link + "\n\n")
			}
		case "assistant":
			if speaker != "assistant" {
				speaker = "assistant"
				name := message.Model
				if name == "" {
					name = "Assistant"
				}
				fmt.Fprintf(&b, "\n## %s\n\n", name)
			}
			if thinking := strings.TrimSpace(message.Thinking); thinking != "" {
				writeDetails(&b, "Thinking", thinking, false)
			}
			writeMarkdownText(&b, message.Content)
			pendingTools = pendingTools[:0]
			for _, call := range message.ToolCalls {
				pendingTools = append(pendingTools, call.Function.Name)
				writeDetails(&b, "Used "+call.Function.Name, call.Function.Arguments, true)
			}
		case "tool":
			title := "Tool result"
			if len(pendingTools) > 0 {
				title = pendingTools[0] + " result"
				pendingTools = pendingTools[1:]
			}
			content := message.Content
			if strings.TrimSpace(content) == "" && message.ToolResult != nil {
				content = string(*message.ToolResult)
			}
			writeDetails(&b, title, content, true)
		default:
			speaker = message.Role
			fmt.Fprintf(&b, "\n## %s\n\n", upperFirst(message.Role))
			writeMarkdownText(&b, message.Content)
		}
	}
	return strings.TrimRight(b.String(), "\n") + "\n", nil
}

func writeMarkdownText(b *strings.Builder, text string) {
	if text = strings.TrimSpace(text); text != "" {
		b.WriteString(text + "\n\n")
	}
}

// writeDetails writes a collapsed section, so thinking and tool output don't
// crowd the conversation. Code is fenced so it keeps its formatting.
func writeDetails(b *strings.Builder, summary, body string, code bool) {
	body = strings.TrimSpace(body)
	if body == "" {
		return
	}
	fmt.Fprintf(b, "<details>\n<summary>%s</summary>\n\n", summary)
	if code {
		fence := codeFence(body)
		fmt.Fprintf(b, "%s\n%s\n%s\n", fence, body, fence)
	} else {
		b.WriteString(body + "\n")
	}
	b.WriteString("\n</details>\n\n")
}

// codeFence returns a backtick fence longer than any run of backticks in body.
func codeFence(body string) string {
	longest, run := 0, 0
	for _, r := range body {
		if r == '`' {
			run++
			longest = max(longest, run)
		} else {
			run = 0
		}
	}
	return strings.Repeat("`", max(3, longest+1))
}

func chatTitle(chat *store.Chat) string {
	if title := strings.TrimSpace(chat.Title); title != "" {
		return title
	}
	for _, message := range chat.Messages {
		if message.Role != "user" {
			continue
		}
		line, _, _ := strings.Cut(strings.TrimSpace(message.Content), "\n")
		if line = strings.TrimSpace(line); line != "" {
			return line
		}
	}
	return "Untitled Chat"
}

// chatFileName names an exported chat by the day it started and its title,
// so the files sort by date.
func chatFileName(chat *store.Chat) string {
	name := sanitizeFileName(chatTitle(chat), "Untitled Chat")
	if chat.CreatedAt.IsZero() {
		return name
	}
	return chat.CreatedAt.Local().Format("2006-01-02") + " " + name
}

// sanitizeFileName makes name safe to use as a file name on macOS and
// Windows, falling back to fallback when nothing is left.
func sanitizeFileName(name, fallback string) string {
	var b strings.Builder
	space := false
	for _, r := range name {
		if strings.ContainsRune(`/\:*?"<>|`, r) || unicode.IsControl(r) || unicode.IsSpace(r) {
			space = b.Len() > 0
			continue
		}
		if space {
			b.WriteByte(' ')
			space = false
		}
		b.WriteRune(r)
	}
	name = b.String()
	if utf8.RuneCountInString(name) > maxChatFileNameLength {
		name = string([]rune(name)[:maxChatFileNameLength])
	}
	// Leading dots hide files, and trailing dots and spaces are invalid on Windows.
	name = strings.Trim(name, ". ")
	if name == "" {
		return fallback
	}
	return name
}

// uniqueFileName returns name, or name with a number added when used already
// has it. File systems are often case-insensitive, so names are compared
// without case.
func uniqueFileName(name string, used map[string]bool) string {
	ext := filepath.Ext(name)
	base := strings.TrimSuffix(name, ext)
	candidate := name
	for i := 2; used[strings.ToLower(candidate)]; i++ {
		candidate = fmt.Sprintf("%s %d%s", base, i, ext)
	}
	used[strings.ToLower(candidate)] = true
	return candidate
}

func isImageAttachment(filename string) bool {
	switch strings.ToLower(filepath.Ext(filename)) {
	case ".png", ".jpg", ".jpeg", ".gif", ".webp":
		return true
	}
	return false
}

func upperFirst(s string) string {
	r, size := utf8.DecodeRuneInString(s)
	if r == utf8.RuneError {
		return s
	}
	return string(unicode.ToUpper(r)) + s[size:]
}
