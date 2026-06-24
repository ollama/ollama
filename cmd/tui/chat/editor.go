package chat

import (
	"fmt"
	"os"
	"os/exec"
	"strings"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/ollama/ollama/envconfig"
)

type chatEditorDoneMsg struct {
	content string
	err     error
}

func (m chatModel) openInputEditor() (tea.Model, tea.Cmd) {
	cmd, path, cleanup, err := inputEditorCommand(string(m.input))
	if err != nil {
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "error", content: err.Error(), err: err.Error()}))
		m.status = "editor error"
		return m, nil
	}

	return m, tea.ExecProcess(cmd, func(err error) tea.Msg {
		defer cleanup()
		if err != nil {
			return chatEditorDoneMsg{err: fmt.Errorf("editor exited with error: %w", err)}
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return chatEditorDoneMsg{err: fmt.Errorf("reading editor content: %w", err)}
		}
		return chatEditorDoneMsg{content: strings.TrimRight(string(data), "\n")}
	})
}

func (m *chatModel) applyEditorResult(msg chatEditorDoneMsg) {
	if msg.err != nil {
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "error", content: msg.err.Error(), err: msg.err.Error()}))
		m.status = "editor error"
		return
	}
	m.input = []rune(msg.content)
	m.inputCursor = len(m.input)
	m.inputCursorSet = true
	m.complete = 0
	m.resetPromptHistoryCursor()
	m.syncInputPlaceholders()
	m.status = "editor"
}

func inputEditorCommand(content string) (*exec.Cmd, string, func(), error) {
	editor := inputEditorName()
	args := strings.Fields(editor)
	if len(args) == 0 {
		return nil, "", nil, fmt.Errorf("editor is empty, set OLLAMA_EDITOR to the path of your preferred editor")
	}
	if _, err := exec.LookPath(args[0]); err != nil {
		return nil, "", nil, fmt.Errorf("editor %q not found, set OLLAMA_EDITOR to the path of your preferred editor", args[0])
	}

	tmpFile, err := os.CreateTemp("", "ollama-prompt-*.txt")
	if err != nil {
		return nil, "", nil, fmt.Errorf("creating temp file: %w", err)
	}
	cleanup := func() { _ = os.Remove(tmpFile.Name()) }
	if content != "" {
		if _, err := tmpFile.WriteString(content); err != nil {
			tmpFile.Close()
			cleanup()
			return nil, "", nil, fmt.Errorf("writing to temp file: %w", err)
		}
	}
	if err := tmpFile.Close(); err != nil {
		cleanup()
		return nil, "", nil, fmt.Errorf("closing temp file: %w", err)
	}

	args = append(args, tmpFile.Name())
	return exec.Command(args[0], args[1:]...), tmpFile.Name(), cleanup, nil
}

func inputEditorName() string {
	if editor := strings.TrimSpace(envconfig.Editor()); editor != "" {
		return editor
	}
	if editor := strings.TrimSpace(os.Getenv("VISUAL")); editor != "" {
		return editor
	}
	if editor := strings.TrimSpace(os.Getenv("EDITOR")); editor != "" {
		return editor
	}
	return defaultEditor
}
