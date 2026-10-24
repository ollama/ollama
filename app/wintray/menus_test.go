//go:build windows

package wintray

import (
	"strings"
	"syscall"
	"testing"
)

func TestLogsCommandLineKeepsPathWithSpaces(t *testing.T) {
	dir := `C:\Users\Jane Doe\AppData\Local\Ollama`

	// Go quotes each argument with syscall.EscapeArg when it builds the
	// command line, so compare the exact string cmd.exe will receive.
	var escaped []string
	for _, arg := range logsCommandArgs(dir) {
		escaped = append(escaped, syscall.EscapeArg(arg))
	}
	got := strings.Join(escaped, " ")
	want := `/c start "" "C:\Users\Jane Doe\AppData\Local\Ollama"`
	if got != want {
		t.Errorf("command line = %s, want %s", got, want)
	}
}
