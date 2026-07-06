package tools

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/ollama/ollama/agent"
)

func TestBashReportsFinalWorkingDir(t *testing.T) {
	root := t.TempDir()
	subdir := filepath.Join(root, "sub")
	if err := os.Mkdir(subdir, 0o755); err != nil {
		t.Fatal(err)
	}

	result, err := (&Bash{}).Execute(context.Background(), agent.ToolContext{WorkingDir: root}, map[string]any{
		"command": shellTestCommand("cd sub && pwd", "Set-Location sub; Get-Location"),
	})
	if err != nil {
		t.Fatal(err)
	}
	wantDir, err := filepath.EvalSymlinks(subdir)
	if err != nil {
		t.Fatal(err)
	}
	if result.WorkingDir != wantDir {
		t.Fatalf("working dir = %q, want %q", result.WorkingDir, wantDir)
	}
	if !strings.Contains(result.Content, "sub") {
		t.Fatalf("content = %q, want pwd output", result.Content)
	}
}

func TestBashBoundsOutputWhileRunning(t *testing.T) {
	result, err := (&Bash{}).Execute(context.Background(), agent.ToolContext{WorkingDir: t.TempDir()}, map[string]any{
		"command": shellTestCommand("yes x | head -c 70000", "[Console]::Out.Write(('x' * 70000))"),
	})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(result.Content, "[stdout truncated: omitted ~") || !strings.Contains(result.Content, " tokens]") {
		t.Fatalf("content = %q, want stdout truncation marker", result.Content)
	}
	if count, want := strings.Count(result.Content, "x"), shellTestCapturedXCount(); count != want {
		t.Fatalf("captured x count = %d, want %d", count, want)
	}
	if len(result.Content) > maxBashOutputBytes+200 {
		t.Fatalf("content length = %d, want bounded output", len(result.Content))
	}
}

func TestBoundedOutputTruncatesAtUTF8Boundary(t *testing.T) {
	var out boundedOutput
	out.Limit = len([]byte("abc")) + 1

	if _, err := out.Write([]byte("abcédef")); err != nil {
		t.Fatal(err)
	}
	content := out.String("stdout")
	if !utf8.ValidString(content) {
		t.Fatalf("content is not valid UTF-8: %q", content)
	}
	if strings.ContainsRune(content, utf8.RuneError) {
		t.Fatalf("content contains replacement rune: %q", content)
	}
	if !strings.HasPrefix(content, "abc\n\n[stdout truncated:") {
		t.Fatalf("content = %q, want complete ASCII prefix and truncation marker", content)
	}
}

func TestBoundedOutputKeepsCompleteUTF8AtBoundary(t *testing.T) {
	var out boundedOutput
	out.Limit = len([]byte("abcé"))

	if _, err := out.Write([]byte("abcédef")); err != nil {
		t.Fatal(err)
	}
	if content := out.String("stdout"); !strings.HasPrefix(content, "abcé\n\n[stdout truncated:") {
		t.Fatalf("content = %q, want complete UTF-8 prefix", content)
	}
}

func TestBoundedOutputTrimsTrailingPartialUTF8(t *testing.T) {
	var out boundedOutput
	out.Limit = 4

	if _, err := out.Write([]byte{'a', 'b', 'c', 0xc3}); err != nil {
		t.Fatal(err)
	}
	if _, err := out.Write([]byte{0xa9}); err != nil {
		t.Fatal(err)
	}
	if content := out.String("stdout"); !utf8.ValidString(content) || !strings.HasPrefix(content, "abc\n\n[stdout truncated:") {
		t.Fatalf("content = %q, want valid UTF-8 with partial suffix trimmed", content)
	}
}

func TestUTF8SafePrefixRejectsMalformedLeadByte(t *testing.T) {
	input := []byte{'a', 0xc0, 0x80, 'b'}
	if got := utf8SafePrefixLen(input); got != 1 {
		t.Fatalf("safe prefix length = %d, want 1", got)
	}
}

func TestBoundedOutputDropsMalformedUTF8(t *testing.T) {
	var out boundedOutput
	out.Limit = 4

	if _, err := out.Write([]byte{'a', 0xc0, 0x80, 'b'}); err != nil {
		t.Fatal(err)
	}
	content := out.String("stdout")
	if !utf8.ValidString(content) {
		t.Fatalf("content is not valid UTF-8: %q", content)
	}
	if strings.ContainsRune(content, utf8.RuneError) {
		t.Fatalf("content contains replacement rune: %q", content)
	}
	if !strings.HasPrefix(content, "a\n\n[stdout truncated:") {
		t.Fatalf("content = %q, want valid prefix and truncation marker", content)
	}
}

func TestBashReportsCanceledCommand(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	result, err := (&Bash{}).Execute(ctx, agent.ToolContext{WorkingDir: t.TempDir()}, map[string]any{
		"command": shellTestCommand("sleep 10", "Start-Sleep -Seconds 10"),
	})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(result.Content, "Error: command was canceled") {
		t.Fatalf("content = %q, want canceled message", result.Content)
	}
	if strings.Contains(result.Content, "Exit code: -1") {
		t.Fatalf("content = %q, should not mask cancellation as exit code", result.Content)
	}
}

func TestRejectUnsafeShellCommand(t *testing.T) {
	tests := []struct {
		name    string
		command string
		wantErr bool
	}{
		{name: "rm root", command: "rm -rf /", wantErr: true},
		{name: "sudo rm root", command: "sudo rm -rf -- /", wantErr: true},
		{name: "rm home", command: "rm -fr $HOME", wantErr: true},
		{name: "rm root wildcard", command: "rm -rf /*", wantErr: true},
		{name: "rm system subdir", command: "rm -rf /etc/ssh", wantErr: true},
		{name: "rm cwd", command: "rm -rf .", wantErr: true},
		{name: "powershell remove root", command: `Remove-Item -Recurse -Force C:\`, wantErr: true},
		{name: "powershell remove system subdir", command: `Remove-Item -Recurse -Force C:\Windows\Temp`, wantErr: true},
		{name: "ssh private key", command: "cat ~/.ssh/id_rsa", wantErr: true},
		{name: "aws credentials", command: "Get-Content $HOME/.aws/credentials", wantErr: true},
		{name: "shadow", command: "head /etc/shadow", wantErr: true},
		{name: "netrc", command: "cat ~/.netrc", wantErr: true},
		{name: "docker config", command: "cat ~/.docker/config.json", wantErr: true},
		{name: "gnupg dir", command: "cat ~/.gnupg/private-keys-v1.d/key", wantErr: true},
		{name: "gh hosts", command: "cat ~/.config/gh/hosts.yml", wantErr: true},
		{name: "ssh config", command: "cat ~/.ssh/config", wantErr: true},
		{name: "printenv dump", command: "printenv", wantErr: false},
		{name: "delete build dir", command: "rm -rf build", wantErr: false},
		{name: "read project file", command: "cat README.md", wantErr: false},
		{name: "mention key text", command: "rg id_rsa docs", wantErr: false},
		{name: "env example", command: "cat .env.example", wantErr: false},
		{name: "rm build then unrelated tilde path", command: "rm -rf build && echo ~/.ssh/config", wantErr: false},
		{name: "rm build then unrelated slash path", command: "rm -rf build; cat /etc/passwd", wantErr: false},
		{name: "rm build then unrelated star glob", command: "rm -rf build && ls *.go", wantErr: false},
		{name: "rm multiple targets one unsafe", command: "rm -rf build /etc", wantErr: true},
		{name: "rm unsafe then safe piped", command: "rm -rf / | tee log", wantErr: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := rejectUnsafeShellCommand(tt.command)
			if tt.wantErr && err == nil {
				t.Fatal("expected unsafe command to be rejected")
			}
			if !tt.wantErr && err != nil {
				t.Fatalf("command rejected: %v", err)
			}
		})
	}
}

func TestBashRejectsUnsafeCommandBeforeExecution(t *testing.T) {
	_, err := (&Bash{}).Execute(context.Background(), agent.ToolContext{WorkingDir: t.TempDir()}, map[string]any{
		"command": "rm -rf /",
	})
	if err == nil || !strings.Contains(err.Error(), "refusing to run unsafe command") {
		t.Fatalf("err = %v, want unsafe command rejection", err)
	}
}

func shellTestCommand(unix, windows string) string {
	if runtime.GOOS == "windows" {
		return windows
	}
	return unix
}

func shellTestCapturedXCount() int {
	if runtime.GOOS == "windows" {
		return maxBashOutputBytes
	}
	return maxBashOutputBytes / 2
}

func TestReadFinalWorkingDirRejectsInvalidPaths(t *testing.T) {
	dir := t.TempDir()
	cwdFile := filepath.Join(dir, "cwd")
	notDir := filepath.Join(dir, "file.txt")
	if err := os.WriteFile(notDir, []byte("not a dir"), 0o644); err != nil {
		t.Fatal(err)
	}

	if err := os.WriteFile(cwdFile, []byte(notDir+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := readFinalWorkingDir(cwdFile); got != "" {
		t.Fatalf("regular file cwd = %q, want empty", got)
	}

	if err := os.WriteFile(cwdFile, []byte(filepath.Join(dir, "missing")+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := readFinalWorkingDir(cwdFile); got != "" {
		t.Fatalf("missing cwd = %q, want empty", got)
	}

	if err := os.WriteFile(cwdFile, []byte(dir+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := readFinalWorkingDir(cwdFile); got != dir {
		t.Fatalf("directory cwd = %q, want %q", got, dir)
	}
}

func TestNormalizeBashWorkingDirWindowsDriveLetter(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("windows path normalization")
	}
	got := normalizeBashWorkingDir("/c/Users/jdoe/project")
	want := filepath.Clean(`C:\Users\jdoe\project`)
	if got != want {
		t.Fatalf("working dir = %q, want %q", got, want)
	}
}
