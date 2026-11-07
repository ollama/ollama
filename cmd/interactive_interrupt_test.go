package cmd

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"regexp"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"
	"unicode/utf8"
)

// The REPL is driven through a pty: the bug is about what the prompt does with
// the line typed *after* the terminal reports an interrupt, which a pipe on
// stdin cannot reproduce.
const replHelperEnv = "OLLAMA_TEST_REPL_HELPER"

func TestMain(m *testing.M) {
	if os.Getenv(replHelperEnv) == "1" {
		runReplHelper()
		return
	}
	os.Exit(m.Run())
}

// runReplHelper is `ollama run` in the child process: the REPL needs a real
// terminal on stdin, so it gets its own process and the whole command tree.
func runReplHelper() {
	root := NewCLI()
	root.SetArgs([]string{"run", "llama3.2:1b"})
	root.SetOut(os.Stdout)
	root.SetErr(os.Stdout)
	if err := root.Execute(); err != nil {
		os.Stderr.WriteString("error: " + err.Error() + "\n")
		os.Exit(1)
	}
}

type replMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type replChat struct {
	Messages []replMessage `json:"messages"`
}

type replServer struct {
	t     *testing.T
	url   string
	mu    sync.Mutex
	chats []replChat
}

// userMessages returns the user text the server was asked to answer, from
// either wire: the REPL picks chat or generate depending on the model.
func (s *replServer) userMessages() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	var out []string
	for _, c := range s.chats {
		for _, m := range c.Messages {
			if m.Role == "user" {
				out = append(out, m.Content)
			}
		}
	}
	return out
}

// waitForUser waits until the server was asked to answer content.
func (s *replServer) waitForUser(content string) {
	s.t.Helper()
	deadline := time.Now().Add(15 * time.Second)
	for time.Now().Before(deadline) {
		for _, m := range s.userMessages() {
			if m == content {
				return
			}
		}
		time.Sleep(20 * time.Millisecond)
	}
	s.t.Fatalf("the server was never asked to answer %q, got %q", content, s.userMessages())
}

func newReplServer(t *testing.T) *replServer {
	t.Helper()
	s := &replServer{t: t}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/", "/api/tags":
			if r.URL.Path == "/" && r.Method == http.MethodHead {
				w.WriteHeader(http.StatusOK)
				return
			}
			json.NewEncoder(w).Encode(map[string]any{"models": []any{}})
		case "/api/version":
			json.NewEncoder(w).Encode(map[string]any{"version": "0.0.0"})
		case "/api/ps":
			json.NewEncoder(w).Encode(map[string]any{"models": []any{}})
		case "/api/show":
			json.NewEncoder(w).Encode(map[string]any{
				"capabilities": []string{"completion"},
				"details":      map[string]any{"format": "gguf", "family": "llama"},
			})
		case "/api/generate":
			// `ollama run` loads the model with an empty prompt first.
			json.NewEncoder(w).Encode(map[string]any{"response": "ok", "done": true})
		case "/api/chat":
			var req replChat
			json.NewDecoder(r.Body).Decode(&req)
			s.mu.Lock()
			s.chats = append(s.chats, req)
			s.mu.Unlock()
			json.NewEncoder(w).Encode(map[string]any{
				"model":   "llama3.2:1b",
				"message": map[string]any{"role": "assistant", "content": "ok"},
				"done":    true,
			})
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(srv.Close)
	s.url = srv.URL
	return s
}

// replSession is the `ollama run` child process attached to a pty.
type replSession struct {
	t      *testing.T
	pty    *os.File
	output *syncBuffer
}

// syncBuffer is the terminal output, appended from the reader goroutine and
// read by the test.
type syncBuffer struct {
	mu  sync.Mutex
	buf strings.Builder
}

func (b *syncBuffer) WriteString(s string) {
	b.mu.Lock()
	b.buf.WriteString(s)
	b.mu.Unlock()
}

func (b *syncBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.String()
}

// Len is a position in the visible text: escapes dropped, counted in runes, so
// a mark taken here lines up with Since.
func (b *syncBuffer) Len() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return utf8.RuneCountInString(ansiRE.ReplaceAllString(b.buf.String(), ""))
}

func (b *syncBuffer) Since(mark int) string {
	b.mu.Lock()
	defer b.mu.Unlock()
	s := []rune(ansiRE.ReplaceAllString(b.buf.String(), ""))
	if mark > len(s) {
		return ""
	}
	return string(s[mark:])
}

func startRepl(t *testing.T, host string) *replSession {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("needs a unix pty")
	}

	pty, tty, err := openPTY()
	if err != nil {
		t.Skipf("no pty: %v", err)
	}

	cmd := exec.Command(os.Args[0], "-test.run=^$")
	cmd.Env = append(os.Environ(),
		replHelperEnv+"=1",
		"OLLAMA_HOST="+host,
		"TERM=xterm-256color",
		"NO_COLOR=1",
		"OLLAMA_NOHISTORY=1",
	)
	cmd.Stdin, cmd.Stdout, cmd.Stderr = tty, tty, tty
	if err := cmd.Start(); err != nil {
		pty.Close()
		tty.Close()
		t.Fatalf("start repl: %v", err)
	}
	tty.Close()

	s := &replSession{t: t, pty: pty, output: &syncBuffer{}}
	go s.pump()
	t.Cleanup(func() {
		pty.Close()
		cmd.Process.Kill()
		cmd.Wait()
	})

	s.waitFor(0, ">>> ")
	return s
}

// pump forwards the terminal output and answers the cursor queries the prompt
// blocks on.
func (s *replSession) pump() {
	buf := make([]byte, 4096)
	for {
		n, err := s.pty.Read(buf)
		if n > 0 {
			chunk := string(buf[:n])
			s.output.WriteString(chunk)
			if strings.Contains(chunk, "\x1b]11;?") {
				s.pty.WriteString("\x1b]11;rgb:0000/0000/0000\x1b\\")
			}
			if strings.Contains(chunk, "\x1b[6n") {
				s.pty.WriteString("\x1b[1;1R")
			}
		}
		if err != nil {
			return
		}
	}
}

// waitFor blocks until the terminal shows snippet after mark, which is a
// position in the escape-free text.
func (s *replSession) waitFor(mark int, snippet string) string {
	s.t.Helper()
	deadline := time.Now().Add(15 * time.Second)
	for time.Now().Before(deadline) {
		if got := s.output.Since(mark); strings.Contains(got, snippet) {
			return got
		}
		time.Sleep(20 * time.Millisecond)
	}
	s.t.Fatalf("timed out waiting for %q, terminal said %q", snippet, tail(s.output.String(), 600))
	return ""
}

// typeLine types the line one rune at a time; the prompt reads raw input.
func (s *replSession) typeLine(line string) {
	s.t.Helper()
	for _, r := range line {
		s.pty.WriteString(string(r))
		time.Sleep(2 * time.Millisecond)
	}
	s.pty.WriteString("\r")
	time.Sleep(80 * time.Millisecond)
}

// ctrlC sends the interrupt and returns once the prompt has redrawn, i.e. once
// the REPL is back at the top of its loop.
func (s *replSession) ctrlC() int {
	s.t.Helper()
	mark := s.output.Len()
	s.pty.WriteString("\x03")
	s.waitFor(mark, "Use Ctrl + d or /bye to exit.")
	s.waitFor(mark, ">>> ")
	return mark
}

// ansiRE matches CSI (ESC [ ... final), OSC (ESC ] ... BEL|ST) and two-byte
// escapes. It is hand-written so the test does not pull in a dependency for
// three lines of terminal noise.
var ansiRE = regexp.MustCompile(`\x1b(?:\[[0-9;?]*[:<=>][0-9;]*[A-Za-z]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[^[\]])`)

func tail(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[len(s)-n:]
}

// Ctrl+C while a """ message is open has to drop the draft entirely. Before
// the fix only the buffer was reset, so the next line was appended to the
// abandoned message and never sent.
func TestInteractiveInterruptAbandonsMultilinePrompt(t *testing.T) {
	server := newReplServer(t)
	repl := startRepl(t, server.url)

	repl.typeLine(`"""draft I want to abandon`)
	repl.waitFor(0, "... ")
	repl.ctrlC()

	repl.typeLine("hello")
	server.waitForUser("hello")

	msgs := server.userMessages()
	if len(msgs) != 1 || msgs[0] != "hello" {
		t.Fatalf("user messages = %q, want just %q", msgs, "hello")
	}
}

// Same for /set system: a line ending in """ typed after Ctrl+C used to become
// the system message.
func TestInteractiveInterruptAbandonsMultilineSystem(t *testing.T) {
	server := newReplServer(t)
	repl := startRepl(t, server.url)

	repl.typeLine(`/set system """You are a pirate`)
	repl.waitFor(0, "... ")
	mark := repl.ctrlC()

	repl.typeLine(`What is 2+2?"""`)
	server.waitForUser(`What is 2+2?"""`)

	got := repl.output.Since(mark)
	if strings.Contains(got, "Set system message.") {
		t.Fatalf("the interrupted system input was applied anyway: %q", tail(got, 400))
	}
	msgs := server.userMessages()
	if len(msgs) != 1 || msgs[0] != `What is 2+2?"""` {
		t.Fatalf("user messages = %q, want the line sent as typed", msgs)
	}
}
