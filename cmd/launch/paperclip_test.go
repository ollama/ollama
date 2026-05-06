package launch

import (
	"reflect"
	"testing"
)

func TestPaperclipString(t *testing.T) {
	p := &Paperclip{}
	if got := p.String(); got != "Paperclip" {
		t.Fatalf("String() = %q, want %q", got, "Paperclip")
	}
}

func TestPaperclipArgs(t *testing.T) {
	p := &Paperclip{}
	got := p.args("qwen3:14b", []string{"--extra"})
	want := []string{"onboard", "--bind", "loopback", "-y", "--run", "--extra"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("args = %v, want %v", got, want)
	}
}

func TestPaperclipArgsNoExtra(t *testing.T) {
	p := &Paperclip{}
	got := p.args("", nil)
	want := []string{"onboard", "--bind", "loopback", "-y", "--run"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("args = %v, want %v", got, want)
	}
}

func TestOllamaHostFromEnvDefault(t *testing.T) {
	t.Setenv("OLLAMA_HOST", "")
	if got := ollamaHostFromEnv(); got != "http://localhost:11434" {
		t.Fatalf("default = %q, want %q", got, "http://localhost:11434")
	}
}

func TestOllamaHostFromEnvOverride(t *testing.T) {
	t.Setenv("OLLAMA_HOST", "http://my-host:9999")
	if got := ollamaHostFromEnv(); got != "http://my-host:9999" {
		t.Fatalf("override = %q, want %q", got, "http://my-host:9999")
	}
}
