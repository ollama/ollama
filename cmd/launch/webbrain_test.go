package launch

import (
	"net/url"
	"testing"
)

func TestWebBrainHandoffURL(t *testing.T) {
	t.Setenv("OLLAMA_HOST", "http://0.0.0.0:11434")

	got := webBrainHandoffURL("qwen3:8b", []LaunchModel{{
		Name:          "qwen3:8b",
		ContextLength: 131072,
	}})
	u, err := url.Parse(got)
	if err != nil {
		t.Fatalf("failed to parse handoff URL: %v", err)
	}

	if u.Scheme != "https" || u.Host != "webbrain.one" || u.Path != "/launch/ollama" {
		t.Fatalf("unexpected handoff URL: %s", got)
	}
	q := u.Query()
	if q.Get("source") != "ollama" {
		t.Fatalf("source = %q, want ollama", q.Get("source"))
	}
	if q.Get("model") != "qwen3:8b" {
		t.Fatalf("model = %q, want qwen3:8b", q.Get("model"))
	}
	if q.Get("baseUrl") != "http://127.0.0.1:11434/v1" {
		t.Fatalf("baseUrl = %q, want connectable /v1 URL", q.Get("baseUrl"))
	}
	if q.Get("contextWindow") != "131072" {
		t.Fatalf("contextWindow = %q, want 131072", q.Get("contextWindow"))
	}
}

func TestWebBrainHandoffURLUsesDefaultContextWindow(t *testing.T) {
	got := webBrainHandoffURL("llama3.2", nil)
	u, err := url.Parse(got)
	if err != nil {
		t.Fatalf("failed to parse handoff URL: %v", err)
	}
	if u.Query().Get("contextWindow") != "65536" {
		t.Fatalf("contextWindow = %q, want 65536", u.Query().Get("contextWindow"))
	}
}

func TestWebBrainRunRejectsExtraArgs(t *testing.T) {
	err := (&WebBrain{}).Run("llama3.2", nil, []string{"--help"})
	if err == nil {
		t.Fatal("expected extra args to be rejected")
	}
}
