package server

import (
	"bytes"
	"context"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/template"
)

// Regression test for ollama/ollama#19942: truncating a long chat history
// used to re-render and re-tokenize the whole remaining history once per
// dropped message, so the work grew quadratically with message count
// (N=1024: 961 tokenize calls, ~523k tokens). The truncation search must
// find the same cut point with a logarithmic number of evaluations.

// simpleChatTemplate renders each message's content followed by a space,
// so the token count of msgs[i:] is exactly len(msgs)-i for one-word
// messages.
func simpleChatTemplate(t *testing.T) *template.Template {
	t.Helper()
	tmpl, err := template.Parse(`{{range .Messages}}{{.Content}} {{end}}`)
	if err != nil {
		t.Fatal(err)
	}
	return tmpl
}

func TestChatPromptTruncationIsSubquadratic(t *testing.T) {
	tmpl := simpleChatTemplate(t)
	m := &Model{Template: tmpl}

	for _, n := range []int{256, 512, 1024} {
		msgs := make([]api.Message, n)
		for i := range msgs {
			msgs[i] = api.Message{Role: "user", Content: "hi"}
		}

		calls, tokensSeen := 0, 0
		tokenize := func(_ context.Context, prompt string) ([]int, error) {
			calls++
			k := len(strings.Fields(prompt))
			tokensSeen += k
			return make([]int, k), nil
		}

		opts := api.Options{Runner: api.Runner{NumCtx: 64}}
		prompt, _, err := chatPrompt(t.Context(), m, tokenize, &opts, msgs, nil, nil, true)
		if err != nil {
			t.Fatal(err)
		}

		// Same cut as before: keep the last 64 messages. The template
		// collates consecutive same-role messages with "\n\n", so the
		// rendered prompt is the 64 contents joined that way.
		contents := make([]string, 64)
		for i := range contents {
			contents[i] = "hi"
		}
		if want := strings.Join(contents, "\n\n") + " "; prompt != want {
			t.Fatalf("N=%d: prompt keeps %d words, want 64", n, len(strings.Fields(prompt)))
		}
		// Logarithmic evaluations: 1 fast-path check + binary search.
		if calls > 12 {
			t.Fatalf("N=%d: tokenize called %d times, want at most 12", n, calls)
		}
		// Total tokenized input must grow linearly, not quadratically.
		if tokensSeen > 4*n {
			t.Fatalf("N=%d: tokenized %d tokens in total, want at most %d", n, tokensSeen, 4*n)
		}
		t.Logf("N=%d calls=%d tokens_tokenized=%d", n, calls, tokensSeen)
	}
}

func TestChatPromptTruncationKeepsSystemMessages(t *testing.T) {
	tmpl := simpleChatTemplate(t)
	m := &Model{Template: tmpl}

	msgs := make([]api.Message, 0, 600)
	msgs = append(msgs, api.Message{Role: "system", Content: "sys"})
	for i := 1; i < 600; i++ {
		role := "user"
		if i%2 == 0 {
			role = "assistant"
		}
		content := "word"
		if i == 500 {
			role = "system"
			content = "sys2"
		}
		msgs = append(msgs, api.Message{Role: role, Content: content})
	}

	// Reference cut from an independent linear scan over rendered token
	// counts: the smallest i whose rendered prompt fits in 64 tokens.
	render := func(list []api.Message) string {
		var b bytes.Buffer
		if err := tmpl.Execute(&b, template.Values{Messages: list}); err != nil {
			t.Fatal(err)
		}
		return b.String()
	}
	last := len(msgs) - 1
	wantCut := last
	for i := 0; i <= last; i++ {
		var sys []api.Message
		for j := range i {
			if msgs[j].Role == "system" {
				sys = append(sys, msgs[j])
			}
		}
		if len(strings.Fields(render(append(sys, msgs[i:]...)))) <= 64 {
			wantCut = i
			break
		}
	}

	calls := 0
	tokenize := func(_ context.Context, prompt string) ([]int, error) {
		calls++
		return make([]int, len(strings.Fields(prompt))), nil
	}

	opts := api.Options{Runner: api.Runner{NumCtx: 64}}
	prompt, _, err := chatPrompt(t.Context(), m, tokenize, &opts, msgs, nil, nil, true)
	if err != nil {
		t.Fatal(err)
	}

	var wantSys []api.Message
	for j := range wantCut {
		if msgs[j].Role == "system" {
			wantSys = append(wantSys, msgs[j])
		}
	}
	if want := render(append(wantSys, msgs[wantCut:]...)); prompt != want {
		t.Fatalf("cut differs from linear scan: got %q, want %q", prompt, want)
	}
	if calls > 12 {
		t.Fatalf("tokenize called %d times, want at most 12", calls)
	}
	t.Logf("cut=%d calls=%d", wantCut, calls)
}

type nativeCountingRunner struct {
	mockRunner
	calls, bytesSeen int
}

func (r *nativeCountingRunner) Tokenize(_ context.Context, prompt string) ([]int, error) {
	r.calls++
	r.bytesSeen += len(prompt)
	return make([]int, len(strings.Fields(prompt))), nil
}

func TestTruncateNativeChatMessagesIsSubquadratic(t *testing.T) {
	for _, n := range []int{256, 512, 1024} {
		msgs := make([]api.Message, n)
		for i := range msgs {
			msgs[i] = api.Message{Role: "user", Content: "hi"}
		}

		r := &nativeCountingRunner{}
		r.TemplateFn = func(_ context.Context, req llm.ChatRequest) (string, error) {
			var b strings.Builder
			for _, m := range req.Messages {
				b.WriteString(m.Content)
				b.WriteByte(' ')
			}
			return b.String(), nil
		}

		opts := &api.Options{Runner: api.Runner{NumCtx: 64}}
		got, err := truncateNativeChatMessages(t.Context(), nil, r, opts, llm.ChatRequest{Messages: msgs}, true)
		if err != nil {
			t.Fatal(err)
		}

		if len(got) != 64 {
			t.Fatalf("N=%d: kept %d messages, want 64", n, len(got))
		}
		if r.calls > 12 {
			t.Fatalf("N=%d: tokenize called %d times, want at most 12", n, r.calls)
		}
		if r.bytesSeen > 12*n {
			t.Fatalf("N=%d: tokenized %d bytes in total, want at most %d", n, r.bytesSeen, 12*n)
		}
		t.Logf("N=%d calls=%d bytes_tokenized=%d kept=%d", n, r.calls, r.bytesSeen, len(got))
	}
}

func TestTruncateNativeChatMessagesKeepsSystemMessages(t *testing.T) {
	msgs := []api.Message{{Role: "system", Content: "s"}}
	for i := 0; i < 8; i++ {
		msgs = append(msgs, api.Message{Role: "user", Content: "hi"})
	}

	r := &nativeCountingRunner{}
	r.TemplateFn = func(_ context.Context, req llm.ChatRequest) (string, error) {
		var b strings.Builder
		for _, m := range req.Messages {
			b.WriteString(m.Content)
			b.WriteByte(' ')
		}
		return b.String(), nil
	}

	// 1 system token + 2 user tokens fit in 3.
	opts := &api.Options{Runner: api.Runner{NumCtx: 3}}
	got, err := truncateNativeChatMessages(t.Context(), nil, r, opts, llm.ChatRequest{Messages: msgs}, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 3 || got[0].Role != "system" || got[0].Content != "s" {
		t.Fatalf("got %+v, want [system s, user hi, user hi]", got)
	}
	if r.calls > 6 {
		t.Fatalf("tokenize called %d times, want at most 6", r.calls)
	}
}
