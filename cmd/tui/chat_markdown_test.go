package tui

import (
	"strings"
	"testing"
)

func TestChatMarkdownRendersAssistantAndSystemOutput(t *testing.T) {
	m := chatModel{width: 100, height: 30}
	m.entries = []chatEntry{
		{role: "assistant", content: "**Current role:** Software Engineer\n\n- AI agents\n- Structured outputs"},
		{role: "system", content: "**Available tools:**\n\n- bash\n- read"},
	}

	transcript := stripANSI(m.renderTranscript(100))
	if strings.Contains(transcript, "**Current role:**") || strings.Contains(transcript, "**Available tools:**") {
		t.Fatalf("markdown markers were not rendered: %q", transcript)
	}
	if !strings.Contains(transcript, "Current role:") || !strings.Contains(transcript, "Available tools:") {
		t.Fatalf("rendered markdown content missing: %q", transcript)
	}
}

func TestMarkdownRendererCacheIsBounded(t *testing.T) {
	previous := chatMarkdownRenderers
	chatMarkdownRenderers = newMarkdownRendererCache()
	defer func() {
		chatMarkdownRenderers = previous
	}()

	for i := 0; i < maxMarkdownRendererCacheEntries+4; i++ {
		if _, err := markdownRendererForWidth(40 + i); err != nil {
			t.Fatal(err)
		}
	}
	if got := chatMarkdownRenderers.len(); got != maxMarkdownRendererCacheEntries {
		t.Fatalf("cache size = %d, want %d", got, maxMarkdownRendererCacheEntries)
	}
}

func TestChatMarkdownExposesLinks(t *testing.T) {
	m := chatModel{width: 100, height: 30}
	m.entries = []chatEntry{{
		role:    "assistant",
		content: "Open [Ollama](https://ollama.com) for details.",
	}}

	transcript := stripANSI(m.renderTranscript(100))
	if !strings.Contains(transcript, "https://ollama.com") {
		t.Fatalf("rendered markdown should expose URL for terminal clicking: %q", transcript)
	}
}

func TestChatMarkdownRendersTableWithinWidth(t *testing.T) {
	markdown := strings.Join([]string{
		"| Tool | Description |",
		"| --- | --- |",
		"| bash | Execute shell commands, inspect files, run tests, and perform development tasks. |",
		"| web_search | Search the web for current information that may not be in the model training data. |",
	}, "\n")

	rendered := stripANSI(renderMarkdownForView(markdown, 48))
	if strings.Contains(rendered, "| --- |") {
		t.Fatalf("table should render instead of preserving markdown separator: %q", rendered)
	}
	for _, line := range strings.Split(rendered, "\n") {
		if len([]rune(line)) > 48 {
			t.Fatalf("rendered table line width = %d, want <= 48: %q\n%s", len([]rune(line)), line, rendered)
		}
	}
	if !strings.Contains(rendered, "bash") || !strings.Contains(rendered, "web_search") {
		t.Fatalf("rendered table missing content: %q", rendered)
	}
}

func TestChatMarkdownRendersTableWithoutOuterPipes(t *testing.T) {
	markdown := strings.Join([]string{
		"Here is a summary table:",
		"",
		"Category | Details",
		"-------- | -------",
		"**Current Role** | Software Engineer",
		"Company | [Ollama](https://ollama.com/)",
		"Previous Experience | Tesla, Apple, Co-founder of Extensible (LLM monitoring / agent reliability startup)",
		"Personal Interests | Latte art, Muay Thai, writing essays",
	}, "\n")

	rendered := stripANSI(renderMarkdownForView(markdown, 76))
	if strings.Contains(rendered, "-------- | -------") ||
		strings.Contains(rendered, "Category | Details") ||
		strings.Contains(rendered, "Previous Experience |") {
		t.Fatalf("table should not preserve raw markdown table syntax: %q", rendered)
	}
	if !strings.Contains(rendered, "Current Role") ||
		!strings.Contains(rendered, "Software Engineer") ||
		!strings.Contains(rendered, "https://ollama.com/") {
		t.Fatalf("rendered table missing content: %q", rendered)
	}
	for _, line := range strings.Split(rendered, "\n") {
		if len([]rune(line)) > 76 {
			t.Fatalf("rendered table line width = %d, want <= 76: %q\n%s", len([]rune(line)), line, rendered)
		}
	}
}

func TestChatMarkdownDoesNotExposeLinksInsideCodeFences(t *testing.T) {
	markdown := strings.Join([]string{
		"Open [Ollama](https://ollama.com) for details.",
		"",
		"```text",
		"[Code Link](https://code.example)",
		"```",
	}, "\n")

	rendered := stripANSI(renderMarkdownForView(markdown, 80))
	if !strings.Contains(rendered, "Ollama (https://ollama.com)") {
		t.Fatalf("prose link should expose URL: %q", rendered)
	}
	if strings.Contains(rendered, "Code Link (https://code.example)") {
		t.Fatalf("code fence link should remain literal: %q", rendered)
	}
	if !strings.Contains(rendered, "[Code Link](https://code.example)") {
		t.Fatalf("code fence content missing literal markdown link: %q", rendered)
	}
}

func TestChatMarkdownDoesNotRenderTablesInsideCodeFences(t *testing.T) {
	markdown := strings.Join([]string{
		"```text",
		"Name | Value",
		"--- | ---",
		"one | two",
		"```",
	}, "\n")

	rendered := stripANSI(renderMarkdownForView(markdown, 80))
	if !strings.Contains(rendered, "Name | Value") ||
		!strings.Contains(rendered, "--- | ---") ||
		!strings.Contains(rendered, "one | two") {
		t.Fatalf("code fence table syntax should remain literal: %q", rendered)
	}
}

func TestChatMarkdownRendersMixedBlocks(t *testing.T) {
	markdown := strings.Join([]string{
		"Intro [link](https://prose.example).",
		"",
		"Name | Link",
		"--- | ---",
		"Ollama | [site](https://ollama.com)",
		"",
		"```diff",
		"--- a/file.go",
		"+++ b/file.go",
		"@@ -1 +1 @@",
		"-old",
		"+new",
		"```",
		"",
		"```text",
		"[literal](https://literal.example)",
		"```",
	}, "\n")

	rendered := stripANSI(renderMarkdownForView(markdown, 90))
	for _, want := range []string{
		"link (https://prose.example)",
		"Ollama",
		"https://ollama.com",
		"-old",
		"+new",
		"[literal](https://literal.example)",
	} {
		if !strings.Contains(rendered, want) {
			t.Fatalf("rendered mixed markdown missing %q: %q", want, rendered)
		}
	}
	for _, notWant := range []string{"--- | ---", "```diff", "```text", "literal (https://literal.example)"} {
		if strings.Contains(rendered, notWant) {
			t.Fatalf("rendered mixed markdown should not contain %q: %q", notWant, rendered)
		}
	}
}

func TestChatMarkdownRendersFencedDiff(t *testing.T) {
	m := chatModel{width: 100, height: 30}
	m.entries = []chatEntry{{
		role: "assistant",
		content: strings.Join([]string{
			"Patch:",
			"",
			"```diff",
			"--- a/file.go",
			"+++ b/file.go",
			"@@ -1 +1 @@",
			"-old",
			"+new",
			"```",
		}, "\n"),
	}}

	rendered := m.renderTranscript(100)
	transcript := stripANSI(rendered)
	if strings.Contains(transcript, "```diff") || strings.Contains(transcript, "```") {
		t.Fatalf("diff fence markers should not render: %q", transcript)
	}
	if !strings.Contains(transcript, "Patch:") ||
		!strings.Contains(transcript, "-old") ||
		!strings.Contains(transcript, "+new") {
		t.Fatalf("rendered fenced diff missing content: %q", transcript)
	}
	if _, ok := renderMarkdownDiffFences(m.entries[0].content, 100); !ok {
		t.Fatal("markdown diff fence should use diff-aware rendering")
	}
}

func TestChatReadToolOutputRendersMarkdown(t *testing.T) {
	output := strings.Join([]string{
		"Name | Value",
		"--- | ---",
		"**Status** | [ok](https://example.com)",
	}, "\n")

	rendered := stripANSI(strings.Join(renderToolOutputLines(chatEntry{detail: "read"}, output, 80), "\n"))
	if strings.Contains(rendered, "--- | ---") || strings.Contains(rendered, "Name | Value") {
		t.Fatalf("read tool markdown table should render: %q", rendered)
	}
	if !strings.Contains(rendered, "Status") || !strings.Contains(rendered, "https://example.com") {
		t.Fatalf("rendered read markdown missing content: %q", rendered)
	}
}
