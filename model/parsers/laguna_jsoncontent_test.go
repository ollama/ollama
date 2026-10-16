package parsers

import (
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

// streamLaguna feeds s to the parser one small chunk at a time, the way tokens
// arrive during generation, and reports the content emitted, the tool calls
// produced, and any error that ended the stream.
func streamLaguna(t *testing.T, tools []api.Tool, s string) (content string, calls []api.ToolCall, err error) {
	t.Helper()

	p := &LagunaV8Parser{}
	p.Init(tools, nil, nil)

	const chunk = 7
	var sb strings.Builder
	for i := 0; i < len(s); i += chunk {
		end := min(i+chunk, len(s))
		c, _, cl, e := p.Add(s[i:end], end == len(s))
		sb.WriteString(c)
		calls = append(calls, cl...)
		if e != nil {
			return sb.String(), calls, e
		}
	}
	return sb.String(), calls, nil
}

// TestLagunaStandaloneJSONInContent covers what the parser does with a reply
// that contains a brace, with at least one tool declared. Source code holds a
// brace but never forms valid JSON, so it must survive as content. A JSON
// document does form valid JSON, and the parser currently treats it as a tool
// call: without a "name" it fails the whole response, and with a "name" it
// converts the document into a call and drops the text.
func TestLagunaStandaloneJSONInContent(t *testing.T) {
	goCode := "Here is the program:\n\n```go\nfunc main() {\n\tfmt.Println(\"hi\")\n}\n```\n"

	mcpConfig := "Add this to your config:\n\n```json\n{\n  \"mcpServers\": {\n    \"github\": {\n      \"command\": \"npx\",\n      \"env\": {\"GITHUB_TOKEN\": \"your_token\"}\n    }\n  }\n}\n```\n"

	namedConfig := "Save this file:\n\n```json\n{\n  \"name\": \"my-project\",\n  \"version\": \"1.0.0\"\n}\n```\n"

	for _, tc := range []struct {
		name string
		text string
	}{
		{"go-source-with-braces", goCode},
		{"json-config-without-name", mcpConfig},
		{"json-config-with-name", namedConfig},
		{"json-object-inline-in-prose", "Logging defaults to {\"level\": \"debug\", \"color\": true} in this app.\n"},
		{"json-array-of-objects", "The endpoint returns [{\"id\": 1, \"ok\": true}] by default.\n"},
		{"json-object-named-like-a-person", "The record is {\"name\": \"John Smith\", \"age\": 30} exactly.\n"},
		{"empty-json-object", "An empty config is {} in this format.\n"},
		{"truncated-json-at-end-of-stream", "The config begins with {\"name\": \"web_s"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			noToolsContent, noToolsCalls, noToolsErr := streamLaguna(t, nil, tc.text)
			toolsContent, toolsCalls, toolsErr := streamLaguna(t, lagunaTestTools(), tc.text)

			t.Logf("no tools declared: err=%v calls=%d content=%d/%d bytes",
				noToolsErr, len(noToolsCalls), len(noToolsContent), len(tc.text))
			t.Logf("one tool declared: err=%v calls=%d content=%d/%d bytes",
				toolsErr, len(toolsCalls), len(toolsContent), len(tc.text))
			for i, c := range toolsCalls {
				t.Logf("  call[%d]: name=%q args=%v", i, c.Function.Name, c.Function.Arguments)
			}
			if toolsContent != tc.text {
				t.Logf("  content delivered: %q", toolsContent)
			}

			// The parser trims trailing whitespace from the final chunk, which
			// is by design, so compare without it.
			want := strings.TrimRight(tc.text, "\n")

			// With no tools declared the reply must always come back intact.
			if noToolsErr != nil {
				t.Fatalf("no tools declared: unexpected error: %v", noToolsErr)
			}
			if len(noToolsCalls) != 0 {
				t.Fatalf("no tools declared: got %d tool calls, want 0", len(noToolsCalls))
			}
			if noToolsContent != want {
				t.Fatalf("no tools declared: content altered\n got: %q\nwant: %q", noToolsContent, tc.text)
			}

			// Declaring an unrelated tool must not change how prose and code
			// in the reply are handled.
			if toolsErr != nil {
				t.Fatalf("one tool declared: response failed with %v", toolsErr)
			}
			if len(toolsCalls) != 0 {
				t.Fatalf("one tool declared: reply turned into %d tool calls, want 0 (%+v)", len(toolsCalls), toolsCalls)
			}
			if toolsContent != want {
				t.Fatalf("one tool declared: content altered\n got: %q\nwant: %q", toolsContent, want)
			}
		})
	}
}

// TestLagunaJSONInContentStreamsPromptly checks that a JSON object in content
// does not stall the stream. The parser must decide against the object soon
// after its first key arrives, rather than holding every later token until the
// generation ends.
func TestLagunaJSONInContentStreamsPromptly(t *testing.T) {
	p := &LagunaV8Parser{}
	p.Init(lagunaTestTools(), nil, nil)

	chunks := []string{
		"Logging defaults to", ` {"`, "level", `": "`, "debug", `"}`,
		" and", " the", " app", " must", " restart", ".",
	}

	var got string
	for i, tok := range chunks {
		c, _, calls, err := p.Add(tok, false)
		if err != nil {
			t.Fatalf("chunk %d (%q): unexpected error: %v", i+1, tok, err)
		}
		if len(calls) > 0 {
			t.Fatalf("chunk %d (%q): unexpected tool call %+v", i+1, tok, calls)
		}
		got += c
		// Once the object has closed and two more words have arrived, the
		// object must already have been delivered as content.
		if i == 7 && !strings.Contains(got, `{"level": "debug"}`) {
			t.Fatalf("content withheld after the object closed: %q", got)
		}
	}

	want := `Logging defaults to {"level": "debug"} and the app must restart.`
	if got != want {
		t.Errorf("content altered\n got: %q\nwant: %q", got, want)
	}
}

// TestLagunaStandaloneJSONToolCallVariants covers bare JSON tool calls the
// parser must still recognize, including text after the object, which earlier
// code discarded.
func TestLagunaStandaloneJSONToolCallVariants(t *testing.T) {
	for _, tc := range []struct {
		name        string
		text        string
		wantName    string
		wantContent string
	}{
		{
			name:     "plain",
			text:     `{"name":"get_weather","arguments":{"location":"Paris"}}`,
			wantName: "get_weather",
		},
		{
			name:     "arguments first",
			text:     `{"arguments":{"location":"Paris"},"name":"get_weather"}`,
			wantName: "get_weather",
		},
		{
			name:        "text after the call",
			text:        `{"name":"get_weather","arguments":{"location":"Paris"}} checking now`,
			wantName:    "get_weather",
			wantContent: "checking now",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			content, calls, err := streamLaguna(t, lagunaTestTools(), tc.text)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(calls) != 1 {
				t.Fatalf("got %d tool calls, want 1: %+v", len(calls), calls)
			}
			if calls[0].Function.Name != tc.wantName {
				t.Errorf("call name = %q, want %q", calls[0].Function.Name, tc.wantName)
			}
			if content != tc.wantContent {
				t.Errorf("content = %q, want %q", content, tc.wantContent)
			}
		})
	}
}
