package cmd

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/decision"
	"github.com/ollama/ollama/types/model"
)

func executeAsk(t *testing.T, args []string, input io.Reader) (string, string, error) {
	t.Helper()
	var out, diagnostics bytes.Buffer
	cmd := NewCLI()
	cmd.SetArgs(append([]string{"ask"}, args...))
	cmd.SetIn(input)
	cmd.SetOut(&out)
	cmd.SetErr(&diagnostics)
	err := cmd.ExecuteContext(t.Context())
	return out.String(), diagnostics.String(), err
}

func TestAskCommand(t *testing.T) {
	for _, tt := range []struct {
		name, text, model, typ, criteria, response, output string
		args                                               []string
		input                                              io.Reader
	}{
		{
			name: "positional text ignores stdin",
			args: []string{"nimble", "Is this a refund request?", "I was charged twice."},
			text: "I was charged twice.", model: "nimble", typ: "noul",
			response: `{"type":"noul","noul":0.9989}`, output: "0.9989\n",
		},
		{
			name:  "stdin preserves quotes multiline and whitespace",
			args:  []string{"tev1", "Is this a refund request?"},
			input: strings.NewReader("  I said \"refund\".\nPlease help.\n"), text: "  I said \"refund\".\nPlease help.\n",
			model: "tev1", typ: "noul", response: `{"type":"noul","noul":0}`, output: "0\n",
		},
		{
			name: "standalone question uses its own text as state",
			args: []string{"nimble", "Is water made of hydrogen and oxygen?"}, input: strings.NewReader(""),
			text: "Is water made of hydrogen and oxygen?", model: "nimble", typ: "noul",
			response: `{"type":"noul","noul":0.99}`, output: "0.99\n",
		},
		{
			name: "choices preserve order descriptions commas and equals",
			args: []string{"nimble", "Which label fits?", "Checkout is down.", "--choice", "bug=Software errors, including 500=errors", "--choice", "billing"},
			text: "Checkout is down.", model: "nimble", typ: "choice",
			criteria: `{"bug":"Software errors, including 500=errors","billing":"billing"}`,
			response: `{"type":"choice","choice":"bug","probabilities":{"bug":0.98,"billing":0.02},"confidence":0.85}`, output: "bug\n",
		},
		{
			name: "score preserves rubric order",
			args: []string{"nimble", "How urgent is this?", "Checkout is down.", "--score", "Routine", "--score", "Soon", "--score", "Immediate"},
			text: "Checkout is down.", model: "nimble", typ: "score", criteria: `["Routine","Soon","Immediate"]`,
			response: `{"type":"score","score":1.8308,"legend":{"0":"Routine","1":"Soon","2":"Immediate"},"probabilities":{"0":0.04,"1":0.0892,"2":0.8708},"confidence":0.71}`, output: "1.8308\n",
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch r.URL.Path {
				case "/":
				case "/api/show":
					json.NewEncoder(w).Encode(api.ShowResponse{Capabilities: []model.Capability{model.CapabilityDecision}})
				case "/v1/systemone":
					if r.Method != http.MethodPost || r.Header.Get("Content-Type") != "application/json" {
						t.Errorf("unexpected request: %s %s", r.Method, r.Header.Get("Content-Type"))
					}
					var req api.SystemOneRequest
					if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
						t.Error(err)
					}
					if _, err := decision.Compile(req); err != nil {
						t.Errorf("server rejected the wire request: %v", err)
					}
					var text string
					json.Unmarshal(req.State, &text)
					question, ok := req.Questions.Get("answer")
					if req.Model != tt.model || text != tt.text || !ok || req.Questions.Len() != 1 || question.Type != tt.typ || string(question.Criteria) != tt.criteria || req.KeepAlive != nil {
						t.Errorf("unexpected request: %+v, text=%q, question=%+v", req, text, question)
					}
					instructions, _ := json.Marshal(tt.args[1])
					if !bytes.Equal(question.Instructions, instructions) {
						t.Errorf("instructions=%s, want %s", question.Instructions, instructions)
					}
					io.WriteString(w, `{"model":"`+req.Model+`","answers":{"answer":`+tt.response+`},"usage":{"input_tokens":174,"output_tokens":1}}`)
				default:
					t.Errorf("unexpected endpoint: %s", r.URL.Path)
					http.NotFound(w, r)
				}
			}))
			defer server.Close()
			t.Setenv("OLLAMA_HOST", server.URL)
			t.Setenv("OLLAMA_AUTH", "0")
			var input io.Reader = unreadAskInput{}
			if tt.input != nil {
				input = tt.input
			}
			out, diagnostics, err := executeAsk(t, tt.args, input)
			if err != nil || out != tt.output || diagnostics != "" {
				t.Fatalf("stdout=%q, stderr=%q, err=%v; want %q", out, diagnostics, err, tt.output)
			}
		})
	}
}

type unreadAskInput struct{}

func (unreadAskInput) Read([]byte) (int, error) {
	return 0, errors.New("stdin should not be read")
}

func TestAskQuestionsJSON(t *testing.T) {
	questions := `{"refund":{"type":"noul","instructions":"Is a refund requested?","criteria":{"false":"No refund","true":"Refund requested"}},"urgency":{"type":"score","instructions":{"task":"How urgent?"},"criteria":["Routine","Soon","Immediate"]}}`
	filename := filepath.Join(t.TempDir(), "questions.json")
	if err := os.WriteFile(filename, []byte(questions), 0o600); err != nil {
		t.Fatal(err)
	}
	response := `{"model":"nimble","answers":{"refund":{"type":"noul","noul":0.9989},"urgency":{"type":"score","score":0.8308,"legend":{"0":"Routine","1":"Soon","2":"Immediate"},"probabilities":{"0":0.2,"1":0.7692,"2":0.0308},"confidence":0.6}},"usage":{"input_tokens":200,"output_tokens":2}}`
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/":
		case "/api/show":
			json.NewEncoder(w).Encode(api.ShowResponse{Capabilities: []model.Capability{model.CapabilityDecision}})
		case "/v1/systemone":
			var req api.SystemOneRequest
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				t.Error(err)
			}
			data, _ := json.Marshal(req.Questions)
			if string(data) != questions {
				t.Errorf("questions=%s, want %s", data, questions)
			}
			io.WriteString(w, response)
		default:
			t.Errorf("unexpected endpoint: %s", r.URL.Path)
		}
	}))
	defer server.Close()
	t.Setenv("OLLAMA_HOST", server.URL)
	t.Setenv("OLLAMA_AUTH", "0")
	for _, extra := range [][]string{nil, {"--json"}} {
		out, diagnostics, err := executeAsk(t, append([]string{"nimble", "--questions", filename, "I was charged twice."}, extra...), unreadAskInput{})
		if err != nil || out != response+"\n" || diagnostics != "" {
			t.Fatalf("stdout=%q, stderr=%q, err=%v", out, diagnostics, err)
		}
	}
}

func TestAskInvalidInput(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Errorf("invalid input contacted server: %s", r.URL.Path)
	}))
	defer server.Close()
	t.Setenv("OLLAMA_HOST", server.URL)
	badFile := filepath.Join(t.TempDir(), "bad.json")
	if err := os.WriteFile(badFile, []byte(`{"answer":{"type":"noul","instructions":"Is this urgent?","criterai":{}}}`), 0o600); err != nil {
		t.Fatal(err)
	}
	validFile := filepath.Join(t.TempDir(), "questions.json")
	if err := os.WriteFile(validFile, []byte(`{"answer":{"type":"noul","instructions":"Is this urgent?"}}`), 0o600); err != nil {
		t.Fatal(err)
	}
	for _, tt := range []struct {
		name, input, error string
		args               []string
	}{
		{name: "missing model", error: "use: ollama ask MODEL", args: nil},
		{name: "missing question", error: "use: ollama ask MODEL", args: []string{"nimble"}},
		{name: "unquoted text", error: "quote", args: []string{"nimble", "Is this urgent?", "Checkout", "is", "down"}},
		{name: "empty input", error: "text must not be empty", args: []string{"nimble", "Is this urgent?"}, input: " \n"},
		{name: "explicit empty text", error: "text must not be empty", args: []string{"nimble", "Is this urgent?", ""}},
		{name: "empty question", error: "instructions", args: []string{"nimble", " ", "text"}},
		{name: "empty model", error: "model is required", args: []string{"", "Question?", "text"}},
		{name: "one choice", error: "2–26", args: []string{"nimble", "Question?", "text", "--choice", "bug"}},
		{name: "duplicate label", error: "duplicate choice", args: []string{"nimble", "Question?", "text", "--choice", "bug=Error", "--choice", "bug"}},
		{name: "one score level", error: "2–26", args: []string{"nimble", "Question?", "text", "--score", "Routine"}},
		{name: "conflicting rubrics", error: "none of the others", args: []string{"nimble", "Question?", "text", "--choice", "bug", "--score", "Routine"}},
		{name: "file flag typo", error: "unknown field", args: []string{"nimble", "--questions", badFile, "text"}},
		{name: "missing file", error: "read questions", args: []string{"nimble", "--questions", badFile + ".missing", "text"}},
		{name: "questions file requires text", error: "text must not be empty", args: []string{"nimble", "--questions", validFile}},
		{name: "large stdin", error: "64 KiB", args: []string{"nimble", "Question?"}, input: strings.Repeat("x", askMaxBytes+1)},
		{name: "JSON escaping exceeds limit", error: "64 KiB", args: []string{"nimble", "Question?", strings.Repeat("a\n", askMaxBytes/2-100)}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			out, diagnostics, err := executeAsk(t, tt.args, strings.NewReader(tt.input))
			if err == nil || !strings.Contains(err.Error(), tt.error) || out != "" || diagnostics != "" {
				t.Fatalf("stdout=%q, stderr=%q, err=%v; want %q", out, diagnostics, err, tt.error)
			}
		})
	}
}

func TestAskServerFailures(t *testing.T) {
	for _, tt := range []struct {
		name, response, error string
		showStatus, status    int
		capability            model.Capability
	}{
		{name: "missing model has pull hint", showStatus: 404, error: "ollama pull nimble"},
		{name: "wrong model capability", capability: model.CapabilityCompletion, error: "does not support decisions"},
		{name: "server error", status: 500, response: `{"error":"runner failed"}`, error: "runner failed"},
		{name: "invalid API request", status: 400, response: `{"error":"input exceeds context length"}`, error: "input exceeds context length"},
		{name: "old server", status: 404, response: "404 page not found", error: "404 page not found"},
		{name: "missing answer", response: `{"answers":{}}`, error: "no answer"},
		{name: "missing probability", response: `{"answers":{"answer":{"type":"noul"}}}`, error: "invalid yes probability"},
		{name: "wrong answer type", response: `{"answers":{"answer":{"type":"choice","choice":"bug"}}}`, error: "unexpected answer type"},
		{name: "malformed response", response: "invalid JSON", error: "invalid character"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch r.URL.Path {
				case "/":
				case "/api/show":
					if tt.showStatus != 0 {
						w.WriteHeader(tt.showStatus)
						io.WriteString(w, `{"error":"model not found"}`)
						return
					}
					capability := tt.capability
					if capability == "" {
						capability = model.CapabilityDecision
					}
					json.NewEncoder(w).Encode(api.ShowResponse{Capabilities: []model.Capability{capability}})
				case "/v1/systemone":
					if tt.status != 0 {
						w.WriteHeader(tt.status)
					}
					io.WriteString(w, tt.response)
				default:
					t.Errorf("unexpected endpoint: %s", r.URL.Path)
				}
			}))
			defer server.Close()
			t.Setenv("OLLAMA_HOST", server.URL)
			t.Setenv("OLLAMA_AUTH", "0")
			out, _, err := executeAsk(t, []string{"nimble", "Is this urgent?", "Checkout is down.", "--json"}, unreadAskInput{})
			if err == nil || !strings.Contains(err.Error(), tt.error) || out != "" {
				t.Fatalf("stdout=%q, err=%v; want %q", out, err, tt.error)
			}
			if tt.status >= 400 {
				var status api.StatusError
				if !errors.As(err, &status) || status.StatusCode != tt.status {
					t.Fatalf("lost API status: %v", err)
				}
			}
		})
	}
}

func TestAskHumanPresentation(t *testing.T) {
	var req api.SystemOneRequest
	json.Unmarshal([]byte(`{"questions":{"refund":{"type":"noul"},"label":{"type":"choice","criteria":{"bug":null,"billing":null}},"urgency":{"type":"score","criteria":["Routine","Soon","Immediate"]}}}`), &req)
	var resp api.SystemOneResponse
	json.Unmarshal([]byte(`{"answers":{"urgency":{"type":"score","score":0.8308},"label":{"type":"choice","choice":"bug","probabilities":{"bug":0.9781}},"refund":{"type":"noul","noul":0.9989}}}`), &resp)
	var out bytes.Buffer
	if err := writeAskResponse(&out, &req, &resp, true, false); err != nil {
		t.Fatal(err)
	}
	if want := "refund: Yes: 99.89%\nlabel: bug (97.81% probability)\nurgency: 0.83 / 2\n"; out.String() != want {
		t.Fatalf("output=%q, want %q", out.String(), want)
	}
	resp.Answers.Set("urgency", json.RawMessage(`{"type":"score","score":3}`))
	out.Reset()
	if err := writeAskResponse(&out, &req, &resp, true, false); err == nil || out.Len() != 0 {
		t.Fatalf("invalid later answer printed partial output: %q, err=%v", out.String(), err)
	}
}

func TestAskCancellation(t *testing.T) {
	started, release := make(chan struct{}), make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/":
		case "/api/show":
			json.NewEncoder(w).Encode(api.ShowResponse{Capabilities: []model.Capability{model.CapabilityDecision}})
		case "/v1/systemone":
			close(started)
			<-release
		}
	}))
	defer server.Close()
	defer close(release)
	t.Setenv("OLLAMA_HOST", server.URL)
	t.Setenv("OLLAMA_AUTH", "0")
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	cmd := NewCLI()
	cmd.SetArgs([]string{"ask", "nimble", "Is this urgent?", "Checkout is down."})
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetErr(io.Discard)
	done := make(chan error, 1)
	go func() { done <- cmd.ExecuteContext(ctx) }()
	select {
	case <-started:
	case <-time.After(5 * time.Second):
		t.Fatal("request did not reach the server")
	}
	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) || out.Len() != 0 {
			t.Fatalf("stdout=%q, err=%v", out.String(), err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("command did not stop after cancellation")
	}
}
