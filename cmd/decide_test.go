package cmd

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

// runDecide runs `ollama run nimble args...` against a decision model, with
// stdin piped in.
func runDecide(t *testing.T, format, stdin string, args []string, handler http.HandlerFunc) (string, error) {
	t.Helper()
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/show" && r.Method == http.MethodPost {
			_ = json.NewEncoder(w).Encode(api.ShowResponse{Capabilities: []model.Capability{model.CapabilityDecision}})
			return
		}
		if r.URL.Path == "/v1/systemone" && r.Method == http.MethodPost {
			handler(w, r)
			return
		}
		http.NotFound(w, r)
	}))
	t.Setenv("OLLAMA_HOST", mockServer.URL)
	t.Cleanup(mockServer.Close)

	cmd := &cobra.Command{}
	cmd.SetContext(t.Context())
	cmd.Flags().String("keepalive", "", "")
	cmd.Flags().Bool("verbose", false, "")
	cmd.Flags().Bool("insecure", false, "")
	cmd.Flags().Bool("nowordwrap", false, "")
	cmd.Flags().String("format", format, "")
	cmd.Flags().String("think", "", "")
	cmd.Flags().Bool("hidethinking", false, "")

	oldStdin, oldStdout := os.Stdin, os.Stdout
	stdinR, stdinW, _ := os.Pipe()
	_, _ = io.WriteString(stdinW, stdin)
	stdinW.Close()
	stdoutR, stdoutW, _ := os.Pipe()
	os.Stdin, os.Stdout = stdinR, stdoutW
	err := RunHandler(cmd, append([]string{"nimble"}, args...))
	stdoutW.Close()
	os.Stdin, os.Stdout = oldStdin, oldStdout
	var out bytes.Buffer
	_, _ = io.Copy(&out, stdoutR)
	return out.String(), err
}

func TestRunDecisionModel(t *testing.T) {
	const refund = `{"Is this a refund request?":{"instructions":"Is this a refund request?","type":"noul"}}`
	for _, tt := range []struct {
		name, stdin string
		args        []string
		questions   string
		answers     string
		want        string
	}{
		{
			"text and a question", "",
			[]string{"I was charged twice", "Is this a refund request?"},
			refund,
			`{"Is this a refund request?":{"type":"noul","noul":0.2}}`, "no (80%)\n",
		},
		{
			"piped text", "I was charged twice",
			[]string{"Is this a refund request?"},
			refund,
			`{"Is this a refund request?":{"type":"noul","noul":0.2}}`, "no (80%)\n",
		},
		{
			"questions in order", "",
			[]string{"I was charged twice", "Is this urgent?", "Is this about billing?"},
			`{"Is this urgent?":{"instructions":"Is this urgent?","type":"noul"},"Is this about billing?":{"instructions":"Is this about billing?","type":"noul"}}`,
			`{"Is this about billing?":{"type":"noul","noul":0.97},"Is this urgent?":{"type":"noul","noul":0.9}}`, "yes (90%)\nyes (97%)\n",
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var got struct {
				Model     string          `json:"model"`
				State     string          `json:"state"`
				Questions json.RawMessage `json:"questions"`
			}
			out, err := runDecide(t, "", tt.stdin, tt.args, func(w http.ResponseWriter, r *http.Request) {
				if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
					t.Error(err)
				}
				_, _ = io.WriteString(w, `{"model":"nimble","answers":`+tt.answers+`}`)
			})
			if err != nil {
				t.Fatal(err)
			}
			if got.Model != "nimble" || got.State != "I was charged twice" || string(got.Questions) != tt.questions {
				t.Fatalf("unexpected request: %+v", got)
			}
			if out != tt.want {
				t.Fatalf("got output %q, want %q", out, tt.want)
			}
		})
	}

	const response = `{"model":"nimble","answers":{"Is this urgent?":{"type":"noul","noul":0.9}}}`
	out, err := runDecide(t, "json", "", []string{"hello", "Is this urgent?"}, func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, response)
	})
	if err != nil || out != response+"\n" {
		t.Fatalf("json output = %q, %v", out, err)
	}
}

func TestRunDecisionModelErrors(t *testing.T) {
	const tooLong = "prompt 0 has 4096 tokens; expected 1–2048 (input is never truncated)"
	_, err := runDecide(t, "", "", []string{"hello", "Is this urgent?"}, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = io.WriteString(w, `{"error":"`+tooLong+`"}`)
	})
	var status api.StatusError
	if !errors.As(err, &status) || status.StatusCode != http.StatusBadRequest || status.ErrorMessage != tooLong {
		t.Fatalf("lost server error: %v", err)
	}

	_, err = runDecide(t, "", "", []string{"hello", "Is this urgent?"}, func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `{"model":"nimble","answers":{}}`)
	})
	if err == nil || !strings.Contains(err.Error(), `no answer to "Is this urgent?"`) {
		t.Fatalf("missing answer: %v", err)
	}

	unexpected := func(w http.ResponseWriter, r *http.Request) { t.Error("asked without text and a question") }
	for _, args := range [][]string{nil, {"hello"}} {
		if _, err := runDecide(t, "", "", args, unexpected); err == nil || !strings.Contains(err.Error(), "answers questions about text") {
			t.Errorf("args %q: got %v", args, err)
		}
	}
}
