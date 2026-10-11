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

// runDecide runs `ollama run nimble args...` against a decision model with
// stdin piped in. flags are name=value pairs.
func runDecide(t *testing.T, stdin string, args []string, handler http.HandlerFunc, flags ...string) (string, error) {
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
	cmd.Flags().String("format", "", "")
	cmd.Flags().String("think", "", "")
	cmd.Flags().Bool("hidethinking", false, "")
	for _, flag := range flags {
		name, value, _ := strings.Cut(flag, "=")
		if err := cmd.Flags().Set(name, value); err != nil {
			t.Fatal(err)
		}
	}

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
	const refund = `{"Is this a refund request?":{"type":"noul","instructions":"Is this a refund request?"}}`
	for _, tt := range []struct {
		name, stdin string
		args        []string
		questions   string
		answers     string
		want        string
	}{
		{
			name:      "text and a question",
			args:      []string{"I was charged twice", "Is this a refund request?"},
			questions: refund,
			answers:   `{"Is this a refund request?":{"type":"noul","noul":0.2}}`,
			want:      "no  80%\n",
		},
		{
			name:      "piped text",
			stdin:     "I was charged twice",
			args:      []string{"Is this a refund request?"},
			questions: refund,
			answers:   `{"Is this a refund request?":{"type":"noul","noul":0.97}}`,
			want:      "yes  97%\n",
		},
		{
			name:      "questions in order",
			args:      []string{"I was charged twice", "Is this urgent?", "Is this about billing?"},
			questions: `{"Is this urgent?":{"type":"noul","instructions":"Is this urgent?"},"Is this about billing?":{"type":"noul","instructions":"Is this about billing?"}}`,
			answers:   `{"Is this about billing?":{"type":"noul","noul":0.97},"Is this urgent?":{"type":"noul","noul":0.9}}`,
			want:      "Is this urgent?         yes  90%\nIs this about billing?  yes  97%\n",
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var got struct {
				Model     string          `json:"model"`
				State     string          `json:"state"`
				Questions json.RawMessage `json:"questions"`
			}
			out, err := runDecide(t, tt.stdin, tt.args, func(w http.ResponseWriter, r *http.Request) {
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

	const response = `{"model":"nimble","answers":{"Is this urgent?":{"type":"noul","noul":0.9}},"usage":{"input_tokens":12,"output_tokens":1}}`
	out, err := runDecide(t, "", []string{"hello", "Is this urgent?"}, func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, response)
	}, "format=json")
	if err != nil || out != response+"\n" {
		t.Fatalf("json output = %q, %v", out, err)
	}
}

func TestRunDecisionModelErrors(t *testing.T) {
	const tooLong = "text and schema must not exceed 64 KiB"
	_, err := runDecide(t, "", []string{"hello", "Is this urgent?"}, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusRequestEntityTooLarge)
		_, _ = io.WriteString(w, `{"error":"`+tooLong+`"}`)
	})
	var status api.StatusError
	if !errors.As(err, &status) || status.StatusCode != http.StatusRequestEntityTooLarge || status.ErrorMessage != tooLong {
		t.Fatalf("lost server error: %v", err)
	}

	_, err = runDecide(t, "", []string{"hello", "Is this urgent?"}, func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `{"model":"nimble","answers":{}}`)
	})
	if err == nil || !strings.Contains(err.Error(), `no answer to "Is this urgent?"`) {
		t.Fatalf("missing answer: %v", err)
	}

	unexpected := func(w http.ResponseWriter, r *http.Request) { t.Error("asked without text and a question") }
	for _, args := range [][]string{nil, {"hello"}} {
		if _, err := runDecide(t, "", args, unexpected); err == nil || !strings.Contains(err.Error(), "answers questions about text") {
			t.Errorf("args %q: got %v", args, err)
		}
	}
}
