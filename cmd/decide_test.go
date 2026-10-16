package cmd

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

// runDecide runs `ollama run args...` with stdin piped in. nimble is a
// decision model and any other model is a chat model.
func runDecide(t *testing.T, stdin string, handler http.HandlerFunc, args ...string) (string, error) {
	t.Helper()
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/show" && r.Method == http.MethodPost {
			var req api.ShowRequest
			_ = json.NewDecoder(r.Body).Decode(&req)
			capability := model.CapabilityCompletion
			if req.Model == "nimble" {
				capability = model.CapabilityDecision
			}
			_ = json.NewEncoder(w).Encode(api.ShowResponse{Capabilities: []model.Capability{capability}})
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

	cmd, _, err := NewCLI().Find([]string{"run"})
	if err != nil {
		t.Fatal(err)
	}
	cmd.SetContext(t.Context())
	if err := cmd.ParseFlags(args); err != nil {
		return "", err
	}

	oldStdin := os.Stdin
	stdinR, stdinW, _ := os.Pipe()
	defer stdinR.Close()
	_, _ = io.WriteString(stdinW, stdin)
	stdinW.Close()
	os.Stdin = stdinR
	defer func() { os.Stdin = oldStdin }()
	out := captureStdout(t, func() {
		err = RunHandler(cmd, cmd.Flags().Args())
	})
	return out, err
}

func TestRunDecisionModel(t *testing.T) {
	const refund = `{"q1":{"type":"noul","instructions":"Is a refund being requested?"}}`
	for _, tt := range []struct {
		name, stdin string
		args        []string
		state       string
		questions   string
		answers     string
		want        string
	}{
		{
			name:      "yes/no",
			stdin:     "I was charged twice\n",
			args:      []string{"--ask", "Is a refund being requested?"},
			questions: refund,
			answers:   `{"q1":{"type":"noul","noul":0.97}}`,
			want:      "yes  97%\n",
		},
		{
			name:      "state as an argument",
			args:      []string{"I was charged twice", "--ask", "Is a refund being requested?"},
			questions: refund,
			answers:   `{"q1":{"type":"noul","noul":0.2}}`,
			want:      "no  80%\n",
		},
		{
			name:      "JSON state",
			args:      []string{"--ask", "Is a refund being requested?", `{"ticket": "I was charged twice"}`},
			state:     `{"ticket":"I was charged twice"}`,
			questions: refund,
			answers:   `{"q1":{"type":"noul","noul":0.97}}`,
			want:      "yes  97%\n",
		},
		{
			name:      "labels",
			stdin:     "I was charged twice",
			args:      []string{"--ask", "Which team should handle this?", "--choice", "billing=Payments and refunds", "--choice", "technical"},
			questions: `{"q1":{"type":"choice","instructions":"Which team should handle this?","criteria":{"billing":"Payments and refunds","technical":null}}}`,
			answers:   `{"q1":{"type":"choice","choice":"billing","probabilities":{"billing":0.91,"technical":0.09},"confidence":0.6}}`,
			want:      "billing  91%\n",
		},
		{
			name:      "levels",
			stdin:     "I was charged twice",
			args:      []string{"--ask", "How urgent is this?", "--level", "Routine", "--level", "Soon", "--level", "Immediate"},
			questions: `{"q1":{"type":"score","instructions":"How urgent is this?","criteria":["Routine","Soon","Immediate"]}}`,
			answers:   `{"q1":{"type":"score","score":1.83,"legend":{"0":"Routine","1":"Soon","2":"Immediate"},"probabilities":{"0":0.04,"1":0.09,"2":0.87},"confidence":0.7}}`,
			want:      "Immediate  87%  1.8 / 2\n",
		},
		{
			name:      "most likely level, not the nearest",
			stdin:     "I was charged twice",
			args:      []string{"--ask", "How urgent is this?", "--level", "Routine", "--level", "Soon", "--level", "Immediate"},
			questions: `{"q1":{"type":"score","instructions":"How urgent is this?","criteria":["Routine","Soon","Immediate"]}}`,
			answers:   `{"q1":{"type":"score","score":1.1,"legend":{"0":"Routine","1":"Soon","2":"Immediate"},"probabilities":{"0":0.45,"1":0,"2":0.55}}}`,
			want:      "Immediate  55%  1.1 / 2\n",
		},
		{
			name:  "questions in order",
			stdin: "I was charged twice",
			args: []string{
				"--ask", "Is a refund being requested?",
				"--ask", "Which team?", "--choice", "billing", "--choice", "technical",
				"--ask", "How urgent?", "--level", "routine", "--level", "soon", "--level", "immediate",
			},
			questions: `{"q1":{"type":"noul","instructions":"Is a refund being requested?"},"q2":{"type":"choice","instructions":"Which team?","criteria":{"billing":null,"technical":null}},"q3":{"type":"score","instructions":"How urgent?","criteria":["routine","soon","immediate"]}}`,
			answers:   `{"q3":{"type":"score","score":1.2,"legend":{"0":"routine","1":"soon","2":"immediate"},"probabilities":{"0":0.1,"1":0.6,"2":0.3}},"q2":{"type":"choice","choice":"billing","probabilities":{"billing":0.91,"technical":0.09}},"q1":{"type":"noul","noul":0.97}}`,
			want:      "Is a refund being requested?  yes      97%\nWhich team?                   billing  91%\nHow urgent?                   soon     60%  1.2 / 2\n",
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var got struct {
				Model     string          `json:"model"`
				Runner    string          `json:"runner"`
				State     json.RawMessage `json:"state"`
				Questions json.RawMessage `json:"questions"`
			}
			out, err := runDecide(t, tt.stdin, func(w http.ResponseWriter, r *http.Request) {
				if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
					t.Error(err)
				}
				_, _ = io.WriteString(w, `{"model":"nimble","answers":`+tt.answers+`}`)
			}, append([]string{"nimble", "--runner", "llamacpp"}, tt.args...)...)
			if err != nil {
				t.Fatal(err)
			}
			if tt.state == "" {
				tt.state = `"I was charged twice"`
			}
			if got.Model != "nimble" || got.Runner != "llamacpp" || string(got.State) != tt.state || string(got.Questions) != tt.questions {
				t.Fatalf("unexpected request: %+v", got)
			}
			if out != tt.want {
				t.Fatalf("got output %q, want %q", out, tt.want)
			}
		})
	}

	const response = `{"model":"nimble","answers":{"q1":{"type":"noul","noul":0.9}},"usage":{"input_tokens":12,"output_tokens":1}}`
	out, err := runDecide(t, "", func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, response)
	}, "nimble", "hello", "--ask", "Is this urgent?", "--format", "json")
	if err != nil || out != response+"\n" {
		t.Fatalf("json output = %q, %v", out, err)
	}
}

func TestYesNoQuestions(t *testing.T) {
	got, err := json.Marshal(yesNoQuestions("Is this spam?", "Is it urgent?"))
	if err != nil {
		t.Fatal(err)
	}
	const want = `{"q1":{"type":"noul","instructions":"Is this spam?"},"q2":{"type":"noul","instructions":"Is it urgent?"}}`
	if string(got) != want {
		t.Fatalf("got %s, want %s", got, want)
	}
}

func TestRunDecisionModelErrors(t *testing.T) {
	const tooLong = "text and schema must not exceed 64 KiB"
	_, err := runDecide(t, "", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusRequestEntityTooLarge)
		_, _ = io.WriteString(w, `{"error":"`+tooLong+`"}`)
	}, "nimble", "hello", "--ask", "Is this urgent?")
	var status api.StatusError
	if !errors.As(err, &status) || status.StatusCode != http.StatusRequestEntityTooLarge || status.ErrorMessage != tooLong {
		t.Fatalf("lost server error: %v", err)
	}

	_, err = runDecide(t, "", func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `{"model":"nimble","answers":{}}`)
	}, "nimble", "hello", "--ask", "Is this urgent?")
	if err == nil || !strings.Contains(err.Error(), `no answer to "Is this urgent?"`) {
		t.Fatalf("missing answer: %v", err)
	}

	unexpected := func(w http.ResponseWriter, r *http.Request) { t.Error("asked without text and a question") }
	for _, tt := range []struct {
		args []string
		err  string
	}{
		{[]string{"nimble"}, "answers questions about text"},
		{[]string{"nimble", "hello"}, "answers questions about text"},
		{[]string{"nimble", "--ask", "Is this urgent?"}, "answers questions about text"},
		{[]string{"nimble", "hello", "--choice", "billing", "--ask", "Which team?"}, "must come after the --ask it answers"},
		{[]string{"nimble", "hello", "--ask", "How urgent?", "--choice", "billing", "--level", "low"}, "--choice or --level, not both"},
		{[]string{"gemma4", "Win a free cruise", "--ask", "Is this spam?"}, "gemma4 is not a decision model"},
	} {
		if _, err := runDecide(t, "", unexpected, tt.args...); err == nil || !strings.Contains(err.Error(), tt.err) {
			t.Errorf("args %q: got %v", tt.args, err)
		}
	}
}
