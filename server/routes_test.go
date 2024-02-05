package server

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"sort"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/jmorganca/ollama/api"
	"github.com/jmorganca/ollama/llm"
	"github.com/jmorganca/ollama/parser"
	"github.com/jmorganca/ollama/version"
)

func setupServer(t *testing.T) (*Server, error) {
	t.Helper()

	return NewServer()
}

func Test_Routes(t *testing.T) {
	type testCase struct {
		Name     string
		Method   string
		Path     string
		Setup    func(t *testing.T, req *http.Request)
		Expected func(t *testing.T, resp *http.Response)
	}

	createTestFile := func(t *testing.T, name string) string {
		f, err := os.CreateTemp(t.TempDir(), name)
		assert.Nil(t, err)
		defer f.Close()

		_, err = f.Write([]byte("GGUF"))
		assert.Nil(t, err)
		_, err = f.Write([]byte{0x2, 0})
		assert.Nil(t, err)

		return f.Name()
	}

	createTestModel := func(t *testing.T, name string) {
		fname := createTestFile(t, "ollama-model")

		modelfile := strings.NewReader(fmt.Sprintf("FROM %s\nPARAMETER seed 42\nPARAMETER top_p 0.9\nPARAMETER stop foo\nPARAMETER stop bar", fname))
		commands, err := parser.Parse(modelfile)
		assert.Nil(t, err)
		fn := func(resp api.ProgressResponse) {
			t.Logf("Status: %s", resp.Status)
		}
		err = CreateModel(context.TODO(), name, "", commands, fn)
		assert.Nil(t, err)
	}

	testCases := []testCase{
		{
			Name:   "Version Handler",
			Method: http.MethodGet,
			Path:   "/api/version",
			Setup: func(t *testing.T, req *http.Request) {
			},
			Expected: func(t *testing.T, resp *http.Response) {
				contentType := resp.Header.Get("Content-Type")
				assert.Equal(t, contentType, "application/json; charset=utf-8")
				body, err := io.ReadAll(resp.Body)
				assert.Nil(t, err)
				assert.Equal(t, fmt.Sprintf(`{"version":"%s"}`, version.Version), string(body))
			},
		},
		{
			Name:   "Tags Handler (no tags)",
			Method: http.MethodGet,
			Path:   "/api/tags",
			Expected: func(t *testing.T, resp *http.Response) {
				contentType := resp.Header.Get("Content-Type")
				assert.Equal(t, contentType, "application/json; charset=utf-8")
				body, err := io.ReadAll(resp.Body)
				assert.Nil(t, err)

				var modelList api.ListResponse

				err = json.Unmarshal(body, &modelList)
				assert.Nil(t, err)

				assert.Equal(t, 0, len(modelList.Models))
			},
		},
		{
			Name:   "Tags Handler (yes tags)",
			Method: http.MethodGet,
			Path:   "/api/tags",
			Setup: func(t *testing.T, req *http.Request) {
				createTestModel(t, "test-model")
			},
			Expected: func(t *testing.T, resp *http.Response) {
				contentType := resp.Header.Get("Content-Type")
				assert.Equal(t, contentType, "application/json; charset=utf-8")
				body, err := io.ReadAll(resp.Body)
				assert.Nil(t, err)

				var modelList api.ListResponse
				err = json.Unmarshal(body, &modelList)
				assert.Nil(t, err)

				assert.Equal(t, 1, len(modelList.Models))
				assert.Equal(t, modelList.Models[0].Name, "test-model:latest")
			},
		},
		{
			Name:   "Create Model Handler",
			Method: http.MethodPost,
			Path:   "/api/create",
			Setup: func(t *testing.T, req *http.Request) {
				f, err := os.CreateTemp(t.TempDir(), "ollama-model")
				assert.Nil(t, err)
				defer f.Close()

				stream := false
				createReq := api.CreateRequest{
					Name:      "t-bone",
					Modelfile: fmt.Sprintf("FROM %s", f.Name()),
					Stream:    &stream,
				}
				jsonData, err := json.Marshal(createReq)
				assert.Nil(t, err)

				req.Body = io.NopCloser(bytes.NewReader(jsonData))
			},
			Expected: func(t *testing.T, resp *http.Response) {
				contentType := resp.Header.Get("Content-Type")
				assert.Equal(t, "application/json", contentType)
				_, err := io.ReadAll(resp.Body)
				assert.Nil(t, err)
				assert.Equal(t, resp.StatusCode, 200)

				model, err := GetModel("t-bone")
				assert.Nil(t, err)
				assert.Equal(t, "t-bone:latest", model.ShortName)
			},
		},
		{
			Name:   "Copy Model Handler",
			Method: http.MethodPost,
			Path:   "/api/copy",
			Setup: func(t *testing.T, req *http.Request) {
				createTestModel(t, "hamshank")
				copyReq := api.CopyRequest{
					Source:      "hamshank",
					Destination: "beefsteak",
				}
				jsonData, err := json.Marshal(copyReq)
				assert.Nil(t, err)

				req.Body = io.NopCloser(bytes.NewReader(jsonData))
			},
			Expected: func(t *testing.T, resp *http.Response) {
				model, err := GetModel("beefsteak")
				assert.Nil(t, err)
				assert.Equal(t, "beefsteak:latest", model.ShortName)
			},
		},
		{
			Name:   "Show Model Handler",
			Method: http.MethodPost,
			Path:   "/api/show",
			Setup: func(t *testing.T, req *http.Request) {
				createTestModel(t, "show-model")
				showReq := api.ShowRequest{Model: "show-model"}
				jsonData, err := json.Marshal(showReq)
				assert.Nil(t, err)
				req.Body = io.NopCloser(bytes.NewReader(jsonData))
			},
			Expected: func(t *testing.T, resp *http.Response) {
				contentType := resp.Header.Get("Content-Type")
				assert.Equal(t, contentType, "application/json; charset=utf-8")
				body, err := io.ReadAll(resp.Body)
				assert.Nil(t, err)

				var showResp api.ShowResponse
				err = json.Unmarshal(body, &showResp)
				assert.Nil(t, err)

				var params []string
				paramsSplit := strings.Split(showResp.Parameters, "\n")
				for _, p := range paramsSplit {
					params = append(params, strings.Join(strings.Fields(p), " "))
				}
				sort.Strings(params)
				expectedParams := []string{
					"seed 42",
					"stop \"bar\"",
					"stop \"foo\"",
					"top_p 0.9",
				}
				assert.Equal(t, expectedParams, params)
			},
		},
	}

	s, err := setupServer(t)
	assert.Nil(t, err)

	router := s.GenerateRoutes()

	httpSrv := httptest.NewServer(router)
	t.Cleanup(httpSrv.Close)

	workDir, err := os.MkdirTemp("", "ollama-test")
	assert.Nil(t, err)
	defer os.RemoveAll(workDir)
	os.Setenv("OLLAMA_MODELS", workDir)

	for _, tc := range testCases {
		t.Logf("Running Test: [%s]", tc.Name)
		u := httpSrv.URL + tc.Path
		req, err := http.NewRequestWithContext(context.TODO(), tc.Method, u, nil)
		assert.Nil(t, err)

		if tc.Setup != nil {
			tc.Setup(t, req)
		}

		resp, err := httpSrv.Client().Do(req)
		assert.Nil(t, err)
		defer resp.Body.Close()

		if tc.Expected != nil {
			tc.Expected(t, resp)
		}

	}
}

func Test_ChatPrompt(t *testing.T) {
	tests := []struct {
		name     string
		template string
		chat     *ChatHistory
		numCtx   int
		runner   MockLLM
		want     string
		wantErr  string
	}{
		{
			name:     "Single Message",
			template: "[INST] {{ .System }} {{ .Prompt }} [/INST]",
			chat: &ChatHistory{
				Prompts: []PromptVars{
					{
						System: "You are a Wizard.",
						Prompt: "What are the potion ingredients?",
						First:  true,
					},
				},
				LastSystem: "You are a Wizard.",
			},
			numCtx: 1,
			runner: MockLLM{
				encoding: []int{1}, // fit the ctxLen
			},
			want: "[INST] You are a Wizard. What are the potion ingredients? [/INST]",
		},
		{
			name:     "First Message",
			template: "[INST] {{if .First}}Hello!{{end}} {{ .System }} {{ .Prompt }} [/INST]",
			chat: &ChatHistory{
				Prompts: []PromptVars{
					{
						System:   "You are a Wizard.",
						Prompt:   "What are the potion ingredients?",
						Response: "eye of newt",
						First:    true,
					},
					{
						Prompt: "Anything else?",
					},
				},
				LastSystem: "You are a Wizard.",
			},
			numCtx: 2,
			runner: MockLLM{
				encoding: []int{1}, // fit the ctxLen
			},
			want: "[INST] Hello! You are a Wizard. What are the potion ingredients? [/INST]eye of newt[INST]   Anything else? [/INST]",
		},
		{
			name:     "Message History",
			template: "[INST] {{ .System }} {{ .Prompt }} [/INST]",
			chat: &ChatHistory{
				Prompts: []PromptVars{
					{
						System:   "You are a Wizard.",
						Prompt:   "What are the potion ingredients?",
						Response: "sugar",
						First:    true,
					},
					{
						Prompt: "Anything else?",
					},
				},
				LastSystem: "You are a Wizard.",
			},
			numCtx: 4,
			runner: MockLLM{
				encoding: []int{1}, // fit the ctxLen, 1 for each message
			},
			want: "[INST] You are a Wizard. What are the potion ingredients? [/INST]sugar[INST]  Anything else? [/INST]",
		},
		{
			name:     "Assistant Only",
			template: "[INST] {{ .System }} {{ .Prompt }} [/INST]",
			chat: &ChatHistory{
				Prompts: []PromptVars{
					{
						Response: "everything nice",
						First:    true,
					},
				},
			},
			numCtx: 1,
			runner: MockLLM{
				encoding: []int{1},
			},
			want: "[INST]   [/INST]everything nice",
		},
		{
			name:     "Message History Truncated, No System",
			template: "[INST] {{ .System }} {{ .Prompt }} [/INST]",
			chat: &ChatHistory{
				Prompts: []PromptVars{
					{
						Prompt:   "What are the potion ingredients?",
						Response: "sugar",
						First:    true,
					},
					{
						Prompt:   "Anything else?",
						Response: "spice",
					},
					{
						Prompt: "... and?",
					},
				},
			},
			numCtx: 2, // only 1 message from history and most recent message
			runner: MockLLM{
				encoding: []int{1},
			},
			want: "[INST]  Anything else? [/INST]spice[INST]  ... and? [/INST]",
		},
		{
			name:     "System is Preserved when Truncated",
			template: "[INST] {{ .System }} {{ .Prompt }} [/INST]",
			chat: &ChatHistory{
				Prompts: []PromptVars{
					{
						Prompt:   "What are the magic words?",
						Response: "abracadabra",
					},
					{
						Prompt: "What is the spell for invisibility?",
					},
				},
				LastSystem: "You are a wizard.",
			},
			numCtx: 2,
			runner: MockLLM{
				encoding: []int{1},
			},
			want: "[INST] You are a wizard. What is the spell for invisibility? [/INST]",
		},
		{
			name:     "System is Preserved when Length Exceeded",
			template: "[INST] {{ .System }} {{ .Prompt }} [/INST]",
			chat: &ChatHistory{
				Prompts: []PromptVars{
					{
						Prompt:   "What are the magic words?",
						Response: "abracadabra",
					},
					{
						Prompt: "What is the spell for invisibility?",
					},
				},
				LastSystem: "You are a wizard.",
			},
			numCtx: 1,
			runner: MockLLM{
				encoding: []int{1},
			},
			want: "[INST] You are a wizard. What is the spell for invisibility? [/INST]",
		},
		{
			name:     "First is Preserved when Truncated",
			template: "[INST] {{ if .First }}{{ .System }} {{ end }}{{ .Prompt }} [/INST]",

			chat: &ChatHistory{
				Prompts: []PromptVars{
					// first message omitted for test
					{
						Prompt:   "Do you have a magic hat?",
						Response: "Of course.",
					},
					{
						Prompt: "What is the spell for invisibility?",
					},
				},
				LastSystem: "You are a wizard.",
			},
			numCtx: 3, // two most recent messages and room for system message
			runner: MockLLM{
				encoding: []int{1},
			},
			want: "[INST] You are a wizard. Do you have a magic hat? [/INST]Of course.[INST] What is the spell for invisibility? [/INST]",
		},
		{
			name:     "Most recent message is returned when longer than ctxLen",
			template: "[INST] {{ .Prompt }} [/INST]",

			chat: &ChatHistory{
				Prompts: []PromptVars{
					{
						Prompt: "What is the spell for invisibility?",
						First:  true,
					},
				},
			},
			numCtx: 1, // two most recent messages
			runner: MockLLM{
				encoding: []int{1, 2},
			},
			want: "[INST] What is the spell for invisibility? [/INST]",
		},
	}

	for _, testCase := range tests {
		tt := testCase
		m := &Model{
			Template: tt.template,
		}
		t.Run(tt.name, func(t *testing.T) {
			loaded.runner = &tt.runner
			loaded.Options = &api.Options{
				Runner: api.Runner{
					NumCtx: tt.numCtx,
				},
			}
			// TODO: add tests for trimming images
			got, _, err := trimmedPrompt(context.Background(), tt.chat, m)
			if tt.wantErr != "" {
				if err == nil {
					t.Errorf("ChatPrompt() expected error, got nil")
				}
				if !strings.Contains(err.Error(), tt.wantErr) {
					t.Errorf("ChatPrompt() error = %v, wantErr %v", err, tt.wantErr)
				}
			}
			if got != tt.want {
				t.Errorf("ChatPrompt() got = %v, want %v", got, tt.want)
			}
		})
	}
}

type MockLLM struct {
	encoding []int
}

func (llm *MockLLM) Predict(ctx context.Context, pred llm.PredictOpts, fn func(llm.PredictResult)) error {
	return nil
}

func (llm *MockLLM) Encode(ctx context.Context, prompt string) ([]int, error) {
	return llm.encoding, nil
}

func (llm *MockLLM) Decode(ctx context.Context, tokens []int) (string, error) {
	return "", nil
}

func (llm *MockLLM) Embedding(ctx context.Context, input string) ([]float64, error) {
	return []float64{}, nil
}

func (llm *MockLLM) Close() {
	// do nothing
}
