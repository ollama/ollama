package server

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"sort"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/openai"
	"github.com/ollama/ollama/parser"
	"github.com/ollama/ollama/types/model"
	"github.com/ollama/ollama/version"
)

func createTestFile(t *testing.T, name string) string {
	t.Helper()

	f, err := os.CreateTemp(t.TempDir(), name)
	if err != nil {
		t.Fatalf("failed to create temp file: %v", err)
	}
	defer f.Close()

	err = binary.Write(f, binary.LittleEndian, []byte("GGUF"))
	if err != nil {
		t.Fatalf("failed to write to file: %v", err)
	}

	err = binary.Write(f, binary.LittleEndian, uint32(3))
	if err != nil {
		t.Fatalf("failed to write to file: %v", err)
	}

	err = binary.Write(f, binary.LittleEndian, uint64(0))
	if err != nil {
		t.Fatalf("failed to write to file: %v", err)
	}

	err = binary.Write(f, binary.LittleEndian, uint64(0))
	if err != nil {
		t.Fatalf("failed to write to file: %v", err)
	}

	return f.Name()
}

// equalStringSlices checks if two slices of strings are equal.
func equalStringSlices(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func Test_Routes(t *testing.T) {
	type testCase struct {
		Name     string
		Method   string
		Path     string
		Setup    func(t *testing.T, req *http.Request)
		Expected func(t *testing.T, resp *http.Response)
	}

	createTestModel := func(t *testing.T, name string) {
		t.Helper()

		fname := createTestFile(t, "ollama-model")

		r := strings.NewReader(fmt.Sprintf("FROM %s\nPARAMETER seed 42\nPARAMETER top_p 0.9\nPARAMETER stop foo\nPARAMETER stop bar", fname))
		modelfile, err := parser.ParseFile(r)
		if err != nil {
			t.Fatalf("failed to parse file: %v", err)
		}
		fn := func(resp api.ProgressResponse) {
			t.Logf("Status: %s", resp.Status)
		}
		err = CreateModel(context.TODO(), model.ParseName(name), "", "", modelfile, fn)
		if err != nil {
			t.Fatalf("failed to create model: %v", err)
		}
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
				if contentType != "application/json; charset=utf-8" {
					t.Errorf("expected content type application/json; charset=utf-8, got %s", contentType)
				}
				body, err := io.ReadAll(resp.Body)
				if err != nil {
					t.Fatalf("failed to read response body: %v", err)
				}
				expectedBody := fmt.Sprintf(`{"version":"%s"}`, version.Version)
				if string(body) != expectedBody {
					t.Errorf("expected body %s, got %s", expectedBody, string(body))
				}
			},
		},
		{
			Name:   "Tags Handler (no tags)",
			Method: http.MethodGet,
			Path:   "/api/tags",
			Expected: func(t *testing.T, resp *http.Response) {
				contentType := resp.Header.Get("Content-Type")
				if contentType != "application/json; charset=utf-8" {
					t.Errorf("expected content type application/json; charset=utf-8, got %s", contentType)
				}
				body, err := io.ReadAll(resp.Body)
				if err != nil {
					t.Fatalf("failed to read response body: %v", err)
				}

				var modelList api.ListResponse

				err = json.Unmarshal(body, &modelList)
				if err != nil {
					t.Fatalf("failed to unmarshal response body: %v", err)
				}

				if modelList.Models == nil || len(modelList.Models) != 0 {
					t.Errorf("expected empty model list, got %v", modelList.Models)
				}
			},
		},
		{
			Name:   "openai empty list",
			Method: http.MethodGet,
			Path:   "/v1/models",
			Expected: func(t *testing.T, resp *http.Response) {
				contentType := resp.Header.Get("Content-Type")
				if contentType != "application/json" {
					t.Errorf("expected content type application/json, got %s", contentType)
				}
				body, err := io.ReadAll(resp.Body)
				if err != nil {
					t.Fatalf("failed to read response body: %v", err)
				}

				var modelList openai.ListCompletion
				err = json.Unmarshal(body, &modelList)
				if err != nil {
					t.Fatalf("failed to unmarshal response body: %v", err)
				}

				if modelList.Object != "list" || len(modelList.Data) != 0 {
					t.Errorf("expected empty model list, got %v", modelList.Data)
				}
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
				if contentType != "application/json; charset=utf-8" {
					t.Errorf("expected content type application/json; charset=utf-8, got %s", contentType)
				}
				body, err := io.ReadAll(resp.Body)
				if err != nil {
					t.Fatalf("failed to read response body: %v", err)
				}

				if strings.Contains(string(body), "expires_at") {
					t.Errorf("response body should not contain 'expires_at'")
				}

				var modelList api.ListResponse
				err = json.Unmarshal(body, &modelList)
				if err != nil {
					t.Fatalf("failed to unmarshal response body: %v", err)
				}

				if len(modelList.Models) != 1 || modelList.Models[0].Name != "test-model:latest" {
					t.Errorf("expected model 'test-model:latest', got %v", modelList.Models)
				}
			},
		},
		{
			Name:   "Delete Model Handler",
			Method: http.MethodDelete,
			Path:   "/api/delete",
			Setup: func(t *testing.T, req *http.Request) {
				createTestModel(t, "model-to-delete")

				deleteReq := api.DeleteRequest{
					Name: "model-to-delete",
				}
				jsonData, err := json.Marshal(deleteReq)
				if err != nil {
					t.Fatalf("failed to marshal delete request: %v", err)
				}

				req.Body = io.NopCloser(bytes.NewReader(jsonData))
			},
			Expected: func(t *testing.T, resp *http.Response) {
				if resp.StatusCode != http.StatusOK {
					t.Errorf("expected status code 200, got %d", resp.StatusCode)
				}

				// Verify the model was deleted
				_, err := GetModel("model-to-delete")
				if err == nil || !os.IsNotExist(err) {
					t.Errorf("expected model to be deleted, got error %v", err)
				}
			},
		},
		{
			Name:   "Delete Non-existent Model",
			Method: http.MethodDelete,
			Path:   "/api/delete",
			Setup: func(t *testing.T, req *http.Request) {
				deleteReq := api.DeleteRequest{
					Name: "non-existent-model",
				}
				jsonData, err := json.Marshal(deleteReq)
				if err != nil {
					t.Fatalf("failed to marshal delete request: %v", err)
				}

				req.Body = io.NopCloser(bytes.NewReader(jsonData))
			},
			Expected: func(t *testing.T, resp *http.Response) {
				if resp.StatusCode != http.StatusNotFound {
					t.Errorf("expected status code 404, got %d", resp.StatusCode)
				}

				body, err := io.ReadAll(resp.Body)
				if err != nil {
					t.Fatalf("failed to read response body: %v", err)
				}

				var errorResp map[string]string
				err = json.Unmarshal(body, &errorResp)
				if err != nil {
					t.Fatalf("failed to unmarshal response body: %v", err)
				}

				if !strings.Contains(errorResp["error"], "not found") {
					t.Errorf("expected error message to contain 'not found', got %s", errorResp["error"])
				}
			},
		},
		{
			Name:   "openai list models with tags",
			Method: http.MethodGet,
			Path:   "/v1/models",
			Expected: func(t *testing.T, resp *http.Response) {
				contentType := resp.Header.Get("Content-Type")
				if contentType != "application/json" {
					t.Errorf("expected content type application/json, got %s", contentType)
				}
				body, err := io.ReadAll(resp.Body)
				if err != nil {
					t.Fatalf("failed to read response body: %v", err)
				}

				var modelList openai.ListCompletion
				err = json.Unmarshal(body, &modelList)
				if err != nil {
					t.Fatalf("failed to unmarshal response body: %v", err)
				}

				if len(modelList.Data) != 1 || modelList.Data[0].Id != "test-model:latest" || modelList.Data[0].OwnedBy != "library" {
					t.Errorf("expected model 'test-model:latest' owned by 'library', got %v", modelList.Data)
				}
			},
		},
		{
			Name:   "Create Model Handler",
			Method: http.MethodPost,
			Path:   "/api/create",
			Setup: func(t *testing.T, req *http.Request) {
				fname := createTestFile(t, "ollama-model")

				stream := false
				createReq := api.CreateRequest{
					Name:      "t-bone",
					Modelfile: fmt.Sprintf("FROM %s", fname),
					Stream:    &stream,
				}
				jsonData, err := json.Marshal(createReq)
				if err != nil {
					t.Fatalf("failed to marshal create request: %v", err)
				}

				req.Body = io.NopCloser(bytes.NewReader(jsonData))
			},
			Expected: func(t *testing.T, resp *http.Response) {
				contentType := resp.Header.Get("Content-Type")
				if contentType != "application/json" {
					t.Errorf("expected content type application/json, got %s", contentType)
				}
				_, err := io.ReadAll(resp.Body)
				if err != nil {
					t.Fatalf("failed to read response body: %v", err)
				}
				if resp.StatusCode != http.StatusOK { // Updated line
					t.Errorf("expected status code 200, got %d", resp.StatusCode)
				}

				model, err := GetModel("t-bone")
				if err != nil {
					t.Fatalf("failed to get model: %v", err)
				}
				if model.ShortName != "t-bone:latest" {
					t.Errorf("expected model name 't-bone:latest', got %s", model.ShortName)
				}
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
				if err != nil {
					t.Fatalf("failed to marshal copy request: %v", err)
				}

				req.Body = io.NopCloser(bytes.NewReader(jsonData))
			},
			Expected: func(t *testing.T, resp *http.Response) {
				model, err := GetModel("beefsteak")
				if err != nil {
					t.Fatalf("failed to get model: %v", err)
				}
				if model.ShortName != "beefsteak:latest" {
					t.Errorf("expected model name 'beefsteak:latest', got %s", model.ShortName)
				}
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
				if err != nil {
					t.Fatalf("failed to marshal show request: %v", err)
				}
				req.Body = io.NopCloser(bytes.NewReader(jsonData))
			},
			Expected: func(t *testing.T, resp *http.Response) {
				contentType := resp.Header.Get("Content-Type")
				if contentType != "application/json; charset=utf-8" {
					t.Errorf("expected content type application/json; charset=utf-8, got %s", contentType)
				}
				body, err := io.ReadAll(resp.Body)
				if err != nil {
					t.Fatalf("failed to read response body: %v", err)
				}

				var showResp api.ShowResponse
				err = json.Unmarshal(body, &showResp)
				if err != nil {
					t.Fatalf("failed to unmarshal response body: %v", err)
				}

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
				if !equalStringSlices(params, expectedParams) {
					t.Errorf("expected parameters %v, got %v", expectedParams, params)
				}
				paramCount, ok := showResp.ModelInfo["general.parameter_count"].(float64)
				if !ok {
					t.Fatalf("expected parameter count to be a float64, got %T", showResp.ModelInfo["general.parameter_count"])
				}
				if math.Abs(paramCount) > 1e-9 {
					t.Errorf("expected parameter count to be 0, got %f", paramCount)
				}
			},
		},
		{
			Name:   "openai retrieve model handler",
			Method: http.MethodGet,
			Path:   "/v1/models/show-model",
			Expected: func(t *testing.T, resp *http.Response) {
				contentType := resp.Header.Get("Content-Type")
				if contentType != "application/json" {
					t.Errorf("expected content type application/json, got %s", contentType)
				}
				body, err := io.ReadAll(resp.Body)
				if err != nil {
					t.Fatalf("failed to read response body: %v", err)
				}

				var retrieveResp api.RetrieveModelResponse
				err = json.Unmarshal(body, &retrieveResp)
				if err != nil {
					t.Fatalf("failed to unmarshal response body: %v", err)
				}

				if retrieveResp.Id != "show-model" || retrieveResp.OwnedBy != "library" {
					t.Errorf("expected model 'show-model' owned by 'library', got %v", retrieveResp)
				}
			},
		},
	}

	t.Setenv("OLLAMA_MODELS", t.TempDir())

	s := &Server{}
	router := s.GenerateRoutes()

	httpSrv := httptest.NewServer(router)
	t.Cleanup(httpSrv.Close)

	for _, tc := range testCases {
		t.Run(tc.Name, func(t *testing.T) {
			u := httpSrv.URL + tc.Path
			req, err := http.NewRequestWithContext(context.TODO(), tc.Method, u, nil)
			if err != nil {
				t.Fatalf("failed to create request: %v", err)
			}

			if tc.Setup != nil {
				tc.Setup(t, req)
			}

			resp, err := httpSrv.Client().Do(req)
			if err != nil {
				t.Fatalf("failed to do request: %v", err)
			}
			defer resp.Body.Close()

			if tc.Expected != nil {
				tc.Expected(t, resp)
			}
		})
	}
}

func TestCase(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	cases := []string{
		"mistral",
		"llama3:latest",
		"library/phi3:q4_0",
		"registry.ollama.ai/library/gemma:q5_K_M",
		// TODO: host:port currently fails on windows (#4107)
		// "localhost:5000/alice/bob:latest",
	}

	var s Server
	for _, tt := range cases {
		t.Run(tt, func(t *testing.T) {
			w := createRequest(t, s.CreateHandler, api.CreateRequest{
				Name:      tt,
				Modelfile: fmt.Sprintf("FROM %s", createBinFile(t, nil, nil)),
				Stream:    &stream,
			})

			if w.Code != http.StatusOK {
				t.Fatalf("expected status 200 got %d", w.Code)
			}

			expect, err := json.Marshal(map[string]string{"error": "a model with that name already exists"})
			if err != nil {
				t.Fatal(err)
			}

			t.Run("create", func(t *testing.T) {
				w = createRequest(t, s.CreateHandler, api.CreateRequest{
					Name:      strings.ToUpper(tt),
					Modelfile: fmt.Sprintf("FROM %s", createBinFile(t, nil, nil)),
					Stream:    &stream,
				})

				if w.Code != http.StatusBadRequest {
					t.Fatalf("expected status 500 got %d", w.Code)
				}

				if !bytes.Equal(w.Body.Bytes(), expect) {
					t.Fatalf("expected error %s got %s", expect, w.Body.String())
				}
			})

			t.Run("pull", func(t *testing.T) {
				w := createRequest(t, s.PullHandler, api.PullRequest{
					Name:   strings.ToUpper(tt),
					Stream: &stream,
				})

				if w.Code != http.StatusBadRequest {
					t.Fatalf("expected status 500 got %d", w.Code)
				}

				if !bytes.Equal(w.Body.Bytes(), expect) {
					t.Fatalf("expected error %s got %s", expect, w.Body.String())
				}
			})

			t.Run("copy", func(t *testing.T) {
				w := createRequest(t, s.CopyHandler, api.CopyRequest{
					Source:      tt,
					Destination: strings.ToUpper(tt),
				})

				if w.Code != http.StatusBadRequest {
					t.Fatalf("expected status 500 got %d", w.Code)
				}

				if !bytes.Equal(w.Body.Bytes(), expect) {
					t.Fatalf("expected error %s got %s", expect, w.Body.String())
				}
			})
		})
	}
}

func TestShow(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	var s Server

	createRequest(t, s.CreateHandler, api.CreateRequest{
		Name: "show-model",
		Modelfile: fmt.Sprintf(
			"FROM %s\nFROM %s",
			createBinFile(t, llm.KV{"general.architecture": "test"}, nil),
			createBinFile(t, llm.KV{"general.architecture": "clip"}, nil),
		),
	})

	w := createRequest(t, s.ShowHandler, api.ShowRequest{
		Name: "show-model",
	})

	if w.Code != http.StatusOK {
		t.Fatalf("expected status code 200, actual %d", w.Code)
	}

	var resp api.ShowResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatal(err)
	}

	if resp.ModelInfo["general.architecture"] != "test" {
		t.Fatal("Expected model architecture to be 'test', but got", resp.ModelInfo["general.architecture"])
	}

	if resp.ProjectorInfo["general.architecture"] != "clip" {
		t.Fatal("Expected projector architecture to be 'clip', but got", resp.ProjectorInfo["general.architecture"])
	}
}

func TestNormalize(t *testing.T) {
	type testCase struct {
		input []float32
	}

	testCases := []testCase{
		{input: []float32{1}},
		{input: []float32{0, 1, 2, 3}},
		{input: []float32{0.1, 0.2, 0.3}},
		{input: []float32{-0.1, 0.2, 0.3, -0.4}},
		{input: []float32{0, 0, 0}},
	}

	isNormalized := func(vec []float32) (res bool) {
		sum := 0.0
		for _, v := range vec {
			sum += float64(v * v)
		}
		if math.Abs(sum-1) > 1e-6 {
			return sum == 0
		} else {
			return true
		}
	}

	for _, tc := range testCases {
		t.Run("", func(t *testing.T) {
			normalized := normalize(tc.input)
			if !isNormalized(normalized) {
				t.Errorf("Vector %v is not normalized", tc.input)
			}
		})
	}
}
