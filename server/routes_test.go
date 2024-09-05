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

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

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
	require.NoError(t, err)
	defer f.Close()

	err = binary.Write(f, binary.LittleEndian, []byte("GGUF"))
	require.NoError(t, err)

	err = binary.Write(f, binary.LittleEndian, uint32(3))
	require.NoError(t, err)

	err = binary.Write(f, binary.LittleEndian, uint64(0))
	require.NoError(t, err)

	err = binary.Write(f, binary.LittleEndian, uint64(0))
	require.NoError(t, err)

	return f.Name()
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
		require.NoError(t, err)
		fn := func(resp api.ProgressResponse) {
			t.Logf("Status: %s", resp.Status)
		}
		err = CreateModel(context.TODO(), model.ParseName(name), "", "", modelfile, fn)
		require.NoError(t, err)
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
				assert.Equal(t, "application/json; charset=utf-8", contentType)
				body, err := io.ReadAll(resp.Body)
				require.NoError(t, err)
				assert.Equal(t, fmt.Sprintf(`{"version":"%s"}`, version.Version), string(body))
			},
		},
		{
			Name:   "Tags Handler (no tags)",
			Method: http.MethodGet,
			Path:   "/api/tags",
			Expected: func(t *testing.T, resp *http.Response) {
				contentType := resp.Header.Get("Content-Type")
				assert.Equal(t, "application/json; charset=utf-8", contentType)
				body, err := io.ReadAll(resp.Body)
				require.NoError(t, err)

				var modelList api.ListResponse

				err = json.Unmarshal(body, &modelList)
				require.NoError(t, err)

				assert.NotNil(t, modelList.Models)
				assert.Empty(t, len(modelList.Models))
			},
		},
		{
			Name:   "openai empty list",
			Method: http.MethodGet,
			Path:   "/v1/models",
			Expected: func(t *testing.T, resp *http.Response) {
				contentType := resp.Header.Get("Content-Type")
				assert.Equal(t, "application/json", contentType)
				body, err := io.ReadAll(resp.Body)
				require.NoError(t, err)

				var modelList openai.ListCompletion
				err = json.Unmarshal(body, &modelList)
				require.NoError(t, err)

				assert.Equal(t, "list", modelList.Object)
				assert.Empty(t, modelList.Data)
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
				assert.Equal(t, "application/json; charset=utf-8", contentType)
				body, err := io.ReadAll(resp.Body)
				require.NoError(t, err)

				assert.NotContains(t, string(body), "expires_at")

				var modelList api.ListResponse
				err = json.Unmarshal(body, &modelList)
				require.NoError(t, err)

				assert.Len(t, modelList.Models, 1)
				assert.Equal(t, "test-model:latest", modelList.Models[0].Name)
			},
		},
		{
			Name:   "openai list models with tags",
			Method: http.MethodGet,
			Path:   "/v1/models",
			Expected: func(t *testing.T, resp *http.Response) {
				contentType := resp.Header.Get("Content-Type")
				assert.Equal(t, "application/json", contentType)
				body, err := io.ReadAll(resp.Body)
				require.NoError(t, err)

				var modelList openai.ListCompletion
				err = json.Unmarshal(body, &modelList)
				require.NoError(t, err)

				assert.Len(t, modelList.Data, 1)
				assert.Equal(t, "test-model:latest", modelList.Data[0].Id)
				assert.Equal(t, "library", modelList.Data[0].OwnedBy)
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
				require.NoError(t, err)

				req.Body = io.NopCloser(bytes.NewReader(jsonData))
			},
			Expected: func(t *testing.T, resp *http.Response) {
				contentType := resp.Header.Get("Content-Type")
				assert.Equal(t, "application/json", contentType)
				_, err := io.ReadAll(resp.Body)
				require.NoError(t, err)
				assert.Equal(t, 200, resp.StatusCode)

				model, err := GetModel("t-bone")
				require.NoError(t, err)
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
				require.NoError(t, err)

				req.Body = io.NopCloser(bytes.NewReader(jsonData))
			},
			Expected: func(t *testing.T, resp *http.Response) {
				model, err := GetModel("beefsteak")
				require.NoError(t, err)
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
				require.NoError(t, err)
				req.Body = io.NopCloser(bytes.NewReader(jsonData))
			},
			Expected: func(t *testing.T, resp *http.Response) {
				contentType := resp.Header.Get("Content-Type")
				assert.Equal(t, "application/json; charset=utf-8", contentType)
				body, err := io.ReadAll(resp.Body)
				require.NoError(t, err)

				var showResp api.ShowResponse
				err = json.Unmarshal(body, &showResp)
				require.NoError(t, err)

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
				assert.InDelta(t, 0, showResp.ModelInfo["general.parameter_count"], 1e-9, "Parameter count should be 0")
			},
		},
		{
			Name:   "openai retrieve model handler",
			Method: http.MethodGet,
			Path:   "/v1/models/show-model",
			Expected: func(t *testing.T, resp *http.Response) {
				contentType := resp.Header.Get("Content-Type")
				assert.Equal(t, "application/json", contentType)
				body, err := io.ReadAll(resp.Body)
				require.NoError(t, err)

				var retrieveResp api.RetrieveModelResponse
				err = json.Unmarshal(body, &retrieveResp)
				require.NoError(t, err)

				assert.Equal(t, "show-model", retrieveResp.Id)
				assert.Equal(t, "library", retrieveResp.OwnedBy)
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
			require.NoError(t, err)

			if tc.Setup != nil {
				tc.Setup(t, req)
			}

			resp, err := httpSrv.Client().Do(req)
			require.NoError(t, err)
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
