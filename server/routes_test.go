package server

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"math"
	"math/rand/v2"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"unicode"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/openai"
	"github.com/ollama/ollama/server/internal/cache/blob"
	"github.com/ollama/ollama/server/internal/client/ollama"
	"github.com/ollama/ollama/types/model"
	"github.com/ollama/ollama/version"
)

func createTestFile(t *testing.T, name string) (string, string) {
	t.Helper()

	modelDir := os.Getenv("OLLAMA_MODELS")
	if modelDir == "" {
		t.Fatalf("OLLAMA_MODELS not specified")
	}

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

	// Calculate sha256 sum of file
	if _, err := f.Seek(0, 0); err != nil {
		t.Fatal(err)
	}

	digest, _ := GetSHA256Digest(f)
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}

	if err := createLink(f.Name(), filepath.Join(modelDir, "blobs", fmt.Sprintf("sha256-%s", strings.TrimPrefix(digest, "sha256:")))); err != nil {
		t.Fatal(err)
	}

	return f.Name(), digest
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

type panicTransport struct{}

func (t *panicTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	panic("unexpected RoundTrip call")
}

var panicOnRoundTrip = &http.Client{Transport: &panicTransport{}}

func TestRoutes(t *testing.T) {
	type testCase struct {
		Name     string
		Method   string
		Path     string
		Setup    func(t *testing.T, req *http.Request)
		Expected func(t *testing.T, resp *http.Response)
	}

	createTestModel := func(t *testing.T, name string) {
		t.Helper()

		_, digest := createTestFile(t, "ollama-model")

		fn := func(resp api.ProgressResponse) {
			t.Logf("Status: %s", resp.Status)
		}

		r := api.CreateRequest{
			Name:  name,
			Files: map[string]string{"test.gguf": digest},
			Parameters: map[string]any{
				"seed":  42,
				"top_p": 0.9,
				"stop":  []string{"foo", "bar"},
			},
		}

		modelName := model.ParseName(name)

		baseLayers, err := ggufLayers(digest, fn)
		if err != nil {
			t.Fatalf("failed to create model: %v", err)
		}

		if err := createModel(r, modelName, baseLayers, fn); err != nil {
			t.Fatal(err)
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
				createTestModel(t, "model_to_delete")

				deleteReq := api.DeleteRequest{
					Name: "model_to_delete",
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
					Name: "non_existent_model",
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
				_, digest := createTestFile(t, "ollama-model")
				stream := false
				createReq := api.CreateRequest{
					Name:   "t-bone",
					Files:  map[string]string{"test.gguf": digest},
					Stream: &stream,
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
			Name: "openai retrieve model handler",
			Setup: func(t *testing.T, req *http.Request) {
				createTestModel(t, "show-model")
			},
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

	modelsDir := t.TempDir()
	t.Setenv("OLLAMA_MODELS", modelsDir)

	c, err := blob.Open(modelsDir)
	if err != nil {
		t.Fatalf("failed to open models dir: %v", err)
	}

	rc := &ollama.Registry{
		// This is a temporary measure to allow us to move forward,
		// surfacing any code contacting ollama.com we do not intended
		// to.
		//
		// Currently, this only handles DELETE /api/delete, which
		// should not make any contact with the ollama.com registry, so
		// be clear about that.
		//
		// Tests that do need to contact the registry here, will be
		// consumed into our new server/api code packages and removed
		// from here.
		HTTPClient: panicOnRoundTrip,
	}

	s := &Server{}
	router, err := s.GenerateRoutes(c, rc)
	if err != nil {
		t.Fatalf("failed to generate routes: %v", err)
	}

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

func casingShuffle(s string) string {
	rr := []rune(s)
	for i := range rr {
		if rand.N(2) == 0 {
			rr[i] = unicode.ToUpper(rr[i])
		} else {
			rr[i] = unicode.ToLower(rr[i])
		}
	}
	return string(rr)
}

func TestManifestCaseSensitivity(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	r := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		io.WriteString(w, `{}`) //nolint:errcheck
	}))
	defer r.Close()

	nameUsed := make(map[string]bool)
	name := func() string {
		const fqmn = "example/namespace/model:tag"
		for {
			v := casingShuffle(fqmn)
			if nameUsed[v] {
				continue
			}
			nameUsed[v] = true
			return v
		}
	}

	wantStableName := name()

	t.Logf("stable name: %s", wantStableName)

	// checkManifestList tests that there is strictly one manifest in the
	// models directory, and that the manifest is for the model under test.
	checkManifestList := func() {
		t.Helper()

		mandir := filepath.Join(os.Getenv("OLLAMA_MODELS"), "manifests/")
		var entries []string
		t.Logf("dir entries:")
		fsys := os.DirFS(mandir)
		err := fs.WalkDir(fsys, ".", func(path string, info fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			t.Logf("    %s", fs.FormatDirEntry(info))
			if info.IsDir() {
				return nil
			}
			path = strings.TrimPrefix(path, mandir)
			entries = append(entries, path)
			return nil
		})
		if err != nil {
			t.Fatalf("failed to walk directory: %v", err)
		}

		if len(entries) != 1 {
			t.Errorf("len(got) = %d, want 1", len(entries))
			return // do not use Fatal so following steps run
		}

		g := entries[0] // raw path
		g = filepath.ToSlash(g)
		w := model.ParseName(wantStableName).Filepath()
		w = filepath.ToSlash(w)
		if g != w {
			t.Errorf("\ngot:  %s\nwant: %s", g, w)
		}
	}

	checkOK := func(w *httptest.ResponseRecorder) {
		t.Helper()
		if w.Code != http.StatusOK {
			t.Errorf("code = %d, want 200", w.Code)
			t.Logf("body: %s", w.Body.String())
		}
	}

	var s Server
	testMakeRequestDialContext = func(ctx context.Context, _, _ string) (net.Conn, error) {
		var d net.Dialer
		return d.DialContext(ctx, "tcp", r.Listener.Addr().String())
	}
	t.Cleanup(func() { testMakeRequestDialContext = nil })

	t.Logf("creating")
	_, digest := createBinFile(t, nil, nil)
	checkOK(createRequest(t, s.CreateHandler, api.CreateRequest{
		// Start with the stable name, and later use a case-shuffled
		// version.
		Name:   wantStableName,
		Files:  map[string]string{"test.gguf": digest},
		Stream: &stream,
	}))
	checkManifestList()

	t.Logf("creating (again)")
	checkOK(createRequest(t, s.CreateHandler, api.CreateRequest{
		Name:   name(),
		Files:  map[string]string{"test.gguf": digest},
		Stream: &stream,
	}))
	checkManifestList()

	t.Logf("pulling")
	checkOK(createRequest(t, s.PullHandler, api.PullRequest{
		Name:     name(),
		Stream:   &stream,
		Insecure: true,
	}))
	checkManifestList()

	t.Logf("copying")
	checkOK(createRequest(t, s.CopyHandler, api.CopyRequest{
		Source:      name(),
		Destination: name(),
	}))
	checkManifestList()

	t.Logf("pushing")
	rr := createRequest(t, s.PushHandler, api.PushRequest{
		Model:    name(),
		Insecure: true,
		Username: "alice",
		Password: "x",
	})
	checkOK(rr)
	if !strings.Contains(rr.Body.String(), `"status":"success"`) {
		t.Errorf("got = %q, want success", rr.Body.String())
	}
}

func TestShow(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	var s Server

	_, digest1 := createBinFile(t, ggml.KV{"general.architecture": "test"}, nil)
	_, digest2 := createBinFile(t, ggml.KV{"general.type": "projector", "general.architecture": "clip"}, nil)

	createRequest(t, s.CreateHandler, api.CreateRequest{
		Name:  "show-model",
		Files: map[string]string{"model.gguf": digest1, "projector.gguf": digest2},
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
