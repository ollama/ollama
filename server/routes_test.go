package server

import (
	"bytes"
	"context"
	"encoding/binary"
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

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/parser"
	"github.com/ollama/ollama/version"
)

func Test_Routes(t *testing.T) {
	type testCase struct {
		Name     string
		Method   string
		Path     string
		Setup    func(t *testing.T, req *http.Request)
		Expected func(t *testing.T, resp *http.Response)
	}

	createTestFile := func(t *testing.T, name string) string {
		t.Helper()

		f, err := os.CreateTemp(t.TempDir(), name)
		assert.Nil(t, err)
		defer f.Close()

		err = binary.Write(f, binary.LittleEndian, []byte("GGUF"))
		assert.Nil(t, err)

		err = binary.Write(f, binary.LittleEndian, uint32(3))
		assert.Nil(t, err)

		err = binary.Write(f, binary.LittleEndian, uint64(0))
		assert.Nil(t, err)

		err = binary.Write(f, binary.LittleEndian, uint64(0))
		assert.Nil(t, err)

		return f.Name()
	}

	createTestModel := func(t *testing.T, name string) {
		fname := createTestFile(t, "ollama-model")

		r := strings.NewReader(fmt.Sprintf("FROM %s\nPARAMETER seed 42\nPARAMETER top_p 0.9\nPARAMETER stop foo\nPARAMETER stop bar", fname))
		modelfile, err := parser.ParseFile(r)
		assert.Nil(t, err)
		fn := func(resp api.ProgressResponse) {
			t.Logf("Status: %s", resp.Status)
		}
		err = CreateModel(context.TODO(), name, "", "", modelfile, fn)
		assert.Nil(t, err)
	}

	// Test Model Digests
	blobs := []string{
		"sha256:a4e5e156ddec27e286f75328784d7106b60a4eb1d246e950a001a3f944fbda99",
		"sha256:4f9d252f34ae677363956ffc6dd2d10918a539c5c91f5ee2fe889d9178be6ae3",
		"sha256:0f239b83e9e2aad7cd997a5bb44124937a32ac1f4e98e95a2f46e7b966bfc878",
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

				assert.NotNil(t, modelList.Models)
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
				fname := createTestFile(t, "ollama-model")

				stream := false
				createReq := api.CreateRequest{
					Name:      "t-bone",
					Modelfile: fmt.Sprintf("FROM %s", fname),
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
		{
			Name:   "Delete Handler (multiple blob reference)",
			Method: http.MethodDelete,
			Path:   "/api/delete",
			Setup: func(t *testing.T, req *http.Request) {
				err := DeleteModel("test-model")
				assert.Nil(t, err)
				err = DeleteModel("hamshank")
				assert.Nil(t, err)
				deleteReq := api.DeleteRequest{Model: "beefsteak"}
				jsonData, err := json.Marshal(deleteReq)
				assert.Nil(t, err)
				req.Body = io.NopCloser(bytes.NewReader(jsonData))
			},
			Expected: func(t *testing.T, resp *http.Response) {
				_, err := io.ReadAll(resp.Body)
				assert.Nil(t, err)
				assert.Equal(t, resp.StatusCode, 200)

				_, err = GetModel("beefsteak")
				assert.True(t, os.IsNotExist(err))

				model, _ := GetModel("show-model")
				assert.Equal(t, "show-model:latest", model.ShortName)

				for _, blob := range blobs {
					blob, _ = GetBlobsPath(blob)
					_, err := os.Stat(blob)
					assert.False(t, os.IsNotExist(err))
				}
			},
		},
		{
			Name:   "Delete Handler (single blob reference)",
			Method: http.MethodDelete,
			Path:   "/api/delete",
			Setup: func(t *testing.T, req *http.Request) {
				deleteReq := api.DeleteRequest{Model: "show-model"}
				jsonData, err := json.Marshal(deleteReq)
				assert.Nil(t, err)
				req.Body = io.NopCloser(bytes.NewReader(jsonData))
			},
			Expected: func(t *testing.T, resp *http.Response) {
				_, err := io.ReadAll(resp.Body)
				assert.Nil(t, err)

				_, err = GetModel("show-model")
				assert.True(t, os.IsNotExist(err))

				for _, blob := range blobs {
					_, err := os.Stat(blob)
					assert.True(t, os.IsNotExist(err))
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
		})
	}
}
