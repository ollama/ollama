package server

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/jmorganca/ollama/api"
	"github.com/jmorganca/ollama/parser"
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
				assert.Equal(t, `{"version":"0.0.0"}`, string(body))
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
				f, err := os.CreateTemp("", "ollama-modelfile")
				assert.Nil(t, err)
				defer os.RemoveAll(f.Name())

				modelfile := strings.NewReader(fmt.Sprintf("FROM %s", f.Name()))
				commands, err := parser.Parse(modelfile)
				assert.Nil(t, err)
				fn := func(resp api.ProgressResponse) {}
				err = CreateModel(context.TODO(), "test", "", commands, fn)
				assert.Nil(t, err)
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
				assert.Equal(t, modelList.Models[0].Name, "test:latest")
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
		u := httpSrv.URL + tc.Path
		req, err := http.NewRequestWithContext(context.TODO(), tc.Method, u, nil)
		assert.Nil(t, err)

		if tc.Setup != nil {
			tc.Setup(t, req)
		}

		resp, err := httpSrv.Client().Do(req)
		assert.Nil(t, err)

		if tc.Expected != nil {
			tc.Expected(t, resp)
		}
	}

}
