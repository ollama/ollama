package server

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"gotest.tools/v3/assert"
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
				assert.NilError(t, err)
				assert.Equal(t, `{"version":"0.0.0"}`, string(body))
			},
		},
	}

	s, err := setupServer(t)
	assert.NilError(t, err)

	router := s.GenerateRoutes()

	httpSrv := httptest.NewServer(router)
	t.Cleanup(httpSrv.Close)

	run := func(t *testing.T, tc testCase) {
		u := httpSrv.URL + tc.Path
		req, err := http.NewRequestWithContext(context.TODO(), tc.Method, u, nil)
		assert.NilError(t, err)

		if tc.Setup != nil {
			tc.Setup(t, req)
		}

		resp, err := httpSrv.Client().Do(req)
		assert.NilError(t, err)

		if tc.Expected != nil {
			tc.Expected(t, resp)
		}
	}

	for _, tc := range testCases {
		t.Run(tc.Name, func(t *testing.T) {
			run(t, tc)
		})
	}

}
