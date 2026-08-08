package server

import (
	"net/http"
	"net/http/httptest"
	"testing"

	internalcloud "github.com/ollama/ollama/internal/cloud"
)

func TestUsageHandlerCloudDisabled(t *testing.T) {
	t.Setenv("OLLAMA_NO_CLOUD", "1")

	server := &Server{}
	router, err := server.GenerateRoutes()
	if err != nil {
		t.Fatal(err)
	}

	request := httptest.NewRequest(http.MethodGet, "/api/usage", nil)
	response := httptest.NewRecorder()
	router.ServeHTTP(response, request)

	if response.Code != http.StatusForbidden {
		t.Fatalf("status = %d, want %d", response.Code, http.StatusForbidden)
	}
	if got, want := response.Body.String(), `{"error":"`+internalcloud.DisabledError(cloudErrUsageUnavailable)+`"}`; got != want {
		t.Fatalf("body = %q, want %q", got, want)
	}
}
