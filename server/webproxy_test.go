package server

import (
    "errors"
    "io"
    "net/http"
    "net/http/httptest"
    "strings"
    "testing"

    "github.com/gin-gonic/gin"
)

type rtStub struct {
    fn func(req *http.Request) *http.Response
}

func (r *rtStub) RoundTrip(req *http.Request) (*http.Response, error) {
    if r.fn == nil {
        return nil, errors.New("no stub")
    }
    return r.fn(req), nil
}

func TestProxyToMain_WithToken_SetsAuthorizationAndForwards(t *testing.T) {
    t.Parallel()

    // backup globals
    origSign := signFunc
    origClient := httpClient
    defer func() { signFunc = origSign; httpClient = origClient }()

    var captured *http.Request

    // stub signing
    signFunc = func(_ any, _ []byte) (string, error) { return "Bearer testtoken", nil }

    // stub transport
    stub := &rtStub{fn: func(req *http.Request) *http.Response {
        captured = req
        return &http.Response{
            StatusCode: 200,
            Body:       io.NopCloser(strings.NewReader("ok")),
            Header:     http.Header{"Content-Type": {"application/json"}},
        }
    }}
    httpClient = &http.Client{Transport: stub}

    gin.SetMode(gin.TestMode)
    w := httptest.NewRecorder()
    c, _ := gin.CreateTestContext(w)
    req := httptest.NewRequest(http.MethodPost, "/api/web_search", strings.NewReader(`{"q":"hi"}`))
    req.Header.Set("Content-Type", "application/json")
    c.Request = req

    s := &Server{}
    s.WebSearchHandler(c)

    if w.Code != 200 {
        t.Fatalf("expected 200 status, got %d", w.Code)
    }
    if strings.TrimSpace(w.Body.String()) != "ok" {
        t.Fatalf("unexpected body: %q", w.Body.String())
    }
    if captured == nil {
        t.Fatal("no outbound request captured")
    }
    if got := captured.Header.Get("Authorization"); got != "Bearer testtoken" {
        t.Fatalf("expected Authorization header set, got %q", got)
    }
    if captured.URL.Path != "/api/web_search" {
        t.Fatalf("expected path /api/web_search, got %s", captured.URL.Path)
    }
}

func TestProxyToMain_SignFails_Returns500(t *testing.T) {
    t.Parallel()

    origSign := signFunc
    defer func() { signFunc = origSign }()

    signFunc = func(_ any, _ []byte) (string, error) { return "", errors.New("no key") }

    gin.SetMode(gin.TestMode)
    w := httptest.NewRecorder()
    c, _ := gin.CreateTestContext(w)
    req := httptest.NewRequest(http.MethodPost, "/api/web_fetch", strings.NewReader(`{"url":"https://example.com"}`))
    req.Header.Set("Content-Type", "application/json")
    c.Request = req

    s := &Server{}
    s.WebFetchHandler(c)

    if w.Code != http.StatusInternalServerError {
        t.Fatalf("expected 500 status, got %d", w.Code)
    }
}
