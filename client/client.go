package client

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"maps"
	"net/http"
	"net/url"
	"runtime"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/version"
)

type Error struct {
	Status     string
	StatusCode int
}

func (e Error) Error() string {
	return e.Status
}

type Client struct {
	baseURL *url.URL
	header  http.Header
}

// WithBaseURL sets the base URL. It panics if it is invalid.
func WithBaseURL(s string) func(*Client) {
	return func(c *Client) {
		parsed, err := url.Parse(s)
		if err != nil {
			panic(err)
		}
		c.baseURL = parsed
	}
}

// WithHeader sets custom HTTP headers.
func WithHeader(h http.Header) func(*Client) {
	return func(c *Client) {
		c.header = h
	}
}

func New(opts ...func(*Client) error) *Client {
	c := Client{
		baseURL: envconfig.Host(),
		header: http.Header{
			"Content-Type": {"application/json"},
			"User-Agent":   {userAgent},
		},
	}
	for _, opt := range opts {
		opt(&c)
	}
	return &c
}

func (c *Client) Ping(ctx context.Context) error {
	_, err := do[struct{}](c, ctx, http.MethodHead, "/", nil)
	return err
}

func (c *Client) Chat(ctx context.Context, r api.ChatRequest) (iter.Seq2[api.ChatResponse, error], error) {
	return doSeq[api.ChatResponse](c, ctx, http.MethodPost, "/api/chat", r)
}

// Pull downloads a model from a remote repository to the Ollama server.
func (c *Client) Pull(ctx context.Context, r api.PullRequest) (iter.Seq2[api.ProgressResponse, error], error) {
	return doSeq[api.ProgressResponse](c, ctx, http.MethodPost, "/api/pull", r)
}

// Push uploads a model from the Ollama server to a remote repository.
func (c *Client) Push(ctx context.Context, r api.PushRequest) (iter.Seq2[api.ProgressResponse, error], error) {
	return doSeq[api.ProgressResponse](c, ctx, http.MethodPost, "/api/push", r)
}

// Create builds a new model on the Ollama server.
func (c *Client) Create(ctx context.Context, r api.CreateRequest) (iter.Seq2[api.ProgressResponse, error], error) {
	return doSeq[api.ProgressResponse](c, ctx, http.MethodPost, "/api/create", r)
}

// List returns the list of models from the Ollama server.
func (c *Client) List(ctx context.Context) (api.ListResponse, error) {
	return do[api.ListResponse](c, ctx, http.MethodGet, "/api/tags", nil)
}

// Delete removes a model from the Ollama server.
func (c *Client) Delete(ctx context.Context, r api.DeleteRequest) error {
	_, err := do[struct{}](c, ctx, http.MethodDelete, "/api/delete", r)
	return err
}

// Version returns the Ollama server version.
func (c *Client) Version(ctx context.Context) (string, error) {
	type versionResponse struct {
		Version string `json:"version"`
	}
	version, err := do[versionResponse](c, ctx, "GET", "/api/version", nil)
	if err != nil {
		return "", err
	}
	return version.Version, nil
}

var userAgent = fmt.Sprintf("ollama/%s (%s %s) Go/%s", version.Version, runtime.GOARCH, runtime.GOOS, runtime.Version())

// do sends the specified HTTP request and returns the raw HTTP response. header are merged with the client's default headers.
func (c *Client) do(ctx context.Context, method, path string, body any, header http.Header) (*http.Response, error) {
	var b bytes.Buffer
	if err := json.NewEncoder(&b).Encode(body); err != nil {
		return nil, err
	}

	r, err := http.NewRequestWithContext(ctx, method, c.baseURL.JoinPath(path).String(), &b)
	if err != nil {
		return nil, err
	}

	// copy headers into the request in order. later headers override earlier ones.
	for _, header := range []http.Header{c.header, header} {
		maps.Copy(r.Header, header)
	}

	w, err := http.DefaultClient.Do(r)
	if err != nil {
		return nil, err
	}

	if w.StatusCode >= 400 {
		return nil, Error{
			Status:     w.Status,
			StatusCode: w.StatusCode,
		}
	}

	return w, nil
}

// do sends the specified HTTP request and returns the JSON response as type T
func do[T any](c *Client, ctx context.Context, method, path string, body any) (t T, err error) {
	w, err := c.do(ctx, method, path, body, http.Header{"Accept": {"application/json"}})
	if err != nil {
		return t, err
	}
	defer w.Body.Close()

	if w.ContentLength > 0 && method != http.MethodHead {
		if err := json.NewDecoder(w.Body).Decode(&t); err != nil {
			return t, err
		}
	}

	return t, nil
}

// doSeq sends the specified HTTP request and returns an iterator that yields the JSON response chunks as type T
func doSeq[T any](c *Client, ctx context.Context, method, path string, body any) (iter.Seq2[T, error], error) {
	w, err := c.do(ctx, method, path, body, http.Header{"Accept": {"application/jsonl", "application/x-ndjson"}})
	if err != nil {
		return nil, err
	}

	return func(yield func(T, error) bool) {
		defer w.Body.Close()

		bts := make([]byte, 0, 512<<10)
		s := bufio.NewScanner(w.Body)
		s.Buffer(bts, len(bts))
		for s.Scan() {
			var t T
			if err := json.Unmarshal(s.Bytes(), &t); err != nil {
				yield(t, err)
				break
			}

			if !yield(t, nil) {
				break
			}
		}
	}, nil
}
