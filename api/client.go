package api

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/url"
)

type Client struct {
	base url.URL
}

func NewClient(hosts ...string) *Client {
	host := "127.0.0.1:11434"
	if len(hosts) > 0 {
		host = hosts[0]
	}

	return &Client{
		base: url.URL{Scheme: "http", Host: host},
	}
}

type options struct {
	requestBody  io.Reader
	responseFunc func(bts []byte) error
}

func OptionRequestBody(data any) func(*options) {
	bts, err := json.Marshal(data)
	if err != nil {
		panic(err)
	}

	return func(opts *options) {
		opts.requestBody = bytes.NewReader(bts)
	}
}

func OptionResponseFunc(fn func([]byte) error) func(*options) {
	return func(opts *options) {
		opts.responseFunc = fn
	}
}

func (c *Client) stream(ctx context.Context, method, path string, fns ...func(*options)) error {
	var opts options
	for _, fn := range fns {
		fn(&opts)
	}

	request, err := http.NewRequestWithContext(ctx, method, c.base.JoinPath(path).String(), opts.requestBody)
	if err != nil {
		return err
	}

	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Accept", "application/json")

	response, err := http.DefaultClient.Do(request)
	if err != nil {
		return err
	}
	defer response.Body.Close()

	if opts.responseFunc != nil {
		scanner := bufio.NewScanner(response.Body)
		for scanner.Scan() {
			if err := opts.responseFunc(scanner.Bytes()); err != nil {
				return err
			}
		}
	}

	return nil
}

type GenerateResponseFunc func(GenerateResponse) error

func (c *Client) Generate(ctx context.Context, req *GenerateRequest, fn GenerateResponseFunc) error {
	return c.stream(ctx, http.MethodPost, "/api/generate",
		OptionRequestBody(req),
		OptionResponseFunc(func(bts []byte) error {
			var resp GenerateResponse
			if err := json.Unmarshal(bts, &resp); err != nil {
				return err
			}

			return fn(resp)
		}),
	)
}

type PullProgressFunc func(PullProgress) error

func (c *Client) Pull(ctx context.Context, req *PullRequest, fn PullProgressFunc) error {
	return c.stream(ctx, http.MethodPost, "/api/pull",
		OptionRequestBody(req),
		OptionResponseFunc(func(bts []byte) error {
			var resp PullProgress
			if err := json.Unmarshal(bts, &resp); err != nil {
				return err
			}

			if resp.Error.Message != "" {
				// couldn't pull the model from the directory, proceed anyway
				return nil
			}

			return fn(resp)
		}),
	)
}
