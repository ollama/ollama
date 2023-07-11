package api

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
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

func (c *Client) stream(ctx context.Context, method, path string, data any, callback func([]byte) error) error {
	var buf *bytes.Buffer
	if data != nil {
		bts, err := json.Marshal(data)
		if err != nil {
			return err
		}

		buf = bytes.NewBuffer(bts)
	}

	request, err := http.NewRequestWithContext(ctx, method, c.base.JoinPath(path).String(), buf)
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

	scanner := bufio.NewScanner(response.Body)
	for scanner.Scan() {
		var errorResponse struct {
			Error string `json:"error"`
		}

		bts := scanner.Bytes()
		if err := json.Unmarshal(bts, &errorResponse); err != nil {
			return fmt.Errorf("unmarshal: %w", err)
		}

		if len(errorResponse.Error) > 0 {
			return fmt.Errorf("stream: %s", errorResponse.Error)
		}

		if err := callback(bts); err != nil {
			return err
		}
	}

	return nil
}

type GenerateResponseFunc func(GenerateResponse) error

func (c *Client) Generate(ctx context.Context, req *GenerateRequest, fn GenerateResponseFunc) error {
	return c.stream(ctx, http.MethodPost, "/api/generate", req, func(bts []byte) error {
		var resp GenerateResponse
		if err := json.Unmarshal(bts, &resp); err != nil {
			return err
		}

		return fn(resp)
	})
}

type PullProgressFunc func(PullProgress) error

func (c *Client) Pull(ctx context.Context, req *PullRequest, fn PullProgressFunc) error {
	return c.stream(ctx, http.MethodPost, "/api/pull", req, func(bts []byte) error {
		var resp PullProgress
		if err := json.Unmarshal(bts, &resp); err != nil {
			return err
		}

		return fn(resp)
	})
}
