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

func (c *Client) stream(ctx context.Context, method string, path string, reqData any, fn func(bts []byte) error) error {
	var reqBody io.Reader
	var data []byte
	var err error
	if reqData != nil {
		data, err = json.Marshal(reqData)
		if err != nil {
			return err
		}
		reqBody = bytes.NewReader(data)
	}

	request, err := http.NewRequestWithContext(ctx, method, c.base.JoinPath(path).String(), reqBody)
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
		if err := fn(scanner.Bytes()); err != nil {
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
