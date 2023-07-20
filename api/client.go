package api

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
)

type Client struct {
	base    url.URL
	HTTP    http.Client
	Headers http.Header
}

func checkError(resp *http.Response, body []byte) error {
	if resp.StatusCode >= 200 && resp.StatusCode < 400 {
		return nil
	}

	apiError := StatusError{StatusCode: resp.StatusCode}

	err := json.Unmarshal(body, &apiError)
	if err != nil {
		// Use the full body as the message if we fail to decode a response.
		apiError.ErrorMessage = string(body)
	}

	return apiError
}

func NewClient(hosts ...string) *Client {
	host := "127.0.0.1:11434"
	if len(hosts) > 0 {
		host = hosts[0]
	}

	return &Client{
		base: url.URL{Scheme: "http", Host: host},
		HTTP: http.Client{},
	}
}

func (c *Client) do(ctx context.Context, method, path string, reqData, respData any) error {
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

	url := c.base.JoinPath(path).String()

	req, err := http.NewRequestWithContext(ctx, method, url, reqBody)
	if err != nil {
		return err
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")

	for k, v := range c.Headers {
		req.Header[k] = v
	}

	respObj, err := c.HTTP.Do(req)
	if err != nil {
		return err
	}
	defer respObj.Body.Close()

	respBody, err := io.ReadAll(respObj.Body)
	if err != nil {
		return err
	}

	if err := checkError(respObj, respBody); err != nil {
		return err
	}

	if len(respBody) > 0 && respData != nil {
		if err := json.Unmarshal(respBody, respData); err != nil {
			return err
		}
	}
	return nil
}

func (c *Client) stream(ctx context.Context, method, path string, data any, fn func([]byte) error) error {
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
			Error string `json:"error,omitempty"`
		}

		bts := scanner.Bytes()
		if err := json.Unmarshal(bts, &errorResponse); err != nil {
			return fmt.Errorf("unmarshal: %w", err)
		}

		if response.StatusCode >= 400 {
			return StatusError{
				StatusCode:   response.StatusCode,
				Status:       response.Status,
				ErrorMessage: errorResponse.Error,
			}
		}

		if err := fn(bts); err != nil {
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

type PullProgressFunc func(ProgressResponse) error

func (c *Client) Pull(ctx context.Context, req *PullRequest, fn PullProgressFunc) error {
	return c.stream(ctx, http.MethodPost, "/api/pull", req, func(bts []byte) error {
		var resp ProgressResponse
		if err := json.Unmarshal(bts, &resp); err != nil {
			return err
		}

		return fn(resp)
	})
}

type PushProgressFunc func(ProgressResponse) error

func (c *Client) Push(ctx context.Context, req *PushRequest, fn PushProgressFunc) error {
	return c.stream(ctx, http.MethodPost, "/api/push", req, func(bts []byte) error {
		var resp ProgressResponse
		if err := json.Unmarshal(bts, &resp); err != nil {
			return err
		}

		return fn(resp)
	})
}

type CreateProgressFunc func(CreateProgress) error

func (c *Client) Create(ctx context.Context, req *CreateRequest, fn CreateProgressFunc) error {
	return c.stream(ctx, http.MethodPost, "/api/create", req, func(bts []byte) error {
		var resp CreateProgress
		if err := json.Unmarshal(bts, &resp); err != nil {
			return err
		}

		return fn(resp)
	})
}

func (c *Client) List(ctx context.Context) (*ListResponse, error) {
	var lr ListResponse
	if err := c.do(ctx, http.MethodGet, "/api/tags", nil, &lr); err != nil {
		return nil, err
	}
	return &lr, nil
}
