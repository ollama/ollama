package api

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

type Client struct {
	URL        string
	HTTP       http.Client
}

func checkError(resp *http.Response, body []byte) error {
	if resp.StatusCode >= 200 && resp.StatusCode < 400 {
		return nil
	}

	apiError := Error{Code: int32(resp.StatusCode)}

	if err := json.Unmarshal(body, &apiError); err != nil {
		// Use the full body as the message if we fail to decode a response.
		apiError.Message = string(body)
	}

	return apiError
}

func (c *Client) stream(ctx context.Context, method string, path string, reqData any, callback func(data []byte)) error {
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

	url := fmt.Sprintf("%s%s", c.URL, path)

	req, err := http.NewRequestWithContext(ctx, method, url, reqBody)
	if err != nil {
		return err
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")

	res, err := c.HTTP.Do(req)
	if err != nil {
		return err
	}
	defer res.Body.Close()

	reader := bufio.NewReader(res.Body)

	for {
		line, err := reader.ReadBytes('\n')
		if err != nil {
			break
		}
		callback(bytes.TrimSuffix(line, []byte("\n")))
	}

	return nil
}

func (c *Client) do(ctx context.Context, method string, path string, reqData any, respData any) error {
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

	url := fmt.Sprintf("%s%s", c.URL, path)

	req, err := http.NewRequestWithContext(ctx, method, url, reqBody)
	if err != nil {
		return err
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")

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

func (c *Client) Generate(ctx context.Context, req *GenerateRequest, callback func(token string)) (*GenerateResponse, error) {
	var res GenerateResponse
	if err := c.stream(ctx, http.MethodPost, "/api/generate", req, func(token []byte) {
		callback(string(token))
	}); err != nil {
		return nil, err
	}

	return &res, nil
}

func (c *Client) Pull(ctx context.Context, req *PullRequest, callback func(progress PullProgress)) (*PullResponse, error) {
	var res PullResponse
	if err := c.stream(ctx, http.MethodPost, "/api/pull", req, func(progressBytes []byte) {
		/*
			Events have the following format for progress:
				event:progress
				data:{"total":123,"completed":123,"percent":0.1}
			Need to parse out the data part and unmarshal it.
		*/
		eventParts := strings.Split(string(progressBytes), "data:")
		if len(eventParts) < 2 {
			// no data part, ignore
			return
		}
		eventData := eventParts[1]
		var progress PullProgress
		if err := json.Unmarshal([]byte(eventData), &progress); err != nil {
			fmt.Println(err)
			return
		}
		callback(progress)
	}); err != nil {
		return nil, err
	}

	return &res, nil
}
