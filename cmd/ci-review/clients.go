package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
)

type cloudChatClient struct {
	apiKey   string
	http     *http.Client
	endpoint string
}

func (c *cloudChatClient) Chat(ctx context.Context, req *api.ChatRequest, fn api.ChatResponseFunc) error {
	body, err := json.Marshal(req)
	if err != nil {
		return err
	}
	client := c.http
	if client == nil {
		client = &http.Client{Timeout: 90 * time.Second}
	}
	for attempt := 0; ; attempt++ {
		httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.chatEndpoint(), bytes.NewReader(body))
		if err != nil {
			return err
		}
		httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)
		httpReq.Header.Set("Content-Type", "application/json")
		httpReq.Header.Set("Accept", "application/x-ndjson")
		resp, err := client.Do(httpReq)
		if err != nil {
			if attempt < 2 && ctx.Err() == nil {
				if err := retryWait(ctx, time.Duration(attempt+1)*time.Second); err != nil {
					return err
				}
				continue
			}
			return err
		}
		err = readChatResponse(resp, fn)
		resp.Body.Close()
		var status api.StatusError
		if attempt < 2 && errorsAsStatus(err, &status) && status.StatusCode == http.StatusTooManyRequests {
			if err := retryWait(ctx, time.Duration(attempt+1)*time.Second); err != nil {
				return err
			}
			continue
		}
		return err
	}
}

func (c *cloudChatClient) chatEndpoint() string {
	if c.endpoint != "" {
		return c.endpoint
	}
	return "https://ollama.com/api/chat"
}

func retryWait(ctx context.Context, delay time.Duration) error {
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

func errorsAsStatus(err error, target *api.StatusError) bool {
	if err == nil {
		return false
	}
	value, ok := err.(api.StatusError)
	if ok {
		*target = value
	}
	return ok
}

func readChatResponse(resp *http.Response, fn api.ChatResponseFunc) error {
	defer io.Copy(io.Discard, io.LimitReader(resp.Body, 4<<10))
	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64<<10), 8<<20)
	for scanner.Scan() {
		line := scanner.Bytes()
		var errorBody struct {
			Error string `json:"error"`
		}
		if err := json.Unmarshal(line, &errorBody); err != nil {
			return api.StatusError{StatusCode: resp.StatusCode, Status: resp.Status, ErrorMessage: string(line)}
		}
		if resp.StatusCode >= 400 {
			return api.StatusError{StatusCode: resp.StatusCode, Status: resp.Status, ErrorMessage: errorBody.Error}
		}
		if errorBody.Error != "" {
			return fmt.Errorf("ollama cloud: %s", errorBody.Error)
		}
		var message api.ChatResponse
		if err := json.Unmarshal(line, &message); err != nil {
			return err
		}
		if err := fn(message); err != nil {
			return err
		}
	}
	return scanner.Err()
}

type githubClient struct {
	token string
	http  *http.Client
}

func newGitHubClient(token string) *githubClient {
	return &githubClient{token: token, http: &http.Client{Timeout: 30 * time.Second}}
}

func (c *githubClient) request(ctx context.Context, method, endpoint string, in, out any) error {
	var body io.Reader
	if in != nil {
		encoded, err := json.Marshal(in)
		if err != nil {
			return err
		}
		body = bytes.NewReader(encoded)
	}
	req, err := http.NewRequestWithContext(ctx, method, "https://api.github.com"+endpoint, body)
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+c.token)
	req.Header.Set("Accept", "application/vnd.github+json")
	req.Header.Set("X-GitHub-Api-Version", "2022-11-28")
	resp, err := c.http.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	if err != nil {
		return err
	}
	if resp.StatusCode >= 300 {
		return fmt.Errorf("GitHub API %s: %s", resp.Status, strings.TrimSpace(string(data)))
	}
	if out != nil && len(data) > 0 {
		return json.Unmarshal(data, out)
	}
	return nil
}

func (c *githubClient) listFiles(ctx context.Context, repo string, number int) ([]pullRequestFile, error) {
	var files []pullRequestFile
	for page := 1; page <= 30; page++ {
		var batch []pullRequestFile
		if err := c.request(ctx, http.MethodGet, fmt.Sprintf("/repos/%s/pulls/%d/files?per_page=100&page=%d", repo, number, page), nil, &batch); err != nil {
			return nil, err
		}
		files = append(files, batch...)
		if len(batch) < 100 {
			return files, nil
		}
	}
	return nil, fmt.Errorf("pull request file list exceeds supported pagination")
}

func (c *githubClient) currentHead(ctx context.Context, repo string, number int) (string, error) {
	var pull struct {
		Head struct {
			SHA string `json:"sha"`
		} `json:"head"`
	}
	if err := c.request(ctx, http.MethodGet, fmt.Sprintf("/repos/%s/pulls/%d", repo, number), nil, &pull); err != nil {
		return "", err
	}
	return pull.Head.SHA, nil
}

func (c *githubClient) upsertScoreComment(ctx context.Context, repo string, number int, body string) error {
	var comments []struct {
		ID   int64  `json:"id"`
		Body string `json:"body"`
		User struct {
			Login string `json:"login"`
		} `json:"user"`
	}
	if err := c.request(ctx, http.MethodGet, fmt.Sprintf("/repos/%s/issues/%d/comments?per_page=100", repo, number), nil, &comments); err != nil {
		return err
	}
	for _, comment := range comments {
		if comment.User.Login == "github-actions[bot]" && strings.Contains(comment.Body, "<!-- ollama-ai-review -->") {
			return c.request(ctx, http.MethodPatch, fmt.Sprintf("/repos/%s/issues/comments/%d", repo, comment.ID), map[string]string{"body": body}, nil)
		}
	}
	return c.request(ctx, http.MethodPost, fmt.Sprintf("/repos/%s/issues/%d/comments", repo, number), map[string]string{"body": body}, nil)
}

func (c *githubClient) markScoreStale(ctx context.Context, repo string, number int, head string) error {
	var comments []struct {
		ID   int64  `json:"id"`
		Body string `json:"body"`
		User struct {
			Login string `json:"login"`
		} `json:"user"`
	}
	if err := c.request(ctx, http.MethodGet, fmt.Sprintf("/repos/%s/issues/%d/comments?per_page=100", repo, number), nil, &comments); err != nil {
		return err
	}
	for _, comment := range comments {
		if comment.User.Login == "github-actions[bot]" && strings.Contains(comment.Body, "<!-- ollama-ai-review -->") {
			body := strings.TrimSpace(comment.Body) + fmt.Sprintf("\n\n> **Status:** stale — AI review was unavailable for head `%s`.\n", head)
			return c.request(ctx, http.MethodPatch, fmt.Sprintf("/repos/%s/issues/comments/%d", repo, comment.ID), map[string]string{"body": body}, nil)
		}
	}
	return nil
}

func (c *githubClient) createReview(ctx context.Context, repo string, number int, head string, review review) error {
	comments := make([]map[string]any, 0, len(review.Findings))
	for _, finding := range review.Findings {
		comments = append(comments, map[string]any{
			"path": finding.Path, "line": finding.Line, "side": "RIGHT", "body": findingBody(finding),
		})
	}
	body := map[string]any{
		"commit_id": head,
		"event":     "COMMENT",
		"body":      fmt.Sprintf("<!-- ollama-ai-review head=%s -->\n%s", head, review.Summary),
		"comments":  comments,
	}
	return c.request(ctx, http.MethodPost, fmt.Sprintf("/repos/%s/pulls/%d/reviews", repo, number), body, nil)
}

func (c *githubClient) hasReview(ctx context.Context, repo string, number int, head string) (bool, error) {
	var reviews []struct {
		Body string `json:"body"`
		User struct {
			Login string `json:"login"`
		} `json:"user"`
	}
	if err := c.request(ctx, http.MethodGet, fmt.Sprintf("/repos/%s/pulls/%d/reviews?per_page=100", repo, number), nil, &reviews); err != nil {
		return false, err
	}
	marker := "<!-- ollama-ai-review head=" + head + " -->"
	for _, review := range reviews {
		if review.User.Login == "github-actions[bot]" && strings.Contains(review.Body, marker) {
			return true, nil
		}
	}
	return false, nil
}

func findingBody(f finding) string {
	return fmt.Sprintf("**%s · %s** — %s\n\n%s\n\nSuggested fix: %s\n\nTest: %s", f.Category, f.Severity, f.Title, f.Impact, f.Recommendation, f.Test)
}
