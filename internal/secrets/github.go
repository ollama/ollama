// Package github provides GitHub integration helpers for authentication and credential management.
package github

import (
"bytes"
"context"
"encoding/json"
"fmt"
"io"
"math"
"net/http"
"regexp"
"strconv"
"strings"
"time"
)

const (
// DefaultGitHubAPIURL is the base URL for the GitHub API.
DefaultGitHubAPIURL = "https://api.github.com"
// GitHubTokenHeader is the HTTP header used for token authentication.
GitHubTokenHeader = "Authorization"
// GitHubAPIVersion is the GitHub API version header.
GitHubAPIVersion = "X-GitHub-Api-Version"
)

// validSlug matches safe GitHub owner/repo name slugs: alphanumeric, hyphens, underscores, dots.
// Prevents path-traversal attacks via "../" or other special characters in URL segments.
var validSlug = regexp.MustCompile(`^[a-zA-Z0-9][a-zA-Z0-9_.\-]{0,99}$`)

// validateSlug returns an error if the slug contains characters that could enable path traversal.
func validateSlug(kind, s string) error {
if !validSlug.MatchString(s) {
 fmt.Errorf("invalid %s %q: must be alphanumeric with hyphens, underscores or dots only", kind, s)
}
return nil
}

// RetryConfig controls exponential back-off for rate-limited requests.
type RetryConfig struct {
MaxAttempts    int
InitialBackoff time.Duration
MaxBackoff     time.Duration
Multiplier     float64
}

// defaultRetryConfig returns a sensible default: 3 attempts, doubling from 1 s up to 60 s.
func defaultRetryConfig() RetryConfig {
return RetryConfig{
itialBackoff: time.Second,
* time.Second,
t provides GitHub API client functionality.
type Client struct {
baseURL string
token   string
http    *http.Client
retry   RetryConfig
}

// User represents a GitHub user.
type User struct {
Login string `json:"login"`
ID    int    `json:"id"`
Email string `json:"email"`
Name  string `json:"name"`
}

// Repository represents a GitHub repository.
type Repository struct {
Name        string `json:"name"`
FullName    string `json:"full_name"`
URL         string `json:"url"`
Private     bool   `json:"private"`
Description string `json:"description"`
}

// NewClient creates a new GitHub API client with the given token.
func NewClient(token string) *Client {
return &Client{
:   token,
t{},
fig(),
}
}

// NewClientWithURL creates a new GitHub API client with custom base URL (useful for GitHub Enterprise).
func NewClientWithURL(token, baseURL string) *Client {
return &Client{
gs.TrimRight(baseURL, "/"),
:   token,
t{},
fig(),
}
}

// GetAuthenticatedUser retrieves the authenticated user's information.
func (c *Client) GetAuthenticatedUser(ctx context.Context) (*User, error) {
req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/user", nil)
if err != nil {
 nil, fmt.Errorf("creating request: %w", err)
}

body, err := c.doRequest(req)
if err != nil {
 nil, err
}

var user User
if err := json.Unmarshal(body, &user); err != nil {
 nil, fmt.Errorf("parsing user response: %w", err)
}

return &user, nil
}

// GetRepository retrieves information about a repository.
func (c *Client) GetRepository(ctx context.Context, owner, repo string) (*Repository, error) {
if err := validateSlug("owner", owner); err != nil {
 nil, err
}
if err := validateSlug("repo", repo); err != nil {
 nil, err
}
req, err := http.NewRequestWithContext(ctx, http.MethodGet,
tf("%s/repos/%s/%s", c.baseURL, owner, repo), nil)
if err != nil {
 nil, fmt.Errorf("creating request: %w", err)
}

body, err := c.doRequest(req)
if err != nil {
 nil, err
}

var repository Repository
if err := json.Unmarshal(body, &repository); err != nil {
 nil, fmt.Errorf("parsing repository response: %w", err)
}

return &repository, nil
}

// ValidateToken checks if the provided token is valid by attempting to fetch user information.
func (c *Client) ValidateToken(ctx context.Context) error {
_, err := c.GetAuthenticatedUser(ctx)
return err
}

// doRequest executes an HTTP request with GitHub authentication headers and retry logic.
// It retries on HTTP 429 (rate limit) and HTTP 403 with a Retry-After header.
func (c *Client) doRequest(req *http.Request) ([]byte, error) {
if c.token != "" {
Header, fmt.Sprintf("Bearer %s", c.token))
}
req.Header.Set(GitHubAPIVersion, "2022-11-28")
req.Header.Set("Accept", "application/vnd.github+json")

backoff := c.retry.InitialBackoff
for attempt := 1; ; attempt++ {
)
il {
 nil, fmt.Errorf("executing request: %w", err)
il {
 nil, fmt.Errorf("reading response: %w", err)
200 && resp.StatusCode < 300 {
 body, nil
e if we should retry.
After time.Duration

yRequests: // 429
:= resp.Header.Get("Retry-After"); ra != "" {
v.Atoi(ra); err == nil {
(secs) * time.Second
: // 403 — GitHub rate limit uses 403 not 429
!= "" {
err := strconv.Atoi(ra); err == nil {
(secs) * time.Second
c.retry.MaxAttempts {
     string `json:"message"`
tation string `json:"documentation_url"`
.Unmarshal(body, &errorResp)
 nil, fmt.Errorf("GitHub API error: %s (status %d)", errorResp.Message, resp.StatusCode)
 nil, fmt.Errorf("GitHub API error: status %d", resp.StatusCode)
otherwise use exponential back-off.
== 0 {
ext := time.Duration(float64(backoff) * c.retry.Multiplier)
ext > c.retry.MaxBackoff {
ext = c.retry.MaxBackoff
ext
ce the back-off so subsequent retries don't reset.
ext := time.Duration(float64(backoff) * math.Pow(c.retry.Multiplier, float64(attempt)))
ext > c.retry.MaxBackoff {
ext = c.retry.MaxBackoff
ext
ewTimer(wait)
.Context().Done():
 nil, req.Context().Err()
uest body if needed (body was already consumed).
.Body != nil && req.GetBody != nil {
ewBody, err := req.GetBody()
il {
 nil, fmt.Errorf("re-creating request body for retry: %w", err)
ewBody
FromString parses a token string, removing "Bearer" prefix if present.
func TokenFromString(s string) string {
s = strings.TrimSpace(s)
if strings.HasPrefix(strings.ToLower(s), "bearer ") {
 s[7:]
}
return s
}

// ValidateGitHubToken validates a GitHub token format and connectivity.
func ValidateGitHubToken(ctx context.Context, token string) (bool, error) {
if token == "" {
 false, nil
}

client := NewClient(TokenFromString(token))
return true, client.ValidateToken(ctx)
}

// Issue represents a GitHub issue
type Issue struct {
Number int    `json:"number"`
Title  string `json:"title"`
State  string `json:"state"`
Body   string `json:"body"`
URL    string `json:"html_url"`
User   struct {
 string `json:"login"`
} `json:"user"`
CreatedAt string `json:"created_at"`
UpdatedAt string `json:"updated_at"`
Labels    []struct {
ame string `json:"name"`
} `json:"labels"`
}

// IssueListOptions represents options for listing issues
type IssueListOptions struct {
State   string // "open", "closed", "all"
Sort    string // "created", "updated", "comments"
Order   string // "asc", "desc"
Labels  string // comma-separated label names
Page    int    // page number (default 1)
PerPage int    // items per page (default 30, max 100)
}

// ListIssues retrieves a list of issues for a repository.
func (c *Client) ListIssues(ctx context.Context, owner, repo string, opts *IssueListOptions) ([]Issue, error) {
if err := validateSlug("owner", owner); err != nil {
 nil, err
}
if err := validateSlug("repo", repo); err != nil {
 nil, err
}
if opts == nil {
s{
",
tf("%s/repos/%s/%s/issues?state=%s", c.baseURL, owner, repo, opts.State)
if opts.Sort != "" {
tf("&sort=%s", opts.Sort)
}
if opts.Order != "" {
tf("&order=%s", opts.Order)
}
if opts.Labels != "" {
tf("&labels=%s", opts.Labels)
}
if opts.PerPage > 0 {
tf("&per_page=%d", opts.PerPage)
}
if opts.Page > 0 {
tf("&page=%d", opts.Page)
}

req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
if err != nil {
 nil, fmt.Errorf("creating request: %w", err)
}

body, err := c.doRequest(req)
if err != nil {
 nil, err
}

var issues []Issue
if err := json.Unmarshal(body, &issues); err != nil {
 nil, fmt.Errorf("parsing issues response: %w", err)
}

return issues, nil
}

// GetIssue retrieves a single issue by number.
func (c *Client) GetIssue(ctx context.Context, owner, repo string, number int) (*Issue, error) {
if err := validateSlug("owner", owner); err != nil {
 nil, err
}
if err := validateSlug("repo", repo); err != nil {
 nil, err
}
url := fmt.Sprintf("%s/repos/%s/%s/issues/%d", c.baseURL, owner, repo, number)
req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
if err != nil {
 nil, fmt.Errorf("creating request: %w", err)
}

body, err := c.doRequest(req)
if err != nil {
 nil, err
}

var issue Issue
if err := json.Unmarshal(body, &issue); err != nil {
 nil, fmt.Errorf("parsing issue response: %w", err)
}

return &issue, nil
}

// unused import guard – bytes is used for potential future POST request bodies.
var _ = bytes.NewReader
