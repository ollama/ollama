package main

import (
  "context"
  "encoding/csv"
  "encoding/json"
  "errors"
  "flag"
  "fmt"
  "io"
  "log"
  "net/http"
  "net/url"
  "os"
  "regexp"
  "strconv"
  "strings"
  "text/tabwriter"
  "time"
)

const defaultGitHubAPIURL = "https://api.github.com"
const defaultHTTPTimeout = 30 * time.Second

var repoSlugRE = regexp.MustCompile(`^[A-Za-z0-9_.-]+$`)

type IssueLabel struct {
  Name string `json:"name"`
}

type IssueUser struct {
  Login string `json:"login"`
}

type Issue struct {
  Number    int          `json:"number"`
  Title     string       `json:"title"`
  State     string       `json:"state"`
  Labels    []IssueLabel `json:"labels"`
  User      IssueUser    `json:"user"`
  HTMLURL   string       `json:"html_url"`
  CreatedAt string       `json:"created_at"`
  UpdatedAt string       `json:"updated_at"`
}

type IssueListOptions struct {
  State   string
  PerPage int
  Sort    string
  Order   string
  Labels  []string
  Page    int
}

type ClientOptions struct {
  BaseURL     string
  HTTPTimeout time.Duration
  Transport   *http.Transport
}

type Client struct {
  baseURL string
  token   string
  http    *http.Client
}

func NewClient(token string) *Client {
  return NewClientWithOptions(token, ClientOptions{})
}

func NewClientWithURL(baseURL, token string) *Client {
  return NewClientWithOptions(token, ClientOptions{BaseURL: baseURL})
}

func NewClientWithOptions(token string, opts ClientOptions) *Client {
  baseURL := opts.BaseURL
  if baseURL == "" {
    baseURL = defaultGitHubAPIURL
  }

  httpTimeout := opts.HTTPTimeout
  if httpTimeout == 0 {
    httpTimeout = defaultHTTPTimeout
  }

  transport := opts.Transport
  if transport == nil {
    transport = &http.Transport{
      Proxy:                 http.ProxyFromEnvironment,
      TLSHandshakeTimeout:   10 * time.Second,
      ResponseHeaderTimeout: 15 * time.Second,
      IdleConnTimeout:       90 * time.Second,
      MaxIdleConnsPerHost:   10,
    }
  }

  return &Client{
    baseURL: strings.TrimRight(baseURL, "/"),
    token:   token,
    http: &http.Client{
      Timeout:   httpTimeout,
      Transport: transport,
    },
  }
}

func (c *Client) ListIssues(ctx context.Context, owner, repo string, opts *IssueListOptions) ([]Issue, error) {
  if !isValidRepoSlug(owner) || !isValidRepoSlug(repo) {
    return nil, fmt.Errorf("invalid owner/repo slug: %s/%s", owner, repo)
  }
  if opts == nil {
    return nil, errors.New("issue list options are required")
  }

  endpoint, err := url.Parse(fmt.Sprintf("%s/repos/%s/%s/issues", c.baseURL, owner, repo))
  if err != nil {
    return nil, err
  }

  query := endpoint.Query()
  query.Set("state", defaultString(opts.State, "open"))
  query.Set("per_page", strconv.Itoa(normalizePerPage(opts.PerPage)))
  query.Set("sort", defaultString(opts.Sort, "created"))
  query.Set("direction", defaultString(opts.Order, "desc"))
  query.Set("page", strconv.Itoa(defaultInt(opts.Page, 1)))
  if len(opts.Labels) > 0 {
    query.Set("labels", strings.Join(opts.Labels, ","))
  }
  endpoint.RawQuery = query.Encode()

  req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint.String(), nil)
  if err != nil {
    return nil, err
  }
  req.Header.Set("Accept", "application/vnd.github+json")
  req.Header.Set("User-Agent", "ollama-github-issues")
  if c.token != "" {
    req.Header.Set("Authorization", "Bearer "+c.token)
  }

  resp, err := c.http.Do(req)
  if err != nil {
    return nil, err
  }
  defer resp.Body.Close()

  if resp.StatusCode != http.StatusOK {
    body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
    return nil, fmt.Errorf("GitHub API error: %s: %s", resp.Status, strings.TrimSpace(string(body)))
  }

  var issues []Issue
  if err := json.NewDecoder(resp.Body).Decode(&issues); err != nil {
    return nil, err
  }

  filtered := issues[:0]
  for _, issue := range issues {
    if issue.Number == 0 {
      continue
    }
    filtered = append(filtered, issue)
  }
  return filtered, nil
}

func main() {
  owner := flag.String("owner", "ollama", "Repository owner")
  repo := flag.String("repo", "ollama", "Repository name")
  state := flag.String("state", "open", "Issue state: open, closed, all")
  limit := flag.Int("limit", 20, "Number of issues per page (max 100)")
  sortBy := flag.String("sort", "created", "Sort by: created, updated, comments")
  order := flag.String("order", "desc", "Order: asc, desc")
  labels := flag.String("labels", "", "Filter by labels (comma-separated)")
  allPages := flag.Bool("all-pages", false, "Fetch all pages of results")
  output := flag.String("output", "table", "Output format: table, json, csv")
  outFile := flag.String("out-file", "", "Write output to file instead of stdout")
  watch := flag.Duration("watch", 0, "Poll interval for watch mode (e.g. 30s). 0 = disabled")
  diff := flag.Bool("diff", false, "In watch mode, only show new/changed issues")
  baseURL := flag.String("base-url", defaultGitHubAPIURL, "GitHub API base URL")
  flag.Parse()

  token := githubToken()
  if token == "" {
    log.Fatal("OLLAMA_GITHUB_TOKEN, GITHUB_TOKEN, or GH_TOKEN must be set")
  }

  client := NewClientWithURL(*baseURL, token)
  opts := &IssueListOptions{
    State:   *state,
    PerPage: *limit,
    Sort:    *sortBy,
    Order:   *order,
    Labels:  parseLabels(*labels),
    Page:    1,
  }

  out := io.Writer(os.Stdout)
  if *outFile != "" {
    file, err := os.Create(*outFile)
    if err != nil {
      log.Fatalf("create output file: %v", err)
    }
    defer file.Close()
    out = file
  }

  if *watch > 0 {
    runWatch(client, *owner, *repo, opts, *allPages, *output, out, *diff, *watch)
    return
  }

  ctx := context.Background()
  issues, err := fetchIssues(ctx, client, *owner, *repo, opts, *allPages)
  if err != nil {
    log.Fatalf("fetch issues: %v", err)
  }

  if err := writeOutput(out, issues, *output); err != nil {
    log.Fatalf("write output: %v", err)
  }
}

func fetchIssues(ctx context.Context, client *Client, owner, repo string, opts *IssueListOptions, allPages bool) ([]Issue, error) {
  if !allPages {
    return client.ListIssues(ctx, owner, repo, opts)
  }

  var all []Issue
  page := 1
  for {
    pageOpts := *opts
    pageOpts.Page = page
    issues, err := client.ListIssues(ctx, owner, repo, &pageOpts)
    if err != nil {
      return nil, err
    }
    if len(issues) == 0 {
      return all, nil
    }
    all = append(all, issues...)
    if len(issues) < normalizePerPage(opts.PerPage) {
      return all, nil
    }
    page++
  }
}

func diffIssues(prev map[int]Issue, current []Issue) []Issue {
  var changed []Issue
  for _, issue := range current {
    previous, ok := prev[issue.Number]
    if !ok || previous.UpdatedAt != issue.UpdatedAt || previous.State != issue.State {
      changed = append(changed, issue)
    }
  }
  return changed
}

func writeOutput(w io.Writer, issues []Issue, format string) error {
  switch format {
  case "json":
    enc := json.NewEncoder(w)
    enc.SetIndent("", "  ")
    return enc.Encode(issues)
  case "csv":
    cw := csv.NewWriter(w)
    if err := cw.Write([]string{"number", "title", "state", "author", "labels", "created_at", "updated_at", "url"}); err != nil {
      return err
    }
    for _, issue := range issues {
      labels := make([]string, 0, len(issue.Labels))
      for _, label := range issue.Labels {
        labels = append(labels, label.Name)
      }
      if err := cw.Write([]string{
        strconv.Itoa(issue.Number),
        issue.Title,
        issue.State,
        issue.User.Login,
        strings.Join(labels, ";"),
        issue.CreatedAt,
        issue.UpdatedAt,
        issue.HTMLURL,
      }); err != nil {
        return err
      }
    }
    cw.Flush()
    return cw.Error()
  case "table":
    tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
    fmt.Fprintln(tw, "#\tSTATE\tAUTHOR\tUPDATED\tTITLE")
    fmt.Fprintln(tw, "---\t---\t---\t---\t---")
    for _, issue := range issues {
      fmt.Fprintf(tw, "%d\t%s\t%s\t%s\t%s\n",
        issue.Number,
        issue.State,
        issue.User.Login,
        formatDate(issue.UpdatedAt),
        truncate(issue.Title, 60),
      )
    }
    return tw.Flush()
  default:
    return fmt.Errorf("unsupported output format: %s", format)
  }
}

func runWatch(client *Client, owner, repo string, opts *IssueListOptions, allPages bool, format string, out io.Writer, diffOnly bool, interval time.Duration) {
  prev := map[int]Issue{}
  ticker := time.NewTicker(interval)
  defer ticker.Stop()

  fmt.Fprintf(os.Stderr, "Watching %s/%s every %s (Ctrl-C to stop)\n", owner, repo, interval)
  for {
    ctx := context.Background()
    issues, err := fetchIssues(ctx, client, owner, repo, opts, allPages)
    if err != nil {
      fmt.Fprintf(os.Stderr, "fetch error: %v\n", err)
    } else {
      toShow := issues
      if diffOnly {
        toShow = diffIssues(prev, issues)
      }
      if len(toShow) > 0 {
        if diffOnly {
          fmt.Fprintf(os.Stderr, "[%s] %d changed issue(s)\n", time.Now().Format(time.RFC3339), len(toShow))
        }
        if err := writeOutput(out, toShow, format); err != nil {
          fmt.Fprintf(os.Stderr, "output error: %v\n", err)
        }
      }
      for _, issue := range issues {
        prev[issue.Number] = issue
      }
    }
    <-ticker.C
  }
}

func githubToken() string {
  for _, key := range []string{"OLLAMA_GITHUB_TOKEN", "GITHUB_TOKEN", "GH_TOKEN"} {
    if value := strings.TrimSpace(os.Getenv(key)); value != "" {
      return value
    }
  }
  return ""
}

func parseLabels(raw string) []string {
  if strings.TrimSpace(raw) == "" {
    return nil
  }
  parts := strings.Split(raw, ",")
  labels := make([]string, 0, len(parts))
  for _, part := range parts {
    label := strings.TrimSpace(part)
    if label != "" {
      labels = append(labels, label)
    }
  }
  return labels
}

func normalizePerPage(perPage int) int {
  if perPage <= 0 {
    return 20
  }
  if perPage > 100 {
    return 100
  }
  return perPage
}

func defaultString(value, fallback string) string {
  if value == "" {
    return fallback
  }
  return value
}

func defaultInt(value, fallback int) int {
  if value == 0 {
    return fallback
  }
  return value
}

func isValidRepoSlug(value string) bool {
  return repoSlugRE.MatchString(value)
}

func truncate(value string, maxLen int) string {
  if len(value) <= maxLen {
    return value
  }
  if maxLen <= 3 {
    return value[:maxLen]
  }
  return value[:maxLen-3] + "..."
}

func formatDate(value string) string {
  if len(value) >= 10 {
    return value[:10]
  }
  return value
}
