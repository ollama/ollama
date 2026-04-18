package main

import (
	"bytes"
	"context"
	"encoding/csv"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
	"time"
)

func TestFetchIssuesAllPages(t *testing.T) {
	var requests []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests = append(requests, r.URL.RawQuery)
		switch r.URL.Query().Get("page") {
		case "1":
			_, _ = w.Write([]byte(`[{"number":1,"title":"first","state":"open","user":{"login":"alice"},"labels":[],"html_url":"https://example.test/1","created_at":"2026-04-18T00:00:00Z","updated_at":"2026-04-18T00:00:00Z"}]`))
		case "2":
			_, _ = w.Write([]byte(`[{"number":2,"title":"second","state":"open","user":{"login":"bob"},"labels":[],"html_url":"https://example.test/2","created_at":"2026-04-18T00:00:00Z","updated_at":"2026-04-18T00:01:00Z"}]`))
		default:
			_, _ = w.Write([]byte(`[]`))
		}
	}))
	defer server.Close()

	client := NewClientWithURL(server.URL, "token")
	opts := &IssueListOptions{State: "open", PerPage: 1, Sort: "created", Order: "desc", Page: 1}

	issues, err := fetchIssues(context.Background(), client, "owner", "repo", opts, true)
	if err != nil {
		t.Fatalf("fetchIssues returned error: %v", err)
	}
	if len(issues) != 2 {
		t.Fatalf("expected 2 issues, got %d", len(issues))
	}
	if len(requests) != 3 {
		t.Fatalf("expected 3 paginated requests, got %d", len(requests))
	}
}

func TestWriteOutputJSON(t *testing.T) {
	issues := sampleIssues()
	buf := &bytes.Buffer{}

	if err := writeOutput(buf, issues, "json"); err != nil {
		t.Fatalf("writeOutput(json) returned error: %v", err)
	}

	var decoded []Issue
	if err := json.Unmarshal(buf.Bytes(), &decoded); err != nil {
		t.Fatalf("json output is invalid: %v", err)
	}
	if len(decoded) != len(issues) {
		t.Fatalf("expected %d issues, got %d", len(issues), len(decoded))
	}
}

func TestWriteOutputCSV(t *testing.T) {
	issues := sampleIssues()
	buf := &bytes.Buffer{}

	if err := writeOutput(buf, issues, "csv"); err != nil {
		t.Fatalf("writeOutput(csv) returned error: %v", err)
	}

	rows, err := csv.NewReader(strings.NewReader(buf.String())).ReadAll()
	if err != nil {
		t.Fatalf("csv output is invalid: %v", err)
	}
	if len(rows) != len(issues)+1 {
		t.Fatalf("expected %d rows, got %d", len(issues)+1, len(rows))
	}
	if rows[0][0] != "number" || rows[0][1] != "title" {
		t.Fatalf("unexpected csv header: %v", rows[0])
	}
}

func TestWriteOutputTable(t *testing.T) {
	issues := sampleIssues()
	buf := &bytes.Buffer{}

	if err := writeOutput(buf, issues, "table"); err != nil {
		t.Fatalf("writeOutput(table) returned error: %v", err)
	}

	output := buf.String()
	for _, expected := range []string{"#", "STATE", "alice", "first issue title"} {
		if !strings.Contains(output, expected) {
			t.Fatalf("expected table output to contain %q, got %q", expected, output)
		}
	}
}

func TestDiffIssues(t *testing.T) {
	prev := map[int]Issue{
		1: {Number: 1, State: "open", UpdatedAt: "2026-04-18T00:00:00Z"},
	}
	current := []Issue{
		{Number: 1, State: "closed", UpdatedAt: "2026-04-18T00:01:00Z"},
		{Number: 2, State: "open", UpdatedAt: "2026-04-18T00:02:00Z"},
	}

	changed := diffIssues(prev, current)
	if len(changed) != 2 {
		t.Fatalf("expected 2 changed issues, got %d", len(changed))
	}
}

func TestNewClientDefaultTimeout(t *testing.T) {
	client := NewClient("token")
	if client.http.Timeout != defaultHTTPTimeout {
		t.Fatalf("expected default timeout %s, got %s", defaultHTTPTimeout, client.http.Timeout)
	}
	transport, ok := client.http.Transport.(*http.Transport)
	if !ok {
		t.Fatalf("expected http.Transport, got %T", client.http.Transport)
	}
	if transport.ResponseHeaderTimeout != 15*time.Second {
		t.Fatalf("expected response header timeout 15s, got %s", transport.ResponseHeaderTimeout)
	}
}

func TestWriteOutputUnsupportedFormat(t *testing.T) {
	err := writeOutput(&bytes.Buffer{}, sampleIssues(), "xml")
	if err == nil {
		t.Fatal("expected error for unsupported format")
	}
	if !strings.Contains(err.Error(), "unsupported output format") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestNormalizePerPage(t *testing.T) {
	if got := normalizePerPage(0); got != 20 {
		t.Fatalf("expected default per-page 20, got %d", got)
	}
	if got := normalizePerPage(101); got != 100 {
		t.Fatalf("expected capped per-page 100, got %d", got)
	}
	if got := normalizePerPage(42); got != 42 {
		t.Fatalf("expected per-page 42, got %d", got)
	}
}

func TestParseLabels(t *testing.T) {
	labels := parseLabels("bug, help wanted , cli")
	if got, want := len(labels), 3; got != want {
		t.Fatalf("expected %d labels, got %d", want, got)
	}
	for index, expected := range []string{"bug", "help wanted", "cli"} {
		if labels[index] != expected {
			t.Fatalf("expected label %q at index %d, got %q", expected, index, labels[index])
		}
	}
}

func TestFormatDateAndTruncate(t *testing.T) {
	if got := formatDate("2026-04-18T00:03:00Z"); got != "2026-04-18" {
		t.Fatalf("unexpected formatted date: %s", got)
	}
	if got := truncate("1234567890", 7); got != "1234..." {
		t.Fatalf("unexpected truncate result: %s", got)
	}
	if got := truncate(strconv.Itoa(42), 7); got != "42" {
		t.Fatalf("unexpected short truncate result: %s", got)
	}
}

func sampleIssues() []Issue {
	return []Issue{
		{
			Number:    1,
			Title:     "first issue title",
			State:     "open",
			User:      IssueUser{Login: "alice"},
			Labels:    []IssueLabel{{Name: "bug"}, {Name: "cli"}},
			HTMLURL:   "https://example.test/1",
			CreatedAt: "2026-04-18T00:00:00Z",
			UpdatedAt: "2026-04-18T00:01:00Z",
		},
		{
			Number:    2,
			Title:     "second issue title",
			State:     "closed",
			User:      IssueUser{Login: "bob"},
			Labels:    []IssueLabel{{Name: "enhancement"}},
			HTMLURL:   "https://example.test/2",
			CreatedAt: "2026-04-18T00:02:00Z",
			UpdatedAt: "2026-04-18T00:03:00Z",
		},
	}
}
