package github

import (
	"testing"
)

func TestTokenFromString(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{
			input:    "Bearer token123",
			expected: "token123",
		},
		{
			input:    "bearer token456",
			expected: "token456",
		},
		{
			input:    "token789",
			expected: "token789",
		},
		{
			input:    "  token_with_spaces  ",
			expected: "token_with_spaces",
		},
		{
			input:    "Bearer   token_with_spaces",
			expected: "token_with_spaces",
		},
		{
			input:    "",
			expected: "",
		},
	}

	for i, tt := range tests {
		result := TokenFromString(tt.input)
		if result != tt.expected {
			t.Errorf("test %d: got %q, want %q", i, result, tt.expected)
		}
	}
}

func TestNewClient(t *testing.T) {
	token := "test-token"
	client := NewClient(token)

	if client == nil {
		t.Fatal("got nil client")
	}

	if client.token != token {
		t.Errorf("got token %q, want %q", client.token, token)
	}

	if client.baseURL != DefaultGitHubAPIURL {
		t.Errorf("got baseURL %q, want %q", client.baseURL, DefaultGitHubAPIURL)
	}
}

func TestNewClientWithURL(t *testing.T) {
	token := "test-token"
	customURL := "https://github.enterprise.com/api/v3/"
	client := NewClientWithURL(token, customURL)

	if client == nil {
		t.Fatal("got nil client")
	}

	expectedURL := "https://github.enterprise.com/api/v3"
	if client.baseURL != expectedURL {
		t.Errorf("got baseURL %q, want %q", client.baseURL, expectedURL)
	}
}

func TestIssueListOptions(t *testing.T) {
	tests := []struct {
		name    string
		opts    *IssueListOptions
		hasTags bool
	}{
		{
			name: "nil_options",
			opts: nil,
		},
		{
			name: "open_issues",
			opts: &IssueListOptions{
				State:   "open",
				PerPage: 30,
			},
		},
		{
			name: "with_labels",
			opts: &IssueListOptions{
				State:   "open",
				Labels:  "bug,enhancement",
				PerPage: 50,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.opts == nil {
				return
			}
			if tt.opts.State == "" {
				t.Error("expected non-empty state")
			}
		})
	}
}
