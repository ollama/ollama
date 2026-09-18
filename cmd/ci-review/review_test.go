package main

import (
	"context"
	"strings"
	"testing"

	"github.com/ollama/ollama/agent"
	"github.com/ollama/ollama/api"
)

func TestChangedRightLines(t *testing.T) {
	patch := "@@ -10,2 +10,3 @@\n keep\n-old\n+new\n+added\n"
	got := changedRightLines(patch)
	if got[11] != "new" || got[12] != "added" || len(got) != 2 {
		t.Fatalf("changedRightLines = %#v", got)
	}
}

func TestChangedRightLinesWithCRLFAndDeletion(t *testing.T) {
	patch := "@@ -5,2 +5,2 @@\r\n-old\r\n+new\r\n unchanged\r\n"
	got := changedRightLines(patch)
	if got[5] != "new\r" || len(got) != 1 {
		t.Fatalf("changedRightLines = %#v", got)
	}
}

func TestParseReviewDropsUngroundedAndDuplicateFindings(t *testing.T) {
	files := []pullRequestFile{{Filename: "server/example.go", Changes: 1, Patch: "@@ -1 +1 @@\n-old\n+new value"}}
	corpus := newReviewCorpus("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", files)
	content := `{"summary":"reviewed","findings":[
{"category":"go","severity":"medium","confidence":0.9,"path":"server/example.go","line":1,"title":"bad state","impact":"breaks callers","recommendation":"keep the prior value","test":"add a regression test","evidence":"new value"},
{"category":"go","severity":"medium","confidence":0.9,"path":"server/example.go","line":1,"title":"bad state","impact":"breaks callers","recommendation":"keep the prior value","test":"add a regression test","evidence":"new value"},
{"category":"go","severity":"medium","confidence":0.9,"path":"server/example.go","line":2,"title":"ungrounded","impact":"breaks callers","recommendation":"keep the prior value","test":"add a regression test","evidence":"missing"}
]}`
	review, err := parseReview(content, corpus)
	if err != nil {
		t.Fatal(err)
	}
	if len(review.Findings) != 1 || score(review) != 3 {
		t.Fatalf("review = %#v, score = %d", review, score(review))
	}
}

func TestParseReviewRejectsOnlyUngroundedFindings(t *testing.T) {
	files := []pullRequestFile{{Filename: "server/example.go", Changes: 1, Patch: "@@ -1 +1 @@\n-old\n+new value"}}
	corpus := newReviewCorpus("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", files)
	_, err := parseReview(`{"summary":"reviewed","findings":[{"category":"go","severity":"medium","confidence":0.9,"path":"server/example.go","line":9,"title":"bad state","impact":"breaks callers","recommendation":"keep the prior value","test":"add a regression test","evidence":"new value"}]}`, corpus)
	if err == nil {
		t.Fatal("expected ungrounded finding to reject the result")
	}
}

func TestParseReviewRejectsOversizedSummary(t *testing.T) {
	corpus := newReviewCorpus("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", nil)
	content := `{"summary":"` + strings.Repeat("x", maxSummaryBytes+1) + `","findings":[]}`
	if _, err := parseReview(content, corpus); err == nil {
		t.Fatal("expected oversized summary to fail")
	}
}

func TestScore(t *testing.T) {
	for _, tt := range []struct {
		name string
		in   review
		want int
	}{
		{"none", review{}, 5},
		{"low", review{Findings: []finding{{Severity: "low"}}}, 4},
		{"medium", review{Findings: []finding{{Severity: "medium"}}}, 3},
		{"high", review{Findings: []finding{{Severity: "high"}}}, 2},
		{"critical", review{Findings: []finding{{Severity: "critical"}}}, 1},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if got := score(tt.in); got != tt.want {
				t.Fatalf("score = %d, want %d", got, tt.want)
			}
		})
	}
}

func TestCleanPath(t *testing.T) {
	for _, path := range []string{"../secret", ".git/config", "/absolute", "a/../b", "-option", "a:b", "a\\b"} {
		if cleanPath(path) {
			t.Fatalf("cleanPath(%q) = true", path)
		}
	}
	if !cleanPath("server/routes.go") {
		t.Fatal("expected repository path to be valid")
	}
}

func TestSearchResults(t *testing.T) {
	input := ""
	for i := range 60 {
		input += "file:" + string(rune('a'+i%26)) + "\n"
	}
	if got := len(strings.Split(searchResults(input), "\n")); got != maxSearchHit {
		t.Fatalf("search results = %d, want %d", got, maxSearchHit)
	}
}

func TestToolBudget(t *testing.T) {
	corpus := &reviewCorpus{}
	for range maxToolCalls {
		if _, err := corpus.toolOutput("x"); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := corpus.toolOutput("x"); err == nil {
		t.Fatal("expected call budget error")
	}
}

func TestParseReviewExplainsInvalidFindings(t *testing.T) {
	files := []pullRequestFile{{Filename: "server/example.go", Changes: 1, Patch: "@@ -1 +1 @@\n-old\n+new value"}}
	corpus := newReviewCorpus("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", files)
	_, err := parseReview(`{"summary":"reviewed","findings":[{"category":"go","severity":"low","confidence":0.8,"path":"server/example.go","line":1,"title":"bad state","impact":"breaks callers","recommendation":"keep the prior value","test":"add a regression test","evidence":"new value"}]}`, corpus)
	if err == nil || !strings.Contains(err.Error(), "confidence") || !strings.Contains(err.Error(), "findings: []") {
		t.Fatalf("expected actionable validation error, got %v", err)
	}
}

func TestParseRunOptions(t *testing.T) {
	for _, args := range [][]string{{"--dry"}, {"--dry-run"}} {
		options, err := parseRunOptions(args)
		if err != nil || !options.dryRun {
			t.Fatalf("parseRunOptions(%q) = %#v, %v", args, options, err)
		}
	}
	if _, err := parseRunOptions([]string{"--dry", "unexpected"}); err == nil {
		t.Fatal("expected unexpected argument error")
	}
	if _, err := parseRunOptions([]string{"--trace", "trace.jsonl"}); err == nil {
		t.Fatal("expected trace without dry-run to fail")
	}
}

func TestDryRunSummaryDoesNotClaimToPost(t *testing.T) {
	got := dryRunSummary(dryRunDraft(review{Summary: "reviewed", Findings: []finding{{Severity: "medium"}}}, "0123456789"))
	if !strings.Contains(got, "DRY RUN") || !strings.Contains(got, "nothing was posted") {
		t.Fatalf("dry run summary = %q", got)
	}
}

func TestDryRunDraftIncludesScoreAndInlineComment(t *testing.T) {
	f := finding{Path: "server/example.go", Line: 3, Severity: "medium", Category: "go", Title: "bad state", Impact: "breaks callers", Recommendation: "keep state", Test: "add a test"}
	draft := dryRunDraft(review{Summary: "reviewed", Findings: []finding{f}}, "0123456789")
	if draft["score"] != 3 || !strings.Contains(draft["sticky_comment"].(string), "3/5") {
		t.Fatalf("draft = %#v", draft)
	}
	comments := draft["inline_comments"].([]map[string]any)
	if len(comments) != 1 || comments[0]["path"] != f.Path || comments[0]["line"] != f.Line {
		t.Fatalf("comments = %#v", comments)
	}
}

func TestValidateHeadRepository(t *testing.T) {
	event := pullRequestEvent{}
	event.Repository.FullName = "ollama/ollama"
	event.PullRequest.Head.Repo.FullName = "fork/ollama"
	if err := validateHeadRepository(event, runOptions{}); err == nil {
		t.Fatal("expected fork to be rejected outside dry-run mode")
	}
	if err := validateHeadRepository(event, runOptions{dryRun: true}); err != nil {
		t.Fatalf("expected fork to be accepted in dry-run mode: %v", err)
	}
}

func TestPullHeadRefspecForceUpdatesPrivateRef(t *testing.T) {
	if got, want := pullHeadRefspec(17384), "+refs/pull/17384/head:refs/ollama-ci-review/head"; got != want {
		t.Fatalf("pullHeadRefspec = %q, want %q", got, want)
	}
}

func TestFinalReviewTool(t *testing.T) {
	files := []pullRequestFile{{Filename: "server/example.go", Changes: 1, Patch: "@@ -1 +1 @@\n-old\n+new value"}}
	tool := &finalReviewTool{corpus: newReviewCorpus("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", files)}
	args := map[string]any{
		"summary": "reviewed",
		"findings": []any{map[string]any{
			"category": "go", "severity": "medium", "confidence": 0.9, "path": "server/example.go", "line": 1,
			"title": "bad state", "impact": "breaks callers", "recommendation": "keep the prior value", "test": "add a regression test", "evidence": "new value",
		}},
	}
	if _, err := tool.Execute(context.Background(), agent.ToolContext{}, args); err != nil {
		t.Fatal(err)
	}
	review, submitted := tool.result()
	if !submitted || score(review) != 3 {
		t.Fatalf("submitted = %v, review = %#v", submitted, review)
	}
	if _, err := tool.Execute(context.Background(), agent.ToolContext{}, args); err == nil {
		t.Fatal("expected duplicate final review to fail")
	}
}

func TestFinalReviewToolCountsRejectedCall(t *testing.T) {
	files := []pullRequestFile{{Filename: "server/example.go", Changes: 1, Patch: "@@ -1 +1 @@\n-old\n+new value"}}
	tool := &finalReviewTool{corpus: newReviewCorpus("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", files)}
	_, err := tool.Execute(context.Background(), agent.ToolContext{}, map[string]any{"summary": "reviewed", "findings": []any{map[string]any{"confidence": 0.7}}})
	if err == nil || !strings.Contains(err.Error(), "1/100") || !strings.Contains(err.Error(), "findings: []") {
		t.Fatalf("expected rejected call counter and feedback, got %v", err)
	}
}

type scriptedReviewClient struct {
	responses []api.ChatResponse
	requests  []api.ChatRequest
}

func (c *scriptedReviewClient) Chat(_ context.Context, request *api.ChatRequest, callback api.ChatResponseFunc) error {
	c.requests = append(c.requests, *request)
	if len(c.responses) == 0 {
		return nil
	}
	response := c.responses[0]
	c.responses = c.responses[1:]
	return callback(response)
}

func TestRunReviewAgentHealsMissingFinalReview(t *testing.T) {
	corpus := newReviewCorpus("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", nil)
	finalReview := &finalReviewTool{corpus: corpus}
	args := api.NewToolCallFunctionArguments()
	args.Set("summary", "reviewed with one bounded assumption. Assumptions: no unreviewed files affect this change.")
	args.Set("findings", []any{})
	client := &scriptedReviewClient{responses: []api.ChatResponse{
		{Message: api.Message{Role: "assistant", Content: "I am done."}},
		{Message: api.Message{Role: "assistant", ToolCalls: []api.ToolCall{{ID: "final", Function: api.ToolCallFunction{Name: "final_review", Arguments: args}}}}},
		{Message: api.Message{Role: "assistant"}},
	}}
	session := &agent.Session{Client: client, Tools: &agent.Registry{}}
	session.Tools.Register(finalReview)
	if err := runReviewAgent(context.Background(), session, finalReview, nil, "review the pull request"); err != nil {
		t.Fatal(err)
	}
	if _, submitted := finalReview.result(); !submitted {
		t.Fatal("healing turn did not submit final review")
	}
	if len(client.requests) != 3 || !strings.Contains(messagesText(client.requests[1].Messages), "healing attempt 1 of 2") {
		t.Fatalf("requests = %#v", client.requests)
	}
}

func TestRunReviewAgentLimitsHealingTurns(t *testing.T) {
	corpus := newReviewCorpus("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", nil)
	finalReview := &finalReviewTool{corpus: corpus}
	client := &scriptedReviewClient{responses: []api.ChatResponse{
		{Message: api.Message{Role: "assistant"}},
		{Message: api.Message{Role: "assistant"}},
		{Message: api.Message{Role: "assistant"}},
	}}
	session := &agent.Session{Client: client}
	if err := runReviewAgent(context.Background(), session, finalReview, nil, "review the pull request"); err != nil {
		t.Fatal(err)
	}
	if _, submitted := finalReview.result(); submitted {
		t.Fatal("unexpected final review")
	}
	if len(client.requests) != maxHealingTurns+1 {
		t.Fatalf("requests = %d, want %d", len(client.requests), maxHealingTurns+1)
	}
}

func messagesText(messages []api.Message) string {
	var text strings.Builder
	for _, message := range messages {
		text.WriteString(message.Content)
		text.WriteByte('\n')
	}
	return text.String()
}
