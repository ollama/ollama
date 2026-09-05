// ci-review runs the maintainer pull request review agent in GitHub Actions.
// It is intentionally not wired into the user-facing ollama command.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/ollama/ollama/agent"
	"github.com/ollama/ollama/api"
)

const (
	reviewModelDisplay = "glm-5.2:cloud"
	reviewModelRemote  = "glm-5.2"
	maxToolRounds      = 100
	maxFindings        = 3
	maxHealingTurns    = 2
)

type pullRequestEvent struct {
	Number     int `json:"number"`
	Repository struct {
		FullName string `json:"full_name"`
	} `json:"repository"`
	PullRequest struct {
		Draft bool `json:"draft"`
		Head  struct {
			SHA  string `json:"sha"`
			Repo struct {
				FullName string `json:"full_name"`
			} `json:"repo"`
		} `json:"head"`
		Base struct {
			SHA string `json:"sha"`
		} `json:"base"`
	} `json:"pull_request"`
}

type runOptions struct {
	dryRun    bool
	tracePath string
}

func main() {
	options, err := parseRunOptions(os.Args[1:])
	if err != nil {
		fmt.Fprintln(os.Stderr, "ai review:", err)
		os.Exit(2)
	}
	if err := run(context.Background(), options); err != nil {
		fmt.Fprintln(os.Stderr, "ai review:", err)
		os.Exit(1)
	}
}

func parseRunOptions(args []string) (runOptions, error) {
	flags := flag.NewFlagSet("ci-review", flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	var options runOptions
	flags.BoolVar(&options.dryRun, "dry", false, "run without creating or updating GitHub comments or reviews")
	flags.BoolVar(&options.dryRun, "dry-run", false, "run without creating or updating GitHub comments or reviews")
	flags.StringVar(&options.tracePath, "trace", "", "write a redacted JSONL execution trace (requires --dry)")
	if err := flags.Parse(args); err != nil {
		return runOptions{}, err
	}
	if flags.NArg() != 0 {
		return runOptions{}, fmt.Errorf("unexpected arguments: %s", strings.Join(flags.Args(), " "))
	}
	if options.tracePath != "" && !options.dryRun {
		return runOptions{}, errors.New("--trace requires --dry")
	}
	return options, nil
}

func run(ctx context.Context, options runOptions) error {
	trace, err := newTraceSink(options.tracePath)
	if err != nil {
		return fmt.Errorf("create trace: %w", err)
	}
	if trace != nil {
		defer trace.Close()
		if err := trace.Record("ci_review_started", map[string]any{"dry_run": options.dryRun}); err != nil {
			return err
		}
	}
	event, err := readEvent(os.Getenv("GITHUB_EVENT_PATH"))
	if err != nil {
		return err
	}
	if event.PullRequest.Draft {
		return writeSummary("AI review skipped: pull request is a draft.")
	}
	if err := validateHeadRepository(event, options); err != nil {
		return err
	}
	githubToken := strings.TrimSpace(os.Getenv("GITHUB_TOKEN"))
	apiKey := strings.TrimSpace(os.Getenv("OLLAMA_API_KEY"))
	if githubToken == "" || apiKey == "" {
		return errors.New("GITHUB_TOKEN and OLLAMA_API_KEY are required")
	}

	gh := newGitHubClient(githubToken)
	files, err := gh.listFiles(ctx, event.Repository.FullName, event.Number)
	if err != nil {
		return fmt.Errorf("list pull request files: %w", err)
	}
	corpus := newReviewCorpus(event.PullRequest.Base.SHA, event.PullRequest.Head.SHA, files)
	if !corpus.hasEligibleFiles() {
		return writeSummary("AI review skipped: no eligible code changes.")
	}
	if err := fetchHead(ctx, event.Number, event.PullRequest.Head.SHA); err != nil {
		return err
	}

	registry := &agent.Registry{}
	registry.Register(&reviewDiffTool{corpus: corpus})
	registry.Register(&readFileTool{corpus: corpus})
	registry.Register(&searchTextTool{corpus: corpus})
	finalReview := &finalReviewTool{corpus: corpus}
	registry.Register(finalReview)
	eventSinks := []agent.EventSink(nil)
	if trace != nil {
		eventSinks = append(eventSinks, trace)
	}
	session := &agent.Session{
		Client:     &cloudChatClient{apiKey: apiKey},
		EventSinks: eventSinks,
		Tools:      registry,
		WorkingDir: mustGetwd(),
	}
	runErr := runReviewAgent(ctx, session, finalReview, trace, reviewTask(corpus))
	if runErr != nil {
		if isTransient(runErr) {
			if options.dryRun {
				return writeSummary("Dry run: AI review unavailable for this head; no GitHub comment was changed.")
			}
			if staleErr := gh.markScoreStale(ctx, event.Repository.FullName, event.Number, event.PullRequest.Head.SHA); staleErr != nil {
				return fmt.Errorf("mark score comment stale after transient review failure: %w", staleErr)
			}
			return writeSummary("AI review unavailable for this head; any previous score was marked stale.")
		}
	}
	review, submitted := finalReview.result()
	if !submitted {
		if runErr != nil {
			return fmt.Errorf("run review agent: %w", runErr)
		}
		return errors.New("review agent did not call final_review")
	}
	if head, err := gh.currentHead(ctx, event.Repository.FullName, event.Number); err != nil {
		return fmt.Errorf("check current pull request head: %w", err)
	} else if head != event.PullRequest.Head.SHA {
		return writeSummary("AI review discarded because the pull request changed while it was running.")
	}
	if options.dryRun {
		draft := dryRunDraft(review, event.PullRequest.Head.SHA)
		if err := trace.Record("dry_run_result", draft); err != nil {
			return err
		}
		return writeSummary(dryRunSummary(draft))
	}
	if err := gh.upsertScoreComment(ctx, event.Repository.FullName, event.Number, scoreComment(review, event.PullRequest.Head.SHA)); err != nil {
		return fmt.Errorf("update score comment: %w", err)
	}
	if len(review.Findings) > 0 {
		alreadyPosted, err := gh.hasReview(ctx, event.Repository.FullName, event.Number, event.PullRequest.Head.SHA)
		if err != nil {
			return fmt.Errorf("check existing review: %w", err)
		}
		if alreadyPosted {
			return writeSummary(summary(review, event.PullRequest.Head.SHA))
		}
		if err := gh.createReview(ctx, event.Repository.FullName, event.Number, event.PullRequest.Head.SHA, review); err != nil {
			return fmt.Errorf("create inline review: %w", err)
		}
	}
	return writeSummary(summary(review, event.PullRequest.Head.SHA))
}

func runReviewAgent(ctx context.Context, session *agent.Session, finalReview *finalReviewTool, trace *traceSink, task string) error {
	var history []api.Message
	for healingTurn := 0; ; healingTurn++ {
		result, err := session.Run(ctx, agent.RunOptions{
			Model:         reviewModelRemote,
			SystemPrompt:  reviewSystemPrompt,
			Messages:      history,
			NewMessages:   []api.Message{{Role: "user", Content: task}},
			Options:       map[string]any{"temperature": 0, "num_predict": 2500},
			Think:         &api.ThinkValue{Value: "high"},
			MaxToolRounds: maxToolRounds,
		})
		if err != nil {
			return err
		}
		if _, submitted := finalReview.result(); submitted || healingTurn == maxHealingTurns {
			return nil
		}

		history = result.Messages
		if trace != nil {
			if err := trace.Record("completion_healing", map[string]any{"attempt": healingTurn + 1}); err != nil {
				return err
			}
		}
		task = reviewHealingTask(healingTurn + 1)
	}
}

func validateHeadRepository(event pullRequestEvent, options runOptions) error {
	if event.PullRequest.Head.Repo.FullName != event.Repository.FullName && !options.dryRun {
		return errors.New("ai review only accepts pull requests from this repository outside dry-run mode")
	}
	return nil
}

func dryRunSummary(draft map[string]any) string {
	text := "## DRY RUN — nothing was posted\n\n" + draft["summary"].(string) + "\n\n### Sticky score-comment draft\n\n" + draft["sticky_comment"].(string)
	comments := draft["inline_comments"].([]map[string]any)
	if len(comments) == 0 {
		return text + "\n\n### Inline review draft\n\nNo inline comments would be submitted."
	}
	text += "\n\n### Inline review draft\n"
	for _, comment := range comments {
		text += fmt.Sprintf("\n- `%s:%d`\n\n%s\n", comment["path"], comment["line"], comment["body"])
	}
	return text
}

func readEvent(path string) (pullRequestEvent, error) {
	if path == "" {
		return pullRequestEvent{}, errors.New("GITHUB_EVENT_PATH is required")
	}
	b, err := os.ReadFile(filepath.Clean(path))
	if err != nil {
		return pullRequestEvent{}, err
	}
	var event pullRequestEvent
	if err := json.Unmarshal(b, &event); err != nil {
		return pullRequestEvent{}, err
	}
	if event.Number <= 0 || event.Repository.FullName == "" || !validSHA(event.PullRequest.Head.SHA) || !validSHA(event.PullRequest.Base.SHA) {
		return pullRequestEvent{}, errors.New("event does not describe a pull request with valid commit SHAs")
	}
	return event, nil
}

func mustGetwd() string {
	dir, err := os.Getwd()
	if err != nil {
		return ""
	}
	return dir
}

func writeSummary(text string) error {
	path := os.Getenv("GITHUB_STEP_SUMMARY")
	if path == "" {
		return nil
	}
	f, err := os.OpenFile(filepath.Clean(path), os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o600)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = fmt.Fprintf(f, "## Ollama AI review\n\n%s\n", text)
	return err
}

func isTransient(err error) bool {
	var status api.StatusError
	if errors.As(err, &status) {
		return status.StatusCode == 429 || status.StatusCode >= 500
	}
	return errors.Is(err, context.DeadlineExceeded) || errors.Is(err, context.Canceled) || strings.Contains(err.Error(), "connection")
}
