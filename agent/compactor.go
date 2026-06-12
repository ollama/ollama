package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/ollama/ollama/api"
)

const (
	defaultCompactionContextWindowTokens = 32768
	defaultCompactionKeepUserTurns       = 3
	defaultCompactionThreshold           = 0.8

	compactionSummaryMessagePrefix = "Conversation summary:\n"
	maxCompactionSummaryBytes      = 16 * 1024
	compactionSummaryTruncated     = "\n\n[summary truncated]"
)

type Compactor interface {
	MaybeCompact(context.Context, CompactionRequest) (CompactionResult, error)
}

type CompactionStore interface {
	ArchiveForCompaction(context.Context, string, int, string) error
}

type CompactionOptions struct {
	ContextWindowTokens int
	KeepUserTurns       int
	Threshold           float64
}

type CompactionRequest struct {
	ChatID    string
	Model     string
	Messages  []api.Message
	Latest    api.ChatResponse
	Options   map[string]any
	KeepAlive *api.Duration
	Force     bool
	Progress  func(CompactionProgress)
}

type CompactionProgress struct {
	Tokens int
}

type CompactionResult struct {
	Messages  []api.Message
	Compacted bool
	Due       bool
	Summary   string
	Reason    string
}

type SimpleCompactor struct {
	Client  ChatClient
	Store   CompactionStore
	Options CompactionOptions
}

func NewSimpleCompactor(client ChatClient, store CompactionStore, opts CompactionOptions) *SimpleCompactor {
	return &SimpleCompactor{Client: client, Store: store, Options: opts}
}

func (c *SimpleCompactor) MaybeCompact(ctx context.Context, req CompactionRequest) (CompactionResult, error) {
	result := CompactionResult{Messages: req.Messages}
	if c == nil {
		return result, nil
	}

	result.Due = req.Force || c.shouldCompact(req)
	if !result.Due {
		return result, nil
	}
	if c.Client == nil {
		result.Reason = "compaction is unavailable"
		return result, nil
	}

	keepUserTurns := c.keepUserTurns()
	prefix, previousSummary, archive, suffix, ok := splitCompactionMessages(req.Messages, keepUserTurns)
	if !ok || len(archive) == 0 {
		result.Reason = "not enough older messages to compact"
		return result, nil
	}

	summary, err := c.summarize(ctx, req, previousSummary, archive)
	if err != nil {
		result.Reason = err.Error()
		return result, err
	}
	summary = truncateCompactionSummary(strings.TrimSpace(summary))
	if summary == "" {
		result.Reason = "summary was empty"
		return result, nil
	}

	if c.Store != nil && req.ChatID != "" {
		if err := c.Store.ArchiveForCompaction(ctx, req.ChatID, keepUserTurns, summary); err != nil {
			result.Reason = err.Error()
			return result, err
		}
	}

	compacted := make([]api.Message, 0, len(prefix)+1+len(suffix))
	compacted = append(compacted, prefix...)
	// Keep the rolling summary as a user message so the stable system prompt prefix
	// remains cache-friendly across compactions.
	compacted = append(compacted, api.Message{Role: "user", Content: compactionSummaryMessagePrefix + summary})
	compacted = append(compacted, suffix...)
	result.Messages = compacted
	result.Compacted = true
	result.Summary = summary
	return result, nil
}

func (c *SimpleCompactor) shouldCompact(req CompactionRequest) bool {
	if req.Latest.PromptEvalCount <= 0 {
		return false
	}

	contextWindow := c.contextWindowTokens(req.Options)
	return float64(req.Latest.PromptEvalCount) >= float64(contextWindow)*c.threshold()
}

func (c *SimpleCompactor) contextWindowTokens(options map[string]any) int {
	return ResolveContextWindowTokens(options, c.Options.ContextWindowTokens)
}

func (c *SimpleCompactor) keepUserTurns() int {
	if c.Options.KeepUserTurns > 0 {
		return c.Options.KeepUserTurns
	}
	return defaultCompactionKeepUserTurns
}

func (c *SimpleCompactor) threshold() float64 {
	return ResolveCompactionThreshold(c.Options.Threshold)
}

func ResolveContextWindowTokens(options map[string]any, configured int) int {
	if n := intOption(options, "num_ctx"); n > 0 {
		if configured > 0 {
			return min(n, configured)
		}
		return n
	}
	if configured > 0 {
		return configured
	}
	return defaultCompactionContextWindowTokens
}

func ResolveCompactionThreshold(configured float64) float64 {
	if configured > 0 {
		return configured
	}
	return defaultCompactionThreshold
}

func (c *SimpleCompactor) summarize(ctx context.Context, req CompactionRequest, previousSummary string, archive []api.Message) (string, error) {
	body, err := compactionPrompt(previousSummary, archive)
	if err != nil {
		return "", err
	}

	chatReq := &api.ChatRequest{
		Model: req.Model,
		Messages: []api.Message{
			{
				Role:    "system",
				Content: "Summarize the archived part of an Ollama CLI agent conversation. Preserve user goals, decisions, files, commands, tool results, and unresolved tasks needed to continue. Omit private reasoning and return only the summary.",
			},
			{
				Role:    "user",
				Content: body,
			},
		},
		Options: req.Options,
	}
	if req.KeepAlive != nil {
		chatReq.KeepAlive = req.KeepAlive
	}

	var summary strings.Builder
	if err := c.Client.Chat(ctx, chatReq, func(response api.ChatResponse) error {
		summary.WriteString(response.Message.Content)
		if req.Progress != nil {
			tokens := response.EvalCount
			if tokens <= 0 {
				tokens = estimateCompactionTokens(summary.String())
			}
			req.Progress(CompactionProgress{Tokens: tokens})
		}
		return nil
	}); err != nil {
		return "", err
	}
	return summary.String(), nil
}

func truncateCompactionSummary(summary string) string {
	if len(summary) <= maxCompactionSummaryBytes {
		return summary
	}
	limit := maxCompactionSummaryBytes - len(compactionSummaryTruncated)
	if limit < 0 {
		limit = 0
	}
	var b strings.Builder
	for _, r := range summary {
		if b.Len()+len(string(r)) > limit {
			break
		}
		b.WriteRune(r)
	}
	return strings.TrimSpace(b.String()) + compactionSummaryTruncated
}

func estimateCompactionTokens(text string) int {
	text = strings.TrimSpace(text)
	if text == "" {
		return 0
	}
	return max(1, (len([]rune(text))+3)/4)
}

func compactionPrompt(previousSummary string, archive []api.Message) (string, error) {
	messages := make([]api.Message, 0, len(archive))
	for _, msg := range archive {
		msg.Thinking = ""
		msg.Images = nil
		messages = append(messages, msg)
	}

	payload, err := json.MarshalIndent(messages, "", "  ")
	if err != nil {
		return "", fmt.Errorf("marshal compaction messages: %w", err)
	}

	var b strings.Builder
	if strings.TrimSpace(previousSummary) != "" {
		b.WriteString("Previous summary:\n")
		b.WriteString(strings.TrimSpace(previousSummary))
		b.WriteString("\n\n")
	}
	b.WriteString("Messages to archive as JSON:\n")
	b.Write(payload)
	return b.String(), nil
}

func splitCompactionMessages(messages []api.Message, keepUserTurns int) (prefix []api.Message, previousSummary string, archive []api.Message, suffix []api.Message, ok bool) {
	if keepUserTurns <= 0 {
		keepUserTurns = defaultCompactionKeepUserTurns
	}

	start := 0
	for start < len(messages) && messages[start].Role == "system" && !isCompactionSummary(messages[start]) {
		prefix = append(prefix, messages[start])
		start++
	}
	if start < len(messages) && isCompactionSummary(messages[start]) {
		previousSummary = strings.TrimSpace(strings.TrimPrefix(messages[start].Content, compactionSummaryMessagePrefix))
		start++
	}

	userTurns := 0
	suffixStart := len(messages)
	for i := len(messages) - 1; i >= start; i-- {
		if messages[i].Role == "user" {
			userTurns++
			if userTurns == keepUserTurns {
				suffixStart = i
				break
			}
		}
	}
	if userTurns < keepUserTurns || suffixStart <= start {
		return prefix, previousSummary, nil, nil, false
	}

	return prefix, previousSummary, messages[start:suffixStart], messages[suffixStart:], true
}

func isCompactionSummary(msg api.Message) bool {
	return (msg.Role == "user" || msg.Role == "system") && strings.HasPrefix(msg.Content, compactionSummaryMessagePrefix)
}

func intOption(options map[string]any, key string) int {
	if options == nil {
		return 0
	}
	switch v := options[key].(type) {
	case int:
		return v
	case int64:
		return int(v)
	case float64:
		return int(v)
	case float32:
		return int(v)
	case json.Number:
		n, _ := v.Int64()
		return int(n)
	default:
		return 0
	}
}
