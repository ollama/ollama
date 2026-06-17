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
	compactionToolName             = "compact_conversation"
	compactionToolCallID           = "ollama_compaction"
	maxCompactionSummaryBytes      = 16 * 1024
	compactionSummaryTruncated     = "\n\n[summary truncated]"

	compactionSystemPrompt = "Summarize the archived part of an Ollama CLI agent conversation. Preserve user goals, decisions, files, commands, tool results, and unresolved tasks needed to continue. Omit private reasoning and return only the summary."
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
	ChatID       string
	Model        string
	SystemPrompt string
	Messages     []api.Message
	Tools        api.Tools
	Format       string
	Latest       api.ChatResponse
	Options      map[string]any
	KeepAlive    *api.Duration
	Force        bool
	Progress     func(CompactionProgress)
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
	prefix, previousSummary, archive, suffix, keptUserTurns, ok := splitCompactionMessages(req.Messages, keepUserTurns)
	if !ok || len(archive) == 0 {
		result.Reason = "nothing to compact"
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
		if err := c.Store.ArchiveForCompaction(ctx, req.ChatID, keptUserTurns, summary); err != nil {
			result.Reason = err.Error()
			return result, err
		}
	}

	compacted := make([]api.Message, 0, len(prefix)+len(suffix)+2)
	compacted = append(compacted, prefix...)
	compacted = append(compacted, suffix...)
	compacted = append(compacted, compactionSummaryMessages(summary)...)
	result.Messages = compacted
	result.Compacted = true
	result.Summary = summary
	return result, nil
}

func (c *SimpleCompactor) shouldCompact(req CompactionRequest) bool {
	contextWindow := c.contextWindowTokens(req.Options)
	threshold := int(float64(contextWindow) * c.threshold())
	if threshold <= 0 {
		return false
	}
	if req.Latest.PromptEvalCount > 0 && req.Latest.PromptEvalCount >= threshold {
		return true
	}
	// TODO(parthsareen): If the newest kept user turn contains the oversized
	// tool output, compaction can remove older history but still leave the next
	// prompt above the safety threshold. Pair this estimate trigger with
	// context-aware tool-output paging/range reads so the kept suffix can shrink.
	return estimateCompactionRequestTokens(req) >= threshold
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
	body, err := compactionPrompt(previousSummary, archive, c.compactionPromptBodyBudgetTokens(req.Options))
	if err != nil {
		return "", err
	}

	chatReq := &api.ChatRequest{
		Model: req.Model,
		Messages: []api.Message{
			{
				Role:    "system",
				Content: compactionSystemPrompt,
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

func compactionSummaryMessage(summary string) string {
	return compactionSummaryMessagePrefix + strings.TrimSpace(summary)
}

func compactionSummaryMessages(summary string) []api.Message {
	args := api.NewToolCallFunctionArguments()
	args.Set("reason", "context compaction")
	return []api.Message{
		{
			Role: "assistant",
			ToolCalls: []api.ToolCall{{
				ID: compactionToolCallID,
				Function: api.ToolCallFunction{
					Name:      compactionToolName,
					Arguments: args,
				},
			}},
		},
		{
			Role:       "tool",
			ToolName:   compactionToolName,
			ToolCallID: compactionToolCallID,
			Content:    compactionSummaryMessage(summary),
		},
	}
}

func (c *SimpleCompactor) compactionPromptBodyBudgetTokens(options map[string]any) int {
	contextWindow := c.contextWindowTokens(options)
	threshold := int(float64(contextWindow) * c.threshold())
	if threshold <= 0 {
		return 0
	}
	systemTokens := estimateCompactionTokens("system") + estimateCompactionTokens(compactionSystemPrompt)
	userRoleTokens := estimateCompactionTokens("user")
	budget := threshold - systemTokens - userRoleTokens
	if budget <= 0 {
		return 0
	}
	return budget
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

func estimateMessagesTokens(messages []api.Message) int {
	var total int
	for _, msg := range messages {
		total += estimateCompactionTokens(msg.Role)
		total += estimateCompactionTokens(msg.Content)
		total += estimateCompactionTokens(msg.Thinking)
		total += estimateCompactionTokens(msg.ToolName)
		total += estimateCompactionTokens(msg.ToolCallID)
		for _, call := range msg.ToolCalls {
			total += estimateCompactionTokens(call.Function.Name)
			total += estimateCompactionTokens(call.Function.Arguments.String())
		}
	}
	return total
}

func estimateCompactionRequestTokens(req CompactionRequest) int {
	requestMessages := sanitizeMessagesForRequest(req.Messages)
	if strings.TrimSpace(req.SystemPrompt) != "" {
		requestMessages = make([]api.Message, 0, len(req.Messages)+1)
		requestMessages = append(requestMessages, api.Message{Role: "system", Content: strings.TrimSpace(req.SystemPrompt)})
		requestMessages = append(requestMessages, sanitizeMessagesForRequest(req.Messages)...)
	}

	payload := struct {
		Messages []api.Message   `json:"messages,omitempty"`
		Tools    api.Tools       `json:"tools,omitempty"`
		Format   json.RawMessage `json:"format,omitempty"`
	}{
		Messages: requestMessages,
		Tools:    req.Tools,
	}
	if rawFormat, ok := compactionFormatForEstimate(req.Format); ok {
		payload.Format = rawFormat
	}
	if data, err := json.Marshal(payload); err == nil {
		return estimateCompactionTokens(string(data))
	}

	total := estimateMessagesTokens(requestMessages)
	total += estimateCompactionTokens(req.Tools.String())
	total += estimateCompactionTokens(req.Format)
	return total
}

func compactionFormatForEstimate(format string) (json.RawMessage, bool) {
	format = strings.TrimSpace(format)
	if format == "" {
		return nil, false
	}
	if format == "json" {
		return json.RawMessage(`"json"`), true
	}
	if !json.Valid([]byte(format)) {
		return nil, false
	}
	return json.RawMessage(format), true
}

func compactionPrompt(previousSummary string, archive []api.Message, maxTokens int) (string, error) {
	messages := make([]api.Message, 0, len(archive))
	for _, msg := range archive {
		msg.Thinking = ""
		msg.Images = nil
		messages = append(messages, msg)
	}
	return renderCompactionPrompt(previousSummary, fitCompactionMessagesToBudget(previousSummary, messages, maxTokens))
}

func renderCompactionPrompt(previousSummary string, messages []api.Message) (string, error) {
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

func fitCompactionMessagesToBudget(previousSummary string, messages []api.Message, maxTokens int) []api.Message {
	if maxTokens <= 0 {
		return messages
	}
	fitted := append([]api.Message(nil), messages...)
	for range 16 {
		body, err := renderCompactionPrompt(previousSummary, fitted)
		if err != nil || estimateCompactionTokens(body) <= maxTokens {
			return fitted
		}

		idx := largestCompactionContentMessage(fitted)
		if idx < 0 {
			return fitted
		}
		overageTokens := estimateCompactionTokens(body) - maxTokens
		currentRunes := len([]rune(fitted[idx].Content))
		nextRunes := currentRunes - overageTokens*4 - 256
		if nextRunes >= currentRunes {
			nextRunes = currentRunes / 2
		}
		fitted[idx].Content = truncateToolResultContentTo(fitted[idx].Content, nextRunes)
	}
	return fitted
}

func largestCompactionContentMessage(messages []api.Message) int {
	idx := -1
	size := 0
	for i, msg := range messages {
		n := len([]rune(msg.Content))
		if n > size {
			idx = i
			size = n
		}
	}
	return idx
}

func splitCompactionMessages(messages []api.Message, keepUserTurns int) (prefix []api.Message, previousSummary string, archive []api.Message, suffix []api.Message, keptUserTurns int, ok bool) {
	if keepUserTurns <= 0 {
		keepUserTurns = defaultCompactionKeepUserTurns
	}

	start := 0
	for start < len(messages) && messages[start].Role == "system" && !isCompactionSummary(messages[start]) {
		prefix = append(prefix, messages[start])
		start++
	}

	candidates := make([]api.Message, 0, len(messages)-start)
	for i := start; i < len(messages); i++ {
		msg := messages[i]
		if isCompactionSummary(msg) {
			previousSummary = compactionSummaryText(msg.Content)
			continue
		}
		if isCompactionToolCall(msg) {
			if i+1 < len(messages) && isCompactionSummary(messages[i+1]) {
				previousSummary = compactionSummaryText(messages[i+1].Content)
				i++
			}
			continue
		}
		candidates = append(candidates, msg)
	}

	userTurnIndexes := make([]int, 0, keepUserTurns)
	for i := len(candidates) - 1; i >= 0; i-- {
		if candidates[i].Role == "user" {
			userTurnIndexes = append(userTurnIndexes, i)
		}
	}
	keptUserTurns = keepUserTurns
	if len(userTurnIndexes) <= keptUserTurns {
		keptUserTurns = len(userTurnIndexes) - 1
	}
	if keptUserTurns < 0 {
		keptUserTurns = 0
	}

	suffixStart := len(candidates)
	if keptUserTurns > 0 {
		suffixStart = userTurnIndexes[keptUserTurns-1]
	}
	if suffixStart <= 0 || len(candidates[:suffixStart]) == 0 {
		return prefix, previousSummary, nil, nil, keptUserTurns, false
	}

	return prefix, previousSummary, candidates[:suffixStart], candidates[suffixStart:], keptUserTurns, true
}

func isCompactionSummary(msg api.Message) bool {
	return (msg.Role == "user" || msg.Role == "system" || (msg.Role == "tool" && msg.ToolName == compactionToolName)) &&
		strings.HasPrefix(msg.Content, compactionSummaryMessagePrefix)
}

func isCompactionToolCall(msg api.Message) bool {
	if msg.Role != "assistant" {
		return false
	}
	for _, call := range msg.ToolCalls {
		if call.Function.Name == compactionToolName {
			return true
		}
	}
	return false
}

func compactionSummaryText(content string) string {
	return strings.TrimSpace(strings.TrimPrefix(content, compactionSummaryMessagePrefix))
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
