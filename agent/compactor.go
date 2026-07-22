package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"

	"github.com/ollama/ollama/api"
)

// Compaction wire-format. These constants and helpers are the single canonical
// definition of how a compacted turn is represented in message history.
const (
	CompactionSummaryMessagePrefix = "Conversation summary:\n"
	CompactionToolName             = "summary"
	CompactionToolCallID           = "ollama_compaction"
	CompactionContinueInstruction  = "continue the task in progress. the history has been compacted, do not mention compaction to the user"
)

const (
	defaultCompactionContextWindowTokens = 32768
	defaultCompactionKeepUserTurns       = 3
	defaultCompactionThreshold           = 0.8
	compactOnlySummaryContextTokens      = 16000

	maxCompactionSummaryRunes = 16 * 1024

	compactionSystemPrompt = "Summarize the archived part of an Ollama agent conversation. Preserve user goals, decisions, files, commands, tool results, and unresolved tasks needed to continue. Omit private reasoning and return only the summary."
)

type Compactor interface {
	MaybeCompact(context.Context, CompactionRequest) (CompactionResult, error)

	// ContextWindowTokens returns the effective context window size in
	// tokens, resolving runtime options against configured defaults.
	ContextWindowTokens(options map[string]any) int

	// Threshold returns the compaction threshold as a fraction of the
	// context window (e.g. 0.8 means compact at 80% capacity).
	Threshold() float64

	// ShouldCompact reports whether a compaction should run and returns the
	// trigger reason. An empty trigger means compaction is not needed.
	ShouldCompact(req CompactionRequest) (trigger string, should bool)
}

type CompactionOptions struct {
	ContextWindowTokens int
	KeepUserTurns       int
	Threshold           float64
}

type CompactionRequest struct {
	ChatID        string
	Model         string
	SystemPrompt  string
	Messages      []api.Message
	Tools         api.Tools
	Format        string
	Latest        api.ChatResponse
	Options       map[string]any
	KeepAlive     *api.Duration
	Think         *api.ThinkValue
	Force         bool
	ContinueTask  bool
	KeepUserTurns *int
	Progress      func(CompactionProgress)
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
	Options CompactionOptions
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

	keepUserTurns := c.keepUserTurns(req.Options)
	if req.KeepUserTurns != nil {
		keepUserTurns = *req.KeepUserTurns
	}
	prefix, previousSummary, archive, suffix, _, ok := splitCompactionMessages(req.Messages, keepUserTurns)
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
		summary, err = c.summarizeEmptyFallback(ctx, req, previousSummary, archive)
		if err != nil {
			result.Reason = err.Error()
			return result, err
		}
		summary = truncateCompactionSummary(strings.TrimSpace(summary))
	}
	if summary == "" {
		result.Reason = "summary was empty"
		return result, nil
	}

	compacted := make([]api.Message, 0, len(prefix)+len(suffix)+2)
	compacted = append(compacted, prefix...)
	compacted = append(compacted, CompactionSummaryMessages(summary, req.ContinueTask)...)
	compacted = append(compacted, suffix...)
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
	return estimateCompactionRequestTokens(req) >= threshold
}

func (c *SimpleCompactor) contextWindowTokens(options map[string]any) int {
	return ResolveContextWindowTokens(options, c.Options.ContextWindowTokens)
}

// ContextWindowTokens resolves the effective context window from runtime
// options or configured defaults. Satisfies the Compactor interface.
func (c *SimpleCompactor) ContextWindowTokens(options map[string]any) int {
	if c == nil {
		return 0
	}
	return c.contextWindowTokens(options)
}

func (c *SimpleCompactor) threshold() float64 {
	return ResolveCompactionThreshold(c.Options.Threshold)
}

// Threshold returns the configured compaction threshold fraction. Satisfies
// the Compactor interface.
func (c *SimpleCompactor) Threshold() float64 {
	if c == nil {
		return 0
	}
	return c.threshold()
}

// ShouldCompact reports whether compaction is due and the trigger reason.
// Satisfies the Compactor interface.
func (c *SimpleCompactor) ShouldCompact(req CompactionRequest) (string, bool) {
	if c == nil {
		return "", false
	}
	if req.Force {
		return "force", true
	}
	if c.shouldCompact(req) {
		contextWindow := c.contextWindowTokens(req.Options)
		threshold := int(float64(contextWindow) * c.threshold())
		if req.Latest.PromptEvalCount > 0 && req.Latest.PromptEvalCount >= threshold {
			return "prompt_eval", true
		}
		return "estimate", true
	}
	return "", false
}

func (c *SimpleCompactor) keepUserTurns(options map[string]any) int {
	contextWindow := c.contextWindowTokens(options)
	if contextWindow > 0 && contextWindow < compactOnlySummaryContextTokens {
		return 0
	}
	if c.Options.KeepUserTurns > 0 {
		return c.Options.KeepUserTurns
	}
	return defaultCompactionKeepUserTurns
}

func ResolveContextWindowTokens(options map[string]any, configured int) int {
	if n := intOption(options, "num_ctx"); n > 0 {
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
		Think:   req.Think,
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

func (c *SimpleCompactor) summarizeEmptyFallback(ctx context.Context, req CompactionRequest, previousSummary string, archive []api.Message) (string, error) {
	retry := req
	retry.Think = &api.ThinkValue{Value: false}
	summary, err := c.summarize(ctx, retry, previousSummary, archive)
	if err == nil {
		return summary, nil
	}
	if !isUnsupportedCompactionThinkError(err) {
		return "", err
	}
	if req.Think == nil {
		return "", nil
	}
	retry.Think = nil
	return c.summarize(ctx, retry, previousSummary, archive)
}

func isUnsupportedCompactionThinkError(err error) bool {
	if err == nil {
		return false
	}
	text := strings.ToLower(err.Error())
	if !strings.Contains(text, "think") {
		return false
	}
	var statusErr api.StatusError
	if errors.As(err, &statusErr) && statusErr.StatusCode != 0 {
		return statusErr.StatusCode == http.StatusBadRequest
	}
	return strings.Contains(text, "does not support") || strings.Contains(text, "not supported") || strings.Contains(text, "unsupported")
}

// compactionSummaryMessageForTask renders a compaction summary as the content
// string stored on the synthetic tool-result message.
func compactionSummaryMessageForTask(summary string, continueTask bool) string {
	content := CompactionSummaryMessagePrefix + strings.TrimSpace(summary)
	if continueTask {
		content = strings.TrimSpace(content) + "\n\n" + CompactionContinueInstruction
	}
	return content
}

// CompactionSummaryMessages renders a compaction summary as the assistant
// tool-call plus tool-result pair that represents a compacted turn in the
// message history.
func CompactionSummaryMessages(summary string, continueTask bool) []api.Message {
	return []api.Message{
		{
			Role: "assistant",
			ToolCalls: []api.ToolCall{{
				ID: CompactionToolCallID,
				Function: api.ToolCallFunction{
					Name: CompactionToolName,
				},
			}},
		},
		{
			Role:       "tool",
			ToolName:   CompactionToolName,
			ToolCallID: CompactionToolCallID,
			Content:    compactionSummaryMessageForTask(summary, continueTask),
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
	return Truncate(summary, TruncateConfig{
		MaxRunes: maxCompactionSummaryRunes,
		Label:    "summary",
	})
}

func estimateCompactionTokens(text string) int {
	text = strings.TrimSpace(text)
	if text == "" {
		return 0
	}
	return ApproximateTokens(len([]rune(text)))
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
	requestMessages := sanitizeMessagesForEstimate(req.Messages)
	if strings.TrimSpace(req.SystemPrompt) != "" {
		requestMessages = make([]api.Message, 0, len(req.Messages)+1)
		requestMessages = append(requestMessages, api.Message{Role: "system", Content: strings.TrimSpace(req.SystemPrompt)})
		requestMessages = append(requestMessages, sanitizeMessagesForEstimate(req.Messages)...)
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

func (s *Session) estimateRunPromptTokens(opts RunOptions, messages []api.Message) int {
	return estimateCompactionRequestTokens(CompactionRequest{
		SystemPrompt: opts.SystemPrompt,
		Messages:     messages,
		Tools:        s.availableTools(),
		Format:       opts.Format,
		Options:      opts.Options,
	})
}

func (s *Session) checkPreflightPromptBudget(opts RunOptions, messages []api.Message) error {
	contextWindow := s.contextWindowTokens(opts)
	if contextWindow <= 0 {
		return nil
	}
	estimated := s.estimateRunPromptTokens(opts, messages)
	if estimated < contextWindow {
		return nil
	}
	return fmt.Errorf("prompt is too large for the current context (~%d/%d tokens). Reduce the system prompt or message history, compact the conversation, or use a model with a larger context", estimated, contextWindow)
}

func (s *Session) checkPostCompactionPromptBudget(opts RunOptions, messages []api.Message) error {
	contextWindow := s.contextWindowTokens(opts)
	if contextWindow <= 0 {
		return nil
	}
	estimated := s.estimateRunPromptTokens(opts, messages)
	if estimated < contextWindow {
		return nil
	}
	return fmt.Errorf("history is still too large after compaction (~%d/%d tokens). Start a fresh request, reduce the system prompt or history, or use a model with a larger context", estimated, contextWindow)
}

func sanitizeMessagesForEstimate(messages []api.Message) []api.Message {
	requestMessages := sanitizeMessagesForRequest(messages)
	for i := range requestMessages {
		// Image token accounting is model-specific. Without the active model's
		// tokenizer and vision accounting, raw image bytes/base64 make the
		// estimate look much larger than the prompt the model actually sees.
		requestMessages[i].Images = nil
	}
	return requestMessages
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
	if keepUserTurns < 0 {
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
			previousSummary = CompactionSummaryText(msg.Content)
			continue
		}
		if isCompactionToolCall(msg) {
			if i+1 < len(messages) && isCompactionSummary(messages[i+1]) {
				previousSummary = CompactionSummaryText(messages[i+1].Content)
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

func isCompactionToolName(name string) bool {
	return name == CompactionToolName
}

func isCompactionSummary(msg api.Message) bool {
	return (msg.Role == "user" || msg.Role == "system" || (msg.Role == "tool" && isCompactionToolName(msg.ToolName))) &&
		strings.HasPrefix(msg.Content, CompactionSummaryMessagePrefix)
}

// IsCompactionSummary reports whether msg uses the canonical compaction
// summary message representation.
func IsCompactionSummary(msg api.Message) bool {
	return isCompactionSummary(msg)
}

// CompactionSummaryContent returns the user-visible summary from msg when it
// is a canonical compaction summary.
func CompactionSummaryContent(msg api.Message) (string, bool) {
	if !isCompactionSummary(msg) {
		return "", false
	}
	return CompactionSummaryText(msg.Content), true
}

// IsCompactionToolResult reports whether msg is the synthetic tool result used
// to represent compaction in message history.
func IsCompactionToolResult(msg api.Message) bool {
	return msg.Role == "tool" && (isCompactionToolName(msg.ToolName) || msg.ToolCallID == CompactionToolCallID)
}

// IsCompactionToolCall reports whether msg is the synthetic assistant tool
// call paired with a compaction summary result.
func IsCompactionToolCall(msg api.Message) bool {
	return isCompactionToolCall(msg)
}

func isCompactionToolCall(msg api.Message) bool {
	if msg.Role != "assistant" {
		return false
	}
	for _, call := range msg.ToolCalls {
		if isCompactionToolName(call.Function.Name) {
			return true
		}
	}
	return false
}

// CompactionSummaryText reverses CompactionSummaryMessages, returning the
// user-visible summary text with the prefix and any continuation instruction
// removed.
func CompactionSummaryText(content string) string {
	return strings.TrimSpace(strings.TrimSuffix(
		strings.TrimSpace(strings.TrimPrefix(content, CompactionSummaryMessagePrefix)),
		CompactionContinueInstruction,
	))
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
