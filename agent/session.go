package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/google/uuid"

	"github.com/ollama/ollama/api"
)

type ChatClient interface {
	Chat(context.Context, *api.ChatRequest, api.ChatResponseFunc) error
}

type ChatStore interface {
	EnsureChat(context.Context, string, string) error
	AppendMessage(context.Context, string, api.Message, string) error
	UpdateLastMessage(context.Context, string, api.Message, string) error
}

type ChatModelStore interface {
	SetChatModel(context.Context, string, string) error
}

type Session struct {
	Client     ChatClient
	Store      ChatStore
	Events     EventSink
	Tools      *Registry
	Approval   ApprovalHandler
	WorkingDir string
	Compactor  Compactor
}

type RunOptions struct {
	ChatID       string
	Model        string
	SystemPrompt string
	Messages     []api.Message
	NewMessages  []api.Message
	Format       string
	Options      map[string]any
	Think        *api.ThinkValue
	KeepAlive    *api.Duration
	UseTools     bool
	// MaxToolRounds limits consecutive model/tool cycles.
	// Zero uses the default guard; negative disables the guard for tests or
	// special callers.
	MaxToolRounds int
}

type RunResult struct {
	Messages   []api.Message
	Latest     api.ChatResponse
	WorkingDir string
}

const (
	streamPersistDeltaThreshold = 20
	defaultMaxToolRounds        = 100
	maxToolResultRunes          = 60000
)

func (s *Session) Run(ctx context.Context, opts RunOptions) (*RunResult, error) {
	if s == nil {
		return nil, errors.New("nil session")
	}
	if s.Client == nil {
		return nil, errors.New("agent session requires a chat client")
	}
	if opts.Model == "" {
		return nil, errors.New("agent session requires a model")
	}

	runID := uuid.NewString()
	startedAt := time.Now()
	if err := emit(s.Events, Event{Type: EventRunStarted, RunID: runID, ChatID: opts.ChatID, StartedAt: startedAt}); err != nil {
		return nil, err
	}

	if opts.ChatID != "" && s.Store != nil {
		if err := s.Store.EnsureChat(ctx, opts.ChatID, ""); err != nil {
			return nil, err
		}
		if modelStore, ok := s.Store.(ChatModelStore); ok {
			if err := modelStore.SetChatModel(ctx, opts.ChatID, opts.Model); err != nil {
				return nil, err
			}
		}
	}

	messages := make([]api.Message, 0, len(opts.Messages)+len(opts.NewMessages))
	for _, msg := range opts.Messages {
		messages = append(messages, sanitizeMessageForRun(msg))
	}
	for _, msg := range opts.NewMessages {
		msg = sanitizeMessageForRun(msg)
		messages = append(messages, msg)
		if opts.ChatID != "" && s.Store != nil {
			if err := s.Store.AppendMessage(ctx, opts.ChatID, msg, ""); err != nil {
				return nil, err
			}
		}
	}

	if opts.UseTools && len(s.runTools(opts)) == 0 {
		if err := emit(s.Events, Event{Type: EventToolsUnavailable, RunID: runID, ChatID: opts.ChatID}); err != nil {
			return nil, err
		}
	}

	var latest api.ChatResponse
	var consecutiveErrors int
	toolRounds := 0
	maxToolRounds := resolvedMaxToolRounds(opts.MaxToolRounds)
	compactionSkipNotified := false
	for {
		assistant, pendingToolCalls, canceled, err := s.chatRound(ctx, runID, opts, messages, &latest)
		if err != nil {
			var statusErr api.StatusError
			if errors.As(err, &statusErr) && statusErr.StatusCode >= 500 && consecutiveErrors < 2 {
				consecutiveErrors++
				errorMsg := api.Message{
					Role:    "user",
					Content: fmt.Sprintf("Your previous response caused an error: %s\n\nPlease try again with a valid response.", statusErr.ErrorMessage),
				}
				messages = append(messages, errorMsg)
				if opts.ChatID != "" && s.Store != nil {
					if appendErr := s.Store.AppendMessage(ctx, opts.ChatID, errorMsg, ""); appendErr != nil {
						return nil, appendErr
					}
				}
				continue
			}
			emit(s.Events, Event{Type: EventError, RunID: runID, ChatID: opts.ChatID, Error: err.Error()})
			return nil, err
		}
		consecutiveErrors = 0

		if !messageEmpty(assistant) {
			messages = append(messages, assistant)
		}
		if len(pendingToolCalls) == 0 {
			messages, compactionSkipNotified = s.maybeCompact(ctx, opts, messages, latest, compactionSkipNotified)
		}

		if canceled {
			if len(pendingToolCalls) > 0 {
				skipped, skipErr := s.skipToolCalls(context.WithoutCancel(ctx), runID, opts, pendingToolCalls, "Tool execution skipped because the run was canceled.")
				if skipErr != nil {
					emit(s.Events, Event{Type: EventError, RunID: runID, ChatID: opts.ChatID, Error: skipErr.Error()})
					return nil, skipErr
				}
				messages = append(messages, skipped...)
			}
			if err := emit(s.Events, Event{Type: EventRunFinished, RunID: runID, ChatID: opts.ChatID, FinishedAt: time.Now(), Response: &latest}); err != nil {
				return nil, err
			}
			return &RunResult{Messages: messages, Latest: latest, WorkingDir: s.WorkingDir}, nil
		}

		if len(pendingToolCalls) == 0 || !opts.UseTools || s.Tools == nil {
			if err := emit(s.Events, Event{Type: EventRunFinished, RunID: runID, ChatID: opts.ChatID, FinishedAt: time.Now(), Response: &latest}); err != nil {
				return nil, err
			}
			return &RunResult{Messages: messages, Latest: latest, WorkingDir: s.WorkingDir}, nil
		}

		if maxToolRounds >= 0 && toolRounds >= maxToolRounds {
			content := fmt.Sprintf("Tool execution skipped because the max tool-round limit of %d was reached. Send another message to continue.", maxToolRounds)
			toolMessages, skipErr := s.skipToolCalls(ctx, runID, opts, pendingToolCalls, content)
			if skipErr != nil {
				emit(s.Events, Event{Type: EventError, RunID: runID, ChatID: opts.ChatID, Error: skipErr.Error()})
				return nil, skipErr
			}
			messages = append(messages, toolMessages...)
			err := fmt.Errorf("tool round limit reached after %d rounds; send another message to continue", maxToolRounds)
			emit(s.Events, Event{Type: EventError, RunID: runID, ChatID: opts.ChatID, Error: err.Error()})
			return &RunResult{Messages: messages, Latest: latest, WorkingDir: s.WorkingDir}, err
		}

		toolMessages, denied, err := s.executeToolCalls(ctx, runID, opts, messages, pendingToolCalls)
		if err != nil {
			emit(s.Events, Event{Type: EventError, RunID: runID, ChatID: opts.ChatID, Error: err.Error()})
			return nil, err
		}
		messages = append(messages, toolMessages...)
		messages, compactionSkipNotified = s.maybeCompact(ctx, opts, messages, latest, compactionSkipNotified)
		if denied {
			if err := emit(s.Events, Event{Type: EventRunFinished, RunID: runID, ChatID: opts.ChatID, FinishedAt: time.Now(), Response: &latest}); err != nil {
				return nil, err
			}
			return &RunResult{Messages: messages, Latest: latest, WorkingDir: s.WorkingDir}, nil
		}
		toolRounds++
	}
}

func (s *Session) chatRound(ctx context.Context, runID string, opts RunOptions, messages []api.Message, latest *api.ChatResponse) (api.Message, []api.ToolCall, bool, error) {
	format := opts.Format
	if format == "json" {
		format = `"` + format + `"`
	}

	requestMessages := sanitizeMessagesForRequest(messages)
	if strings.TrimSpace(opts.SystemPrompt) != "" {
		requestMessages = make([]api.Message, 0, len(messages)+1)
		requestMessages = append(requestMessages, api.Message{Role: "system", Content: opts.SystemPrompt})
		requestMessages = append(requestMessages, sanitizeMessagesForRequest(messages)...)
	}

	req := &api.ChatRequest{
		Model:    opts.Model,
		Messages: requestMessages,
		Format:   json.RawMessage(format),
		Options:  opts.Options,
		Think:    opts.Think,
	}
	if opts.KeepAlive != nil {
		req.KeepAlive = opts.KeepAlive
	}
	if tools := s.runTools(opts); len(tools) > 0 {
		req.Tools = tools
	}

	assistant := api.Message{Role: "assistant"}
	var started bool
	var persisted bool
	var dirty bool
	var dirtyDeltas int
	var pendingToolCalls []api.ToolCall

	persist := func(persistCtx context.Context, force bool) error {
		if opts.ChatID == "" || s.Store == nil {
			return nil
		}
		if !dirty || messageEmpty(assistant) {
			return nil
		}
		if !force && dirtyDeltas < streamPersistDeltaThreshold {
			return nil
		}
		if !persisted {
			if err := s.Store.AppendMessage(persistCtx, opts.ChatID, assistant, opts.Model); err != nil {
				return err
			}
			persisted = true
			dirty = false
			dirtyDeltas = 0
			return nil
		}
		if err := s.Store.UpdateLastMessage(persistCtx, opts.ChatID, assistant, opts.Model); err != nil {
			return err
		}
		dirty = false
		dirtyDeltas = 0
		return nil
	}

	err := s.Client.Chat(ctx, req, func(response api.ChatResponse) error {
		if response.Message.Role != "" {
			assistant.Role = response.Message.Role
		}

		if response.Message.Content == "" && response.Message.Thinking == "" && len(response.Message.ToolCalls) == 0 {
			*latest = response
			return nil
		}

		if !started {
			if err := emit(s.Events, Event{Type: EventMessageStarted, RunID: runID, ChatID: opts.ChatID}); err != nil {
				return err
			}
			started = true
		}

		if response.Message.Thinking != "" {
			assistant.Thinking += response.Message.Thinking
			if err := emit(s.Events, Event{Type: EventThinkingDelta, RunID: runID, ChatID: opts.ChatID, Thinking: response.Message.Thinking, Response: &response}); err != nil {
				return err
			}
		}

		if response.Message.Content != "" {
			assistant.Content += response.Message.Content
			if err := emit(s.Events, Event{Type: EventMessageDelta, RunID: runID, ChatID: opts.ChatID, Content: response.Message.Content, Response: &response}); err != nil {
				return err
			}
		}

		if len(response.Message.ToolCalls) > 0 {
			assistant.ToolCalls = append(assistant.ToolCalls, response.Message.ToolCalls...)
			pendingToolCalls = append(pendingToolCalls, response.Message.ToolCalls...)
			if err := emit(s.Events, Event{Type: EventToolCallDetected, RunID: runID, ChatID: opts.ChatID, ToolCalls: response.Message.ToolCalls, Response: &response}); err != nil {
				return err
			}
		}

		*latest = response
		dirty = true
		dirtyDeltas++
		return persist(ctx, false)
	})
	if err != nil {
		if isContextCanceledError(ctx, err) {
			if flushErr := persist(context.WithoutCancel(ctx), true); flushErr != nil {
				return assistant, pendingToolCalls, true, flushErr
			}
			return assistant, pendingToolCalls, true, nil
		}
		return assistant, pendingToolCalls, false, err
	}

	if err := persist(ctx, true); err != nil {
		return assistant, pendingToolCalls, false, err
	}

	return assistant, pendingToolCalls, false, nil
}

func (s *Session) executeToolCalls(ctx context.Context, runID string, opts RunOptions, messages []api.Message, calls []api.ToolCall) ([]api.Message, bool, error) {
	approval := s.Approval
	if approval == nil {
		approval = AutoAllowApproval{}
	}

	toolMessages := make([]api.Message, 0, len(calls))
	projectedMessages := append([]api.Message(nil), messages...)
	for i, call := range calls {
		toolName := call.Function.Name
		args := call.Function.Arguments.ToMap()
		tool, ok := s.Tools.Get(toolName)
		if !ok {
			content := fmt.Sprintf("Error: unknown tool: %s", toolName)
			msg := s.toolMessageForContext(toolName, call.ID, content, opts, projectedMessages)
			if err := s.appendToolMessage(ctx, opts.ChatID, msg); err != nil {
				return nil, false, err
			}
			toolMessages = append(toolMessages, msg)
			projectedMessages = append(projectedMessages, msg)
			content = msg.Content
			if emitErr := emit(s.Events, Event{Type: EventToolFinished, RunID: runID, ChatID: opts.ChatID, ToolCallID: call.ID, ToolName: toolName, Args: args, Content: content, Error: fmt.Sprintf("unknown tool: %s", toolName), FinishedAt: time.Now()}); emitErr != nil {
				return nil, false, emitErr
			}
			continue
		}

		approvalRequest := ApprovalRequest{
			ToolCallID: call.ID,
			ToolName:   toolName,
			Args:       args,
			WorkingDir: s.currentWorkingDir(),
		}
		if toolNeedsApproval(ctx, approval, tool, approvalRequest) {
			result, err := approval.Approve(ctx, approvalRequest)
			if err != nil {
				return nil, false, err
			}
			if result.Decision == ApprovalDeny {
				content := result.Reason
				if content == "" {
					content = "Tool execution denied."
				}
				msg := s.toolMessageForContext(toolName, call.ID, content, opts, projectedMessages)
				if err := s.appendToolMessage(ctx, opts.ChatID, msg); err != nil {
					return nil, false, err
				}
				toolMessages = append(toolMessages, msg)
				projectedMessages = append(projectedMessages, msg)
				content = msg.Content
				if emitErr := emit(s.Events, Event{Type: EventToolFinished, RunID: runID, ChatID: opts.ChatID, ToolCallID: call.ID, ToolName: toolName, Args: args, Content: content, Error: content, FinishedAt: time.Now()}); emitErr != nil {
					return nil, false, emitErr
				}
				for _, skipped := range calls[i+1:] {
					skippedToolName := skipped.Function.Name
					skippedArgs := skipped.Function.Arguments.ToMap()
					skippedContent := "Tool execution skipped because a previous tool call in this assistant message was denied."
					skippedMsg := s.toolMessageForContext(skippedToolName, skipped.ID, skippedContent, opts, projectedMessages)
					if err := s.appendToolMessage(ctx, opts.ChatID, skippedMsg); err != nil {
						return nil, false, err
					}
					toolMessages = append(toolMessages, skippedMsg)
					projectedMessages = append(projectedMessages, skippedMsg)
					skippedContent = skippedMsg.Content
					if emitErr := emit(s.Events, Event{Type: EventToolFinished, RunID: runID, ChatID: opts.ChatID, ToolCallID: skipped.ID, ToolName: skippedToolName, Args: skippedArgs, Content: skippedContent, Error: skippedContent, FinishedAt: time.Now()}); emitErr != nil {
						return nil, false, emitErr
					}
				}
				return toolMessages, true, nil
			}
		}

		startedAt := time.Now()
		if err := emit(s.Events, Event{Type: EventToolStarted, RunID: runID, ChatID: opts.ChatID, ToolCallID: call.ID, ToolName: toolName, WorkingDir: s.currentWorkingDir(), Args: args, StartedAt: startedAt}); err != nil {
			return nil, false, err
		}

		result, err := s.Tools.Execute(ctx, s.toolContext(), call)
		if err != nil {
			content := fmt.Sprintf("Error: %v", err)
			msg := s.toolMessageForContext(toolName, call.ID, content, opts, projectedMessages)
			if appendErr := s.appendToolMessage(ctx, opts.ChatID, msg); appendErr != nil {
				return nil, false, appendErr
			}
			toolMessages = append(toolMessages, msg)
			projectedMessages = append(projectedMessages, msg)
			content = msg.Content
			if emitErr := emit(s.Events, Event{Type: EventToolFinished, RunID: runID, ChatID: opts.ChatID, ToolCallID: call.ID, ToolName: toolName, Args: args, Content: content, Error: err.Error(), FinishedAt: time.Now()}); emitErr != nil {
				return nil, false, emitErr
			}
			continue
		}

		s.applyToolWorkingDir(result.WorkingDir)
		content := result.Content

		msg := s.toolMessageForContext(toolName, call.ID, content, opts, projectedMessages)
		if err := s.appendToolMessage(ctx, opts.ChatID, msg); err != nil {
			return nil, false, err
		}
		toolMessages = append(toolMessages, msg)
		projectedMessages = append(projectedMessages, msg)
		content = msg.Content

		if err := emit(s.Events, Event{Type: EventToolFinished, RunID: runID, ChatID: opts.ChatID, ToolCallID: call.ID, ToolName: toolName, WorkingDir: s.WorkingDir, Args: args, Content: content, FinishedAt: time.Now()}); err != nil {
			return nil, false, err
		}
	}
	return toolMessages, false, nil
}

func (s *Session) skipToolCalls(ctx context.Context, runID string, opts RunOptions, calls []api.ToolCall, content string) ([]api.Message, error) {
	toolMessages := make([]api.Message, 0, len(calls))
	for _, call := range calls {
		toolName := call.Function.Name
		args := call.Function.Arguments.ToMap()
		msg := toolMessage(toolName, call.ID, content)
		if err := s.appendToolMessage(ctx, opts.ChatID, msg); err != nil {
			return nil, err
		}
		toolMessages = append(toolMessages, msg)
		if emitErr := emit(s.Events, Event{Type: EventToolFinished, RunID: runID, ChatID: opts.ChatID, ToolCallID: call.ID, ToolName: toolName, Args: args, Content: msg.Content, Error: msg.Content, FinishedAt: time.Now()}); emitErr != nil {
			return nil, emitErr
		}
	}
	return toolMessages, nil
}

func (s *Session) toolContext() ToolContext {
	return ToolContext{
		WorkingDir: s.currentWorkingDir(),
	}
}

func (s *Session) currentWorkingDir() string {
	if s.WorkingDir != "" {
		return s.WorkingDir
	}
	wd, err := os.Getwd()
	if err != nil {
		return ""
	}
	s.WorkingDir = wd
	return s.WorkingDir
}

func (s *Session) applyToolWorkingDir(next string) bool {
	next = strings.TrimSpace(next)
	if next == "" {
		return false
	}
	current := s.currentWorkingDir()
	nextAbs, err := canonicalSessionPath(next)
	if err != nil {
		return false
	}
	if current == nextAbs {
		return false
	}
	s.WorkingDir = nextAbs
	return true
}

func canonicalSessionPath(path string) (string, error) {
	abs, err := filepath.Abs(path)
	if err != nil {
		return "", err
	}
	resolved, err := filepath.EvalSymlinks(abs)
	if err == nil {
		return resolved, nil
	}
	return abs, nil
}

func isContextCanceledError(ctx context.Context, err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, context.Canceled) {
		return true
	}
	return ctx != nil && errors.Is(ctx.Err(), context.Canceled) && strings.Contains(err.Error(), "context canceled")
}

func toolNeedsApproval(ctx context.Context, approval ApprovalHandler, tool Tool, req ApprovalRequest) bool {
	return approval.RequiresApproval(ctx, tool, req)
}

func (s *Session) maybeCompact(ctx context.Context, opts RunOptions, messages []api.Message, latest api.ChatResponse, skipNotified bool) ([]api.Message, bool) {
	if s.Compactor == nil {
		return messages, skipNotified
	}
	req := CompactionRequest{
		ChatID:       opts.ChatID,
		Model:        opts.Model,
		SystemPrompt: opts.SystemPrompt,
		Messages:     messages,
		Tools:        s.runTools(opts),
		Format:       opts.Format,
		Latest:       latest,
		Options:      opts.Options,
		KeepAlive:    opts.KeepAlive,
		ContinueTask: true,
		Progress: func(progress CompactionProgress) {
			_ = emit(s.Events, Event{Type: EventCompactionProgress, ChatID: opts.ChatID, Tokens: progress.Tokens})
		},
	}
	if s.autoCompactionDue(req) {
		_ = emit(s.Events, Event{Type: EventCompactionStarted, ChatID: opts.ChatID, StartedAt: time.Now()})
	}
	result, err := s.Compactor.MaybeCompact(ctx, req)
	if err != nil {
		if result.Due && !skipNotified {
			emit(s.Events, Event{Type: EventCompactionSkipped, ChatID: opts.ChatID, Content: CompactionSkippedMessage(result.Reason)})
			skipNotified = true
		}
		return messages, skipNotified
	}
	if !result.Compacted {
		if result.Due && !skipNotified {
			emit(s.Events, Event{Type: EventCompactionSkipped, ChatID: opts.ChatID, Content: CompactionSkippedMessage(result.Reason)})
			skipNotified = true
		}
		return messages, skipNotified
	}
	emit(s.Events, Event{Type: EventCompacted, ChatID: opts.ChatID, Content: result.Summary, Messages: result.Messages})
	return result.Messages, skipNotified
}

func (s *Session) autoCompactionDue(req CompactionRequest) bool {
	if compactor, ok := s.Compactor.(*SimpleCompactor); ok && compactor != nil {
		return req.Force || compactor.shouldCompact(req)
	}
	return false
}

func CompactionSkippedMessage(reason string) string {
	reason = strings.TrimSpace(reason)
	if reason == "" {
		reason = "compaction could not run"
	}
	return reason
}

func (s *Session) appendToolMessage(ctx context.Context, chatID string, msg api.Message) error {
	if chatID == "" || s.Store == nil {
		return nil
	}
	return s.Store.AppendMessage(ctx, chatID, msg, "")
}

func resolvedMaxToolRounds(value int) int {
	if value == 0 {
		return defaultMaxToolRounds
	}
	return value
}

func (s *Session) toolMessageForContext(toolName, toolCallID, content string, opts RunOptions, messages []api.Message) api.Message {
	msg := toolMessage(toolName, toolCallID, content)
	threshold := s.compactionThresholdTokens(opts)
	if threshold <= 0 {
		return msg
	}

	projected := append(append([]api.Message(nil), messages...), msg)
	projectedTokens := s.estimateRunPromptTokens(opts, projected)
	if projectedTokens < threshold {
		return msg
	}

	baseTokens := s.estimateRunPromptTokens(opts, messages)
	overheadTokens := estimateMessagesTokens([]api.Message{{
		Role:       "tool",
		ToolName:   toolName,
		ToolCallID: toolCallID,
	}})
	availableRunes := (threshold - baseTokens - overheadTokens) * 4
	maxRunes := min(maxToolResultRunes, max(0, availableRunes))
	msg.Content = truncateToolResultContentTo(content, maxRunes)
	return msg
}

func (s *Session) runTools(opts RunOptions) api.Tools {
	if !opts.UseTools || s.Tools == nil {
		return nil
	}
	return s.Tools.Tools()
}

func (s *Session) estimateRunPromptTokens(opts RunOptions, messages []api.Message) int {
	return estimateCompactionRequestTokens(CompactionRequest{
		SystemPrompt: opts.SystemPrompt,
		Messages:     sanitizeMessagesForRequest(messages),
		Tools:        s.runTools(opts),
		Format:       opts.Format,
		Options:      opts.Options,
	})
}

func (s *Session) compactionThresholdTokens(opts RunOptions) int {
	if s.Compactor == nil {
		return 0
	}

	configuredWindow := 0
	configuredThreshold := 0.0
	if compactor, ok := s.Compactor.(*SimpleCompactor); ok && compactor != nil {
		configuredWindow = compactor.Options.ContextWindowTokens
		configuredThreshold = compactor.Options.Threshold
	}

	contextWindow := ResolveContextWindowTokens(opts.Options, configuredWindow)
	threshold := int(float64(contextWindow) * ResolveCompactionThreshold(configuredThreshold))
	if threshold <= 0 {
		return 0
	}
	return threshold
}

func toolMessage(toolName, toolCallID, content string) api.Message {
	return api.Message{
		Role:       "tool",
		Content:    truncateToolResultContent(content),
		ToolName:   toolName,
		ToolCallID: toolCallID,
	}
}

func sanitizeMessageForRun(msg api.Message) api.Message {
	if msg.Role == "tool" {
		msg.Content = truncateToolResultContent(msg.Content)
	}
	return msg
}

func sanitizeMessagesForRequest(messages []api.Message) []api.Message {
	if len(messages) == 0 {
		return nil
	}
	sanitized := make([]api.Message, len(messages))
	for i, msg := range messages {
		sanitized[i] = sanitizeMessageForRequest(msg)
	}
	return sanitized
}

func sanitizeMessageForRequest(msg api.Message) api.Message {
	return sanitizeMessageForRun(msg)
}

func truncateToolResultContent(content string) string {
	return truncateToolResultContentTo(content, maxToolResultRunes)
}

func truncateToolResultContentTo(content string, maxRunes int) string {
	if strings.Contains(content, "[tool output truncated: omitted ") {
		return content
	}
	if maxRunes <= 0 {
		return fmt.Sprintf("[tool output truncated: omitted %d characters]", len([]rune(content)))
	}
	if len(content) <= maxRunes {
		return content
	}
	runes := []rune(content)
	if len(runes) <= maxRunes {
		return content
	}
	head := maxRunes * 3 / 4
	tail := maxRunes - head
	omitted := len(runes) - head - tail
	// TODO(parthsareen): Allow the model to page through full tool output or
	// request specific ranges while staying aware of the available context.
	marker := fmt.Sprintf("\n\n[tool output truncated: omitted %d characters]\n\n", omitted)
	return string(runes[:head]) + marker + string(runes[len(runes)-tail:])
}

func messageEmpty(msg api.Message) bool {
	return msg.Content == "" && msg.Thinking == "" && len(msg.ToolCalls) == 0
}
