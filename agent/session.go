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
	AppendMessage(context.Context, string, api.Message) error
	UpdateLastMessage(context.Context, string, api.Message) error
}

type ModelAwareChatStore interface {
	AppendMessageWithModel(context.Context, string, api.Message, string) error
	UpdateLastMessageWithModel(context.Context, string, api.Message, string) error
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
	// MaxToolRounds is an optional guard for tests or callers that need one.
	// Zero means unlimited.
	MaxToolRounds int
}

type RunResult struct {
	Messages   []api.Message
	Latest     api.ChatResponse
	WorkingDir string
}

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
	}

	messages := make([]api.Message, 0, len(opts.Messages)+len(opts.NewMessages))
	messages = append(messages, opts.Messages...)
	for _, msg := range opts.NewMessages {
		messages = append(messages, msg)
		if opts.ChatID != "" && s.Store != nil {
			if err := s.appendStoreMessage(ctx, opts.ChatID, msg, ""); err != nil {
				return nil, err
			}
		}
	}

	if opts.UseTools && (s.Tools == nil || len(s.Tools.Tools()) == 0) {
		if err := emit(s.Events, Event{Type: EventToolsUnavailable, RunID: runID, ChatID: opts.ChatID}); err != nil {
			return nil, err
		}
	}

	var latest api.ChatResponse
	var consecutiveErrors int
	toolRounds := 0
	compactionSkipNotified := false
	for {
		assistant, pendingToolCalls, err := s.chatRound(ctx, runID, opts, messages, &latest)
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
					if appendErr := s.appendStoreMessage(ctx, opts.ChatID, errorMsg, ""); appendErr != nil {
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
		messages, compactionSkipNotified = s.maybeCompact(ctx, opts, messages, latest, compactionSkipNotified)

		if len(pendingToolCalls) == 0 || !opts.UseTools || s.Tools == nil {
			if err := emit(s.Events, Event{Type: EventRunFinished, RunID: runID, ChatID: opts.ChatID, FinishedAt: time.Now(), Response: &latest}); err != nil {
				return nil, err
			}
			return &RunResult{Messages: messages, Latest: latest, WorkingDir: s.WorkingDir}, nil
		}

		if opts.MaxToolRounds > 0 && toolRounds >= opts.MaxToolRounds {
			break
		}

		toolMessages, denied, err := s.executeToolCalls(ctx, runID, opts, pendingToolCalls)
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

	err := fmt.Errorf("agent stopped after %d tool rounds", opts.MaxToolRounds)
	emit(s.Events, Event{Type: EventError, RunID: runID, ChatID: opts.ChatID, Error: err.Error()})
	return nil, err
}

func (s *Session) chatRound(ctx context.Context, runID string, opts RunOptions, messages []api.Message, latest *api.ChatResponse) (api.Message, []api.ToolCall, error) {
	format := opts.Format
	if format == "json" {
		format = `"` + format + `"`
	}

	requestMessages := messages
	if strings.TrimSpace(opts.SystemPrompt) != "" {
		requestMessages = make([]api.Message, 0, len(messages)+1)
		requestMessages = append(requestMessages, api.Message{Role: "system", Content: opts.SystemPrompt})
		requestMessages = append(requestMessages, messages...)
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
	if opts.UseTools && s.Tools != nil {
		if tools := s.Tools.Tools(); len(tools) > 0 {
			req.Tools = tools
		}
	}

	assistant := api.Message{Role: "assistant"}
	var persisted bool
	var pendingToolCalls []api.ToolCall

	persist := func() error {
		if opts.ChatID == "" || s.Store == nil {
			return nil
		}
		if !persisted {
			if err := s.appendStoreMessage(ctx, opts.ChatID, assistant, opts.Model); err != nil {
				return err
			}
			persisted = true
			return nil
		}
		return s.updateStoreLastMessage(ctx, opts.ChatID, assistant, opts.Model)
	}

	err := s.Client.Chat(ctx, req, func(response api.ChatResponse) error {
		*latest = response
		if response.Message.Role != "" {
			assistant.Role = response.Message.Role
		}

		if response.Message.Content == "" && response.Message.Thinking == "" && len(response.Message.ToolCalls) == 0 {
			return nil
		}

		if !persisted {
			if err := emit(s.Events, Event{Type: EventMessageStarted, RunID: runID, ChatID: opts.ChatID}); err != nil {
				return err
			}
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

		return persist()
	})
	if err != nil {
		if errors.Is(err, context.Canceled) {
			return assistant, pendingToolCalls, nil
		}
		return assistant, pendingToolCalls, err
	}

	return assistant, pendingToolCalls, nil
}

func (s *Session) executeToolCalls(ctx context.Context, runID string, opts RunOptions, calls []api.ToolCall) ([]api.Message, bool, error) {
	approval := s.Approval
	if approval == nil {
		approval = AutoAllowApproval{}
	}

	toolMessages := make([]api.Message, 0, len(calls))
	for _, call := range calls {
		toolName := call.Function.Name
		args := call.Function.Arguments.ToMap()
		tool, ok := s.Tools.Get(toolName)
		if !ok {
			content := fmt.Sprintf("Error: unknown tool: %s", toolName)
			msg := api.Message{Role: "tool", Content: content, ToolName: toolName, ToolCallID: call.ID}
			if err := s.appendToolMessage(ctx, opts.ChatID, msg); err != nil {
				return nil, false, err
			}
			toolMessages = append(toolMessages, msg)
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
				msg := api.Message{Role: "tool", Content: content, ToolName: toolName, ToolCallID: call.ID}
				if err := s.appendToolMessage(ctx, opts.ChatID, msg); err != nil {
					return nil, false, err
				}
				toolMessages = append(toolMessages, msg)
				if emitErr := emit(s.Events, Event{Type: EventToolFinished, RunID: runID, ChatID: opts.ChatID, ToolCallID: call.ID, ToolName: toolName, Args: args, Content: content, Error: content, FinishedAt: time.Now()}); emitErr != nil {
					return nil, false, emitErr
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
			msg := api.Message{Role: "tool", Content: content, ToolName: toolName, ToolCallID: call.ID}
			if appendErr := s.appendToolMessage(ctx, opts.ChatID, msg); appendErr != nil {
				return nil, false, appendErr
			}
			toolMessages = append(toolMessages, msg)
			if emitErr := emit(s.Events, Event{Type: EventToolFinished, RunID: runID, ChatID: opts.ChatID, ToolCallID: call.ID, ToolName: toolName, Args: args, Content: content, Error: err.Error(), FinishedAt: time.Now()}); emitErr != nil {
				return nil, false, emitErr
			}
			continue
		}

		s.applyToolWorkingDir(result.WorkingDir)
		content := result.Content

		msg := api.Message{Role: "tool", Content: content, ToolName: toolName, ToolCallID: call.ID}
		if err := s.appendToolMessage(ctx, opts.ChatID, msg); err != nil {
			return nil, false, err
		}
		toolMessages = append(toolMessages, msg)

		if err := emit(s.Events, Event{Type: EventToolFinished, RunID: runID, ChatID: opts.ChatID, ToolCallID: call.ID, ToolName: toolName, WorkingDir: s.WorkingDir, Args: args, Content: content, FinishedAt: time.Now()}); err != nil {
			return nil, false, err
		}
	}
	return toolMessages, false, nil
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

func toolNeedsApproval(ctx context.Context, approval ApprovalHandler, tool Tool, req ApprovalRequest) bool {
	return approval.RequiresApproval(ctx, tool, req)
}

func (s *Session) maybeCompact(ctx context.Context, opts RunOptions, messages []api.Message, latest api.ChatResponse, skipNotified bool) ([]api.Message, bool) {
	if s.Compactor == nil {
		return messages, skipNotified
	}
	result, err := s.Compactor.MaybeCompact(ctx, CompactionRequest{
		ChatID:    opts.ChatID,
		Model:     opts.Model,
		Messages:  messages,
		Latest:    latest,
		Options:   opts.Options,
		KeepAlive: opts.KeepAlive,
	})
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
	emit(s.Events, Event{Type: EventCompacted, ChatID: opts.ChatID, Content: result.Summary})
	return result.Messages, skipNotified
}

func CompactionSkippedMessage(reason string) string {
	reason = strings.TrimSpace(reason)
	if reason == "" {
		reason = "compaction could not run"
	}
	return fmt.Sprintf("%s. Start a new chat with /new if you need a fresh context.", reason)
}

func (s *Session) appendToolMessage(ctx context.Context, chatID string, msg api.Message) error {
	if chatID == "" || s.Store == nil {
		return nil
	}
	return s.Store.AppendMessage(ctx, chatID, msg)
}

func (s *Session) appendStoreMessage(ctx context.Context, chatID string, msg api.Message, model string) error {
	if chatID == "" || s.Store == nil {
		return nil
	}
	if modelStore, ok := s.Store.(ModelAwareChatStore); ok {
		return modelStore.AppendMessageWithModel(ctx, chatID, msg, model)
	}
	return s.Store.AppendMessage(ctx, chatID, msg)
}

func (s *Session) updateStoreLastMessage(ctx context.Context, chatID string, msg api.Message, model string) error {
	if chatID == "" || s.Store == nil {
		return nil
	}
	if modelStore, ok := s.Store.(ModelAwareChatStore); ok {
		return modelStore.UpdateLastMessageWithModel(ctx, chatID, msg, model)
	}
	return s.Store.UpdateLastMessage(ctx, chatID, msg)
}

func messageEmpty(msg api.Message) bool {
	return msg.Content == "" && msg.Thinking == "" && len(msg.ToolCalls) == 0
}
