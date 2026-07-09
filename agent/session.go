package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/google/uuid"

	"github.com/ollama/ollama/api"
)

type ChatClient interface {
	Chat(context.Context, *api.ChatRequest, api.ChatResponseFunc) error
}

type ApprovalRequest struct {
	WorkingDir string
	Calls      []ApprovalToolCall
}

type ApprovalToolCall struct {
	ToolCallID string
	ToolName   string
	Args       map[string]any
}

type Approval struct {
	Allow    bool
	AllowAll bool
	Reason   string
}

type ApprovalPrompter interface {
	PromptApproval(context.Context, ApprovalRequest) (Approval, error)
}

type Session struct {
	Client           ChatClient
	EventSinks       []EventSink
	Tools            *Registry
	ApprovalPrompter ApprovalPrompter
	AllowAllTools    bool
	WorkingDir       string
	Compactor        Compactor
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
	defaultMaxToolRounds              = 100
	maxToolResultRunes                = 60000
	smallContextToolResultRunes       = 6000
	tinyContextToolResultRunes        = 3200
	smallContextToolResultTokenWindow = 8192
	tinyContextToolResultTokenWindow  = 4096
	toolTruncationMarkerReserveTokens = 64
	toolOutputFullOmissionPrefix      = "[tool output truncated: output omitted because the context is full;"
)

type toolOutputOverflow struct {
	toolName   string
	toolCallID string
	content    string
}

type toolBatchResult struct {
	messages  []api.Message
	stop      toolExecutionStop
	overflows []toolOutputOverflow
}

type toolExecutionStop string

const (
	toolExecutionDenied   toolExecutionStop = "denied"
	toolExecutionCanceled toolExecutionStop = "canceled"
)

type runPhase int

const (
	runPhaseModel runPhase = iota
	runPhaseTools
	runPhaseCompact
	runPhaseDone
)

type runState struct {
	runID string
	opts  RunOptions

	phase runPhase

	messages []api.Message
	latest   api.ChatResponse

	assistant        api.Message
	pendingToolCalls []api.ToolCall
	canceled         bool

	toolBatch *toolBatchResult

	consecutiveModelErrors int
	toolRounds             int
	maxToolRounds          int
	compactionSkipNotified bool

	finish runFinish
}

type runFinish struct {
	status         string
	ignoreCanceled bool
	err            error
}

func (st *runState) finishDone() {
	st.finish = runFinish{status: "done"}
	st.phase = runPhaseDone
}

func (st *runState) finishDenied() {
	st.finish = runFinish{status: "denied"}
	st.phase = runPhaseDone
}

func (st *runState) finishCanceled() {
	st.finish = runFinish{status: "canceled", ignoreCanceled: true}
	st.phase = runPhaseDone
}

func (st *runState) finishError(err error) {
	st.finish = runFinish{err: err}
	st.phase = runPhaseDone
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
	messages := make([]api.Message, 0, len(opts.Messages)+len(opts.NewMessages))
	for _, msg := range opts.Messages {
		messages = append(messages, sanitizeMessageForRun(msg))
	}
	for _, msg := range opts.NewMessages {
		msg = sanitizeMessageForRun(msg)
		messages = append(messages, msg)
	}

	if err := s.checkPreflightPromptBudget(opts, messages); err != nil {
		s.emit(Event{Type: EventError, RunID: runID, ChatID: opts.ChatID, Model: opts.Model, Error: err.Error()})
		return nil, err
	}

	st := runState{
		runID:         runID,
		opts:          opts,
		phase:         runPhaseModel,
		messages:      messages,
		maxToolRounds: resolvedMaxToolRounds(opts.MaxToolRounds),
	}
	for {
		switch st.phase {
		case runPhaseModel:
			if err := s.runModelStep(ctx, &st); err != nil {
				return nil, err
			}
		case runPhaseTools:
			if err := s.runToolStep(ctx, &st); err != nil {
				return nil, err
			}
		case runPhaseCompact:
			if err := s.runCompactionStep(ctx, &st); err != nil {
				return &RunResult{Messages: st.messages, Latest: st.latest, WorkingDir: s.WorkingDir}, err
			}
		case runPhaseDone:
			return s.finishRun(ctx, &st)
		}
	}
}

func (s *Session) runModelStep(ctx context.Context, st *runState) error {
	opts := st.opts

	assistant, pendingToolCalls, canceled, err := s.chatRound(ctx, st.runID, opts, st.messages, &st.latest)
	if err != nil {
		var statusErr api.StatusError
		if errors.As(err, &statusErr) && statusErr.StatusCode >= 500 && st.consecutiveModelErrors < 2 {
			st.consecutiveModelErrors++
			st.messages = append(st.messages, api.Message{
				Role:    "user",
				Content: fmt.Sprintf("Your previous response caused an error: %s\n\nPlease try again with a valid response.", statusErr.ErrorMessage),
			})
			return nil
		}
		s.emit(Event{Type: EventError, RunID: st.runID, ChatID: opts.ChatID, Model: opts.Model, Error: err.Error()})
		return err
	}
	st.consecutiveModelErrors = 0
	st.assistant = assistant
	st.pendingToolCalls = pendingToolCalls
	st.canceled = canceled

	if !messageEmpty(assistant) {
		st.messages = append(st.messages, assistant)
	}

	if len(pendingToolCalls) == 0 {
		st.toolBatch = nil
		st.phase = runPhaseCompact
		return nil
	}

	if canceled {
		skipped, skipErr := s.skipToolCalls(ctx, st.runID, opts, pendingToolCalls, "Tool execution skipped because the run was canceled.")
		if skipErr != nil {
			s.emit(Event{Type: EventError, RunID: st.runID, ChatID: opts.ChatID, Model: opts.Model, Error: skipErr.Error()})
			return skipErr
		}
		st.messages = append(st.messages, skipped...)
		st.finishCanceled()
		return nil
	}

	if s.Tools == nil {
		st.finishDone()
		return nil
	}

	if st.maxToolRounds >= 0 && st.toolRounds >= st.maxToolRounds {
		content := fmt.Sprintf("Tool execution skipped because the max tool-round limit of %d was reached. Send another message to continue.", st.maxToolRounds)
		toolMessages, skipErr := s.skipToolCalls(ctx, st.runID, opts, pendingToolCalls, content)
		if skipErr != nil {
			s.emit(Event{Type: EventError, RunID: st.runID, ChatID: opts.ChatID, Model: opts.Model, Error: skipErr.Error()})
			return skipErr
		}
		st.messages = append(st.messages, toolMessages...)
		err := fmt.Errorf("tool round limit reached after %d rounds; send another message to continue", st.maxToolRounds)
		s.emit(Event{Type: EventError, RunID: st.runID, ChatID: opts.ChatID, Model: opts.Model, Error: err.Error()})
		st.finishError(err)
		return nil
	}

	st.phase = runPhaseTools
	return nil
}

func (s *Session) runToolStep(ctx context.Context, st *runState) error {
	batch, err := s.executeToolCalls(ctx, st.runID, st.opts, st.messages, st.pendingToolCalls)
	if err != nil {
		s.emit(Event{Type: EventError, RunID: st.runID, ChatID: st.opts.ChatID, Model: st.opts.Model, Error: err.Error()})
		return err
	}

	st.messages = append(st.messages, batch.messages...)
	st.toolBatch = &batch
	st.phase = runPhaseCompact
	return nil
}

func (s *Session) runCompactionStep(ctx context.Context, st *runState) error {
	opts := st.opts
	var err error
	if st.toolBatch != nil && len(st.toolBatch.overflows) > 0 {
		st.messages, st.compactionSkipNotified, err = s.compactForToolOutputOverflow(ctx, st.runID, opts, st.messages, st.latest, st.assistant, st.toolBatch.messages, st.toolBatch.overflows, st.compactionSkipNotified)
	} else {
		st.messages, st.compactionSkipNotified, err = s.maybeCompact(ctx, st.runID, opts, st.messages, st.latest, st.compactionSkipNotified)
	}
	if err != nil {
		s.emit(Event{Type: EventError, RunID: st.runID, ChatID: opts.ChatID, Model: opts.Model, Error: err.Error()})
		return err
	}

	if st.toolBatch == nil {
		if st.canceled {
			st.finishCanceled()
		} else {
			st.finishDone()
		}
		return nil
	}

	switch st.toolBatch.stop {
	case toolExecutionDenied:
		st.finishDenied()
	case toolExecutionCanceled:
		st.finishCanceled()
	default:
		st.toolRounds++
		st.assistant = api.Message{}
		st.pendingToolCalls = nil
		st.toolBatch = nil
		st.phase = runPhaseModel
	}
	return nil
}

func (s *Session) finishRun(ctx context.Context, st *runState) (*RunResult, error) {
	if st.finish.status != "" {
		event := Event{Type: EventRunFinished, RunID: st.runID, ChatID: st.opts.ChatID, Model: st.opts.Model, Status: st.finish.status}
		var err error
		if st.finish.ignoreCanceled {
			err = s.emitIgnoringCanceled(ctx, event)
		} else {
			err = s.emit(event)
		}
		if err != nil {
			return nil, err
		}
	}
	return &RunResult{Messages: st.messages, Latest: st.latest, WorkingDir: s.WorkingDir}, st.finish.err
}

func (s *Session) chatRound(ctx context.Context, runID string, opts RunOptions, messages []api.Message, latest *api.ChatResponse) (api.Message, []api.ToolCall, bool, error) {
	requestMessages := sanitizeMessagesForRequest(messages)
	if strings.TrimSpace(opts.SystemPrompt) != "" {
		withSystem := make([]api.Message, 0, len(requestMessages)+1)
		withSystem = append(withSystem, api.Message{Role: "system", Content: opts.SystemPrompt})
		requestMessages = append(withSystem, requestMessages...)
	}

	format := opts.Format
	if format == "json" {
		format = `"` + format + `"`
	}

	req := api.ChatRequest{
		Model:    opts.Model,
		Messages: requestMessages,
		Format:   json.RawMessage(format),
		Options:  opts.Options,
		Think:    opts.Think,
	}
	if opts.KeepAlive != nil {
		req.KeepAlive = opts.KeepAlive
	}
	if tools := s.availableTools(); len(tools) > 0 {
		req.Tools = tools
	}

	assistant := api.Message{Role: "assistant"}
	var pendingToolCalls []api.ToolCall

	err := s.Client.Chat(ctx, &req, func(response api.ChatResponse) error {
		if response.Message.Role != "" {
			assistant.Role = response.Message.Role
		}

		if response.Message.Content == "" && response.Message.Thinking == "" && len(response.Message.ToolCalls) == 0 {
			*latest = response
			return nil
		}

		if response.Message.Thinking != "" {
			assistant.Thinking += response.Message.Thinking
			if err := s.emit(Event{Type: EventThinkingDelta, RunID: runID, ChatID: opts.ChatID, Model: opts.Model, Thinking: response.Message.Thinking}); err != nil {
				return err
			}
		}

		if response.Message.Content != "" {
			assistant.Content += response.Message.Content
			if err := s.emit(Event{Type: EventMessageDelta, RunID: runID, ChatID: opts.ChatID, Model: opts.Model, Content: response.Message.Content}); err != nil {
				return err
			}
		}

		if len(response.Message.ToolCalls) > 0 {
			assistant.ToolCalls = append(assistant.ToolCalls, response.Message.ToolCalls...)
			pendingToolCalls = append(pendingToolCalls, response.Message.ToolCalls...)
			if err := s.emit(Event{Type: EventToolCallDetected, RunID: runID, ChatID: opts.ChatID, Model: opts.Model, ToolCalls: response.Message.ToolCalls}); err != nil {
				return err
			}
		}

		*latest = response
		return nil
	})
	if err != nil {
		if isContextCanceledError(ctx, err) {
			return assistant, pendingToolCalls, true, nil
		}
		return assistant, pendingToolCalls, false, err
	}

	return assistant, pendingToolCalls, false, nil
}

func (s *Session) executeToolCalls(ctx context.Context, runID string, opts RunOptions, messages []api.Message, calls []api.ToolCall) (toolBatchResult, error) {
	batch := toolBatchResult{
		messages: make([]api.Message, 0, len(calls)),
	}
	projectedMessages := append([]api.Message(nil), messages...)

	type plannedToolCall struct {
		call       api.ToolCall
		tool       Tool
		toolName   string
		args       map[string]any
		workingDir string
	}
	plans := make([]plannedToolCall, 0, len(calls))
	batchWorkingDir := s.currentWorkingDir()
	approvalReq := ApprovalRequest{WorkingDir: batchWorkingDir}
	for _, call := range calls {
		toolName := call.Function.Name
		args := call.Function.Arguments.ToMap()
		tool, ok := s.Tools.Get(toolName)
		plans = append(plans, plannedToolCall{
			call:       call,
			tool:       tool,
			toolName:   toolName,
			args:       args,
			workingDir: batchWorkingDir,
		})
		if ok && ToolRequiresApproval(tool, args) {
			approvalReq.Calls = append(approvalReq.Calls, ApprovalToolCall{
				ToolCallID: call.ID,
				ToolName:   toolName,
				Args:       args,
			})
		}
	}

	if len(approvalReq.Calls) > 0 {
		approvalResult, err := s.authorizeToolCalls(ctx, approvalReq)
		if err != nil {
			if ctx.Err() != nil {
				skipped, skipErr := s.skipToolCalls(ctx, runID, opts, calls, "Tool execution skipped because the run was canceled.")
				if skipErr != nil {
					return toolBatchResult{}, skipErr
				}
				batch.messages = append(batch.messages, skipped...)
				batch.stop = toolExecutionCanceled
				return batch, nil
			}
			return toolBatchResult{}, err
		}
		if !approvalResult.Allow {
			content := approvalResult.Reason
			if content == "" {
				content = "Tool execution denied."
			}
			for _, plan := range plans {
				msg := s.toolMessageForContext(plan.toolName, plan.call.ID, content, opts, projectedMessages)
				batch.messages = append(batch.messages, msg)
				projectedMessages = append(projectedMessages, msg)
				deniedContent := msg.Content
				if emitErr := s.emit(Event{Type: EventToolFinished, RunID: runID, ChatID: opts.ChatID, Model: opts.Model, Status: "denied", ToolCallID: plan.call.ID, ToolName: plan.toolName, Args: plan.args, Content: deniedContent, Error: deniedContent}); emitErr != nil {
					return toolBatchResult{}, emitErr
				}
			}
			batch.stop = toolExecutionDenied
			return batch, nil
		}
	}

	for i, plan := range plans {
		call := plan.call
		toolName := plan.toolName
		args := plan.args
		if ctx.Err() != nil {
			skipped, skipErr := s.skipToolCalls(ctx, runID, opts, calls[i:], "Tool execution skipped because the run was canceled.")
			if skipErr != nil {
				return toolBatchResult{}, skipErr
			}
			batch.messages = append(batch.messages, skipped...)
			batch.stop = toolExecutionCanceled
			return batch, nil
		}
		if plan.tool == nil {
			content := fmt.Sprintf("Error: unknown tool: %s", toolName)
			msg := s.toolMessageForContext(toolName, call.ID, content, opts, projectedMessages)
			batch.messages = append(batch.messages, msg)
			projectedMessages = append(projectedMessages, msg)
			content = msg.Content
			if toolOutputFullyOmitted(content) {
				batch.overflows = append(batch.overflows, toolOutputOverflow{toolName: toolName, toolCallID: call.ID, content: fmt.Sprintf("Error: unknown tool: %s", toolName)})
			}
			if emitErr := s.emit(Event{Type: EventToolFinished, RunID: runID, ChatID: opts.ChatID, Model: opts.Model, Status: "failed", ToolCallID: call.ID, ToolName: toolName, Args: args, Content: content, Error: fmt.Sprintf("unknown tool: %s", toolName)}); emitErr != nil {
				return toolBatchResult{}, emitErr
			}
			continue
		}

		if err := s.emit(Event{Type: EventToolStarted, RunID: runID, ChatID: opts.ChatID, Model: opts.Model, Status: "running", ToolCallID: call.ID, ToolName: toolName, WorkingDir: plan.workingDir, Args: args}); err != nil {
			return toolBatchResult{}, err
		}

		result, err := s.Tools.Execute(ctx, ToolContext{WorkingDir: plan.workingDir}, call)
		if err != nil {
			rawContent := fmt.Sprintf("Error: %v", err)
			msg := s.toolMessageForContext(toolName, call.ID, rawContent, opts, projectedMessages)
			batch.messages = append(batch.messages, msg)
			projectedMessages = append(projectedMessages, msg)
			content := msg.Content
			if toolOutputFullyOmitted(content) {
				batch.overflows = append(batch.overflows, toolOutputOverflow{toolName: toolName, toolCallID: call.ID, content: rawContent})
			}
			if emitErr := s.emitIgnoringCanceled(ctx, Event{Type: EventToolFinished, RunID: runID, ChatID: opts.ChatID, Model: opts.Model, Status: "failed", ToolCallID: call.ID, ToolName: toolName, Args: args, Content: content, Error: err.Error()}); emitErr != nil {
				return toolBatchResult{}, emitErr
			}
			if ctx.Err() != nil {
				skipped, skipErr := s.skipToolCalls(ctx, runID, opts, calls[i+1:], "Tool execution skipped because the run was canceled.")
				if skipErr != nil {
					return toolBatchResult{}, skipErr
				}
				batch.messages = append(batch.messages, skipped...)
				batch.stop = toolExecutionCanceled
				return batch, nil
			}
			continue
		}

		eventWorkingDir := plan.workingDir
		if s.applyToolWorkingDir(result.WorkingDir) {
			eventWorkingDir = s.WorkingDir
		}
		rawContent := result.Content

		msg := s.toolMessageForContext(toolName, call.ID, rawContent, opts, projectedMessages)
		batch.messages = append(batch.messages, msg)
		projectedMessages = append(projectedMessages, msg)
		content := msg.Content

		if toolOutputFullyOmitted(content) {
			batch.overflows = append(batch.overflows, toolOutputOverflow{toolName: toolName, toolCallID: call.ID, content: rawContent})
		}
		if err := s.emitIgnoringCanceled(ctx, Event{Type: EventToolFinished, RunID: runID, ChatID: opts.ChatID, Model: opts.Model, Status: "done", ToolCallID: call.ID, ToolName: toolName, WorkingDir: eventWorkingDir, Args: args, Content: content}); err != nil {
			return toolBatchResult{}, err
		}
		if ctx.Err() != nil {
			skipped, skipErr := s.skipToolCalls(ctx, runID, opts, calls[i+1:], "Tool execution skipped because the run was canceled.")
			if skipErr != nil {
				return toolBatchResult{}, skipErr
			}
			batch.messages = append(batch.messages, skipped...)
			batch.stop = toolExecutionCanceled
			return batch, nil
		}
	}
	return batch, nil
}

func (s *Session) authorizeToolCalls(ctx context.Context, req ApprovalRequest) (Approval, error) {
	if s == nil || s.AllowAllTools || len(req.Calls) == 0 {
		return Approval{Allow: true}, nil
	}
	if s.ApprovalPrompter == nil {
		return Approval{
			Reason: "Tool execution requires approval, but no approval prompter is available.",
		}, nil
	}

	result, err := s.ApprovalPrompter.PromptApproval(ctx, req)
	if err != nil {
		return Approval{}, err
	}
	if result.AllowAll {
		result.Allow = true
		s.AllowAllTools = true
	}
	return result, nil
}

func (s *Session) skipToolCalls(ctx context.Context, runID string, opts RunOptions, calls []api.ToolCall, content string) ([]api.Message, error) {
	toolMessages := make([]api.Message, 0, len(calls))
	for _, call := range calls {
		toolName := call.Function.Name
		args := call.Function.Arguments.ToMap()
		msg := toolMessage(toolName, call.ID, content)
		toolMessages = append(toolMessages, msg)
		if emitErr := s.emitIgnoringCanceled(ctx, Event{Type: EventToolFinished, RunID: runID, ChatID: opts.ChatID, Model: opts.Model, Status: "skipped", ToolCallID: call.ID, ToolName: toolName, Args: args, Content: msg.Content, Error: msg.Content}); emitErr != nil {
			return nil, emitErr
		}
	}
	return toolMessages, nil
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

func (s *Session) maybeCompact(ctx context.Context, runID string, opts RunOptions, messages []api.Message, latest api.ChatResponse, skipNotified bool) ([]api.Message, bool, error) {
	if s.Compactor == nil {
		return messages, skipNotified, nil
	}
	req := s.compactionRequest(runID, opts, messages, latest)
	trigger := s.autoCompactionTrigger(req)
	if trigger != "" {
		s.emitCompactionStarted(runID, opts, trigger)
	}
	result, err := s.Compactor.MaybeCompact(ctx, req)
	if err != nil {
		if result.Due && !skipNotified {
			if trigger == "" {
				trigger = "error"
			}
			s.emitCompactionSkipped(runID, opts, trigger, result.Reason)
			skipNotified = true
		}
		return messages, skipNotified, nil
	}
	if !result.Compacted {
		if result.Due && !skipNotified {
			if trigger == "" {
				trigger = "due"
			}
			s.emitCompactionSkipped(runID, opts, trigger, result.Reason)
			skipNotified = true
		}
		return messages, skipNotified, nil
	}
	s.emitCompacted(runID, opts, result.Messages, trigger, result.Summary)
	if err := s.checkPostCompactionPromptBudget(opts, result.Messages); err != nil {
		return result.Messages, skipNotified, err
	}
	return result.Messages, skipNotified, nil
}

func (s *Session) compactForToolOutputOverflow(ctx context.Context, runID string, opts RunOptions, messages []api.Message, latest api.ChatResponse, assistant api.Message, toolMessages []api.Message, overflows []toolOutputOverflow, skipNotified bool) ([]api.Message, bool, error) {
	if s.Compactor == nil {
		return messages, skipNotified, nil
	}

	keepUserTurns := 0
	req := s.compactionRequest(runID, opts, messages, latest)
	req.Force = true
	req.KeepUserTurns = &keepUserTurns
	s.emitCompactionStarted(runID, opts, "tool_output")

	result, err := s.Compactor.MaybeCompact(ctx, req)
	if err != nil {
		if result.Due && !skipNotified {
			s.emitCompactionSkipped(runID, opts, "tool_output", result.Reason)
			skipNotified = true
		}
		return messages, skipNotified, nil
	}
	if !result.Compacted {
		if result.Due && !skipNotified {
			s.emitCompactionSkipped(runID, opts, "tool_output", result.Reason)
			skipNotified = true
		}
		return messages, skipNotified, nil
	}

	overflowByID := make(map[string]toolOutputOverflow, len(overflows))
	for _, overflow := range overflows {
		overflowByID[overflow.toolCallID] = overflow
	}

	compacted := append([]api.Message(nil), result.Messages...)
	if !messageEmpty(assistant) {
		compacted = append(compacted, assistant)
	}

	for _, msg := range toolMessages {
		content := msg.Content
		toolName := msg.ToolName
		if overflow, ok := overflowByID[msg.ToolCallID]; ok {
			content = overflow.content
			if overflow.toolName != "" {
				toolName = overflow.toolName
			}
		}
		refit := s.toolMessageForPostCompactionContext(toolName, msg.ToolCallID, content, opts, compacted)
		compacted = append(compacted, refit)
	}

	s.emitCompacted(runID, opts, compacted, "tool_output", result.Summary)
	if err := s.checkPostCompactionPromptBudget(opts, compacted); err != nil {
		return compacted, skipNotified, err
	}
	return compacted, skipNotified, nil
}

func (s *Session) compactionRequest(runID string, opts RunOptions, messages []api.Message, latest api.ChatResponse) CompactionRequest {
	return CompactionRequest{
		ChatID:       opts.ChatID,
		Model:        opts.Model,
		SystemPrompt: opts.SystemPrompt,
		Messages:     messages,
		Tools:        s.availableTools(),
		Format:       opts.Format,
		Latest:       latest,
		Options:      opts.Options,
		KeepAlive:    opts.KeepAlive,
		Think:        opts.Think,
		ContinueTask: true,
		Progress: func(progress CompactionProgress) {
			_ = s.emit(Event{Type: EventCompactionProgress, RunID: runID, ChatID: opts.ChatID, Model: opts.Model, Tokens: progress.Tokens})
		},
	}
}

func (s *Session) emitCompactionStarted(runID string, opts RunOptions, status string) {
	_ = s.emit(Event{
		Type:   EventCompactionStarted,
		RunID:  runID,
		ChatID: opts.ChatID,
		Model:  opts.Model,
		Status: status,
	})
}

func (s *Session) emitCompactionSkipped(runID string, opts RunOptions, status, reason string) {
	_ = s.emit(Event{
		Type:    EventCompactionSkipped,
		RunID:   runID,
		ChatID:  opts.ChatID,
		Model:   opts.Model,
		Status:  status,
		Content: CompactionSkippedMessage(reason),
	})
}

func (s *Session) emitCompacted(runID string, opts RunOptions, messages []api.Message, status, summary string) {
	_ = s.emit(Event{
		Type:     EventCompacted,
		RunID:    runID,
		ChatID:   opts.ChatID,
		Model:    opts.Model,
		Status:   status,
		Content:  summary,
		Messages: messages,
	})
}

func (s *Session) autoCompactionTrigger(req CompactionRequest) string {
	if compactor, ok := s.Compactor.(*SimpleCompactor); ok && compactor != nil {
		if req.Force {
			return "force"
		}
		contextWindow := compactor.contextWindowTokens(req.Options)
		threshold := int(float64(contextWindow) * compactor.threshold())
		if threshold <= 0 {
			return ""
		}
		if req.Latest.PromptEvalCount > 0 && req.Latest.PromptEvalCount >= threshold {
			return "prompt_eval"
		}
		if estimateCompactionRequestTokens(req) >= threshold {
			return "estimate"
		}
	}
	return ""
}

func CompactionSkippedMessage(reason string) string {
	reason = strings.TrimSpace(reason)
	if reason == "" {
		reason = "compaction could not run"
	}
	return reason
}

func resolvedMaxToolRounds(value int) int {
	if value == 0 {
		return defaultMaxToolRounds
	}
	return value
}

func (s *Session) toolMessageForContext(toolName, toolCallID, content string, opts RunOptions, messages []api.Message) api.Message {
	maxRunes := maxToolResultRunes
	if limit := smallContextToolResultLimitRunes(s.contextWindowTokens(opts)); limit > 0 {
		maxRunes = min(maxRunes, limit)
	}

	msg := toolMessageWithLimit(toolName, toolCallID, content, maxRunes)
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
	// Keep oversized tool output below the compaction threshold before it is
	// appended to history. This is especially important for <=8k contexts: the
	// next step must still have enough room to compact and continue the same
	// user request instead of asking the user to prompt again.
	availableRunes := (threshold - baseTokens - overheadTokens - toolTruncationMarkerReserveTokens) * 4
	maxRunes = min(maxRunes, max(0, availableRunes))
	msg.Content = truncateToolResultContentTo(content, maxRunes)
	return msg
}

func (s *Session) toolMessageForPostCompactionContext(toolName, toolCallID, content string, opts RunOptions, messages []api.Message) api.Message {
	maxRunes := maxToolResultRunes
	if limit := smallContextToolResultLimitRunes(s.contextWindowTokens(opts)); limit > 0 {
		maxRunes = min(maxRunes, limit)
	}

	contextWindow := s.contextWindowTokens(opts)
	if contextWindow <= 0 {
		return toolMessageWithLimit(toolName, toolCallID, content, maxRunes)
	}

	baseTokens := s.estimateRunPromptTokens(opts, messages)
	overheadTokens := estimateMessagesTokens([]api.Message{{
		Role:       "tool",
		ToolName:   toolName,
		ToolCallID: toolCallID,
	}})
	availableRunes := (contextWindow - baseTokens - overheadTokens - toolTruncationMarkerReserveTokens) * 4
	maxRunes = min(maxRunes, max(0, availableRunes))
	return toolMessageWithLimit(toolName, toolCallID, content, maxRunes)
}

func toolMessageWithLimit(toolName, toolCallID, content string, maxRunes int) api.Message {
	return api.Message{
		Role:       "tool",
		Content:    truncateToolResultContentTo(content, maxRunes),
		ToolName:   toolName,
		ToolCallID: toolCallID,
	}
}

func smallContextToolResultLimitRunes(contextWindow int) int {
	switch {
	case contextWindow > 0 && contextWindow <= tinyContextToolResultTokenWindow:
		return tinyContextToolResultRunes
	case contextWindow > 0 && contextWindow <= smallContextToolResultTokenWindow:
		return smallContextToolResultRunes
	default:
		return 0
	}
}

func (s *Session) availableTools() api.Tools {
	if s == nil || s.Tools == nil {
		return nil
	}
	return s.Tools.Tools()
}

func (s *Session) compactionThresholdTokens(opts RunOptions) int {
	contextWindow := s.contextWindowTokens(opts)
	if contextWindow <= 0 {
		return 0
	}

	configuredThreshold := 0.0
	if compactor, ok := s.Compactor.(*SimpleCompactor); ok && compactor != nil {
		configuredThreshold = compactor.Options.Threshold
	}

	threshold := int(float64(contextWindow) * ResolveCompactionThreshold(configuredThreshold))
	if threshold <= 0 {
		return 0
	}
	return threshold
}

func (s *Session) contextWindowTokens(opts RunOptions) int {
	if s.Compactor == nil {
		return 0
	}

	configuredWindow := 0
	if compactor, ok := s.Compactor.(*SimpleCompactor); ok && compactor != nil {
		configuredWindow = compactor.Options.ContextWindowTokens
	}

	return ResolveContextWindowTokens(opts.Options, configuredWindow)
}

func toolMessage(toolName, toolCallID, content string) api.Message {
	return toolMessageWithLimit(toolName, toolCallID, content, maxToolResultRunes)
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
	if maxRunes <= 0 {
		return fmt.Sprintf("%s omitted ~%d tokens. Use a narrower command, line range, or search query if more detail is needed.]", toolOutputFullOmissionPrefix, approximateTokensFromRunes(len([]rune(content))))
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
	marker := fmt.Sprintf(
		"\n\n[tool output truncated: showing first ~%d tokens and last ~%d tokens; omitted ~%d tokens. Use a narrower command, line range, or search query if more detail is needed.]\n\n",
		approximateTokensFromRunes(head),
		approximateTokensFromRunes(tail),
		approximateTokensFromRunes(omitted),
	)
	return string(runes[:head]) + marker + string(runes[len(runes)-tail:])
}

func toolOutputFullyOmitted(content string) bool {
	return strings.HasPrefix(content, toolOutputFullOmissionPrefix)
}

func approximateTokensFromRunes(n int) int {
	if n <= 0 {
		return 0
	}
	return max(1, (n+3)/4)
}

func messageEmpty(msg api.Message) bool {
	return msg.Content == "" && msg.Thinking == "" && len(msg.ToolCalls) == 0
}
