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

type Session struct {
	Client           ChatClient
	EventSinks       []EventSink
	Tools            *Registry
	DisableTools     bool
	ApprovalPrompter ApprovalPrompter
	ApprovalState    *ApprovalState
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

// toolExecutionStop is the batch-level outcome for a group of tool calls,
// distinct from per-call Event.Status values. The values overlap with
// runFinish.status ("denied", "canceled") because a denied or canceled
// batch also terminates the run with the matching status.
type toolExecutionStop string

const (
	toolExecutionDenied   toolExecutionStop = "denied"
	toolExecutionCanceled toolExecutionStop = "canceled"
)

const toolExecutionDisabledMessage = "Tool execution disabled."

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
	if err := s.validateRun(opts); err != nil {
		return nil, err
	}
	if s.ApprovalState == nil {
		s.ApprovalState = &ApprovalState{}
	}
	runID := uuid.NewString()
	messages, err := s.buildRunMessages(ctx, runID, opts)
	if err != nil {
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
				return nil, err
			}
		case runPhaseDone:
			return s.finishRun(ctx, &st)
		}
	}
}

// validateRun checks the preconditions for a run.
func (s *Session) validateRun(opts RunOptions) error {
	if s == nil {
		return errors.New("nil session")
	}
	if s.Client == nil {
		return errors.New("agent session requires a chat client")
	}
	if opts.Model == "" {
		return errors.New("agent session requires a model")
	}
	return nil
}

// buildRunMessages sanitizes the provided message history, runs the preflight
// prompt-budget check, and returns the initial message list for the run. It
// emits an EventError and returns it if the preflight check fails.
func (s *Session) buildRunMessages(ctx context.Context, runID string, opts RunOptions) ([]api.Message, error) {
	messages := make([]api.Message, 0, len(opts.Messages)+len(opts.NewMessages))
	for _, msg := range opts.Messages {
		messages = append(messages, sanitizeMessageForRun(msg))
	}
	for _, msg := range opts.NewMessages {
		msg = sanitizeMessageForRun(msg)
		messages = append(messages, msg)
	}

	if err := s.checkPreflightPromptBudget(opts, messages); err != nil {
		s.emit(newErrorEvent(newEventMetadata(runID, opts), err.Error()))
		return nil, err
	}
	return messages, nil
}

func (s *Session) runModelStep(ctx context.Context, st *runState) error {
	opts := st.opts
	meta := newEventMetadata(st.runID, opts)

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
		s.emit(newErrorEvent(meta, err.Error()))
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
			s.emit(newErrorEvent(meta, skipErr.Error()))
			return skipErr
		}
		st.messages = append(st.messages, skipped...)
		st.finishCanceled()
		return nil
	}

	if s.DisableTools {
		batch, skipErr := s.disabledToolCalls(ctx, st.runID, opts, st.messages, pendingToolCalls)
		if skipErr != nil {
			s.emit(newErrorEvent(meta, skipErr.Error()))
			return skipErr
		}
		st.messages = append(st.messages, batch.messages...)
		st.toolBatch = &batch
		st.phase = runPhaseCompact
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
			s.emit(newErrorEvent(meta, skipErr.Error()))
			return skipErr
		}
		st.messages = append(st.messages, toolMessages...)
		err := fmt.Errorf("tool round limit reached after %d rounds; send another message to continue", st.maxToolRounds)
		s.emit(newErrorEvent(meta, err.Error()))
		st.finishError(err)
		return nil
	}

	st.phase = runPhaseTools
	return nil
}

func (s *Session) runToolStep(ctx context.Context, st *runState) error {
	batch, err := s.executeToolCalls(ctx, st.runID, st.opts, st.messages, st.pendingToolCalls)
	if err != nil {
		s.emit(newErrorEvent(newEventMetadata(st.runID, st.opts), err.Error()))
		return err
	}

	st.messages = append(st.messages, batch.messages...)
	st.toolBatch = &batch
	st.phase = runPhaseCompact
	return nil
}

func (s *Session) runCompactionStep(ctx context.Context, st *runState) error {
	opts := st.opts
	meta := newEventMetadata(st.runID, opts)
	var err error
	if st.toolBatch != nil && len(st.toolBatch.overflows) > 0 {
		st.messages, st.compactionSkipNotified, err = s.compactForToolOutputOverflow(ctx, st.runID, opts, st.messages, st.latest, st.assistant, st.toolBatch.messages, st.toolBatch.overflows, st.compactionSkipNotified)
	} else {
		st.messages, st.compactionSkipNotified, err = s.maybeCompact(ctx, st.runID, opts, st.messages, st.latest, st.compactionSkipNotified)
	}
	if err != nil {
		s.emit(newErrorEvent(meta, err.Error()))
		st.finishError(err)
		return nil
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
		event := newRunFinished(newEventMetadata(st.runID, st.opts), st.finish.status)
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
	meta := newEventMetadata(runID, opts)
	var tools api.Tools
	if !s.DisableTools {
		tools = s.availableTools()
	}
	req := buildChatRequest(opts, messages, tools)

	assistant := api.Message{Role: "assistant"}
	var pendingToolCalls []api.ToolCall

	err := s.Client.Chat(ctx, &req, func(response api.ChatResponse) error {
		if response.Message.Role != "" {
			assistant.Role = response.Message.Role
		}

		if messageEmpty(response.Message) {
			*latest = response
			return nil
		}

		if response.Message.Thinking != "" {
			assistant.Thinking += response.Message.Thinking
			if err := s.emit(newThinkingDelta(meta, response.Message.Thinking)); err != nil {
				return err
			}
		}

		if response.Message.Content != "" {
			assistant.Content += response.Message.Content
			if err := s.emit(newMessageDelta(meta, response.Message.Content)); err != nil {
				return err
			}
		}

		if len(response.Message.ToolCalls) > 0 {
			assistant.ToolCalls = append(assistant.ToolCalls, response.Message.ToolCalls...)
			pendingToolCalls = append(pendingToolCalls, response.Message.ToolCalls...)
			if err := s.emit(newToolCallDetected(meta, response.Message.ToolCalls)); err != nil {
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

func buildChatRequest(opts RunOptions, messages []api.Message, tools api.Tools) api.ChatRequest {
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
	if len(tools) > 0 {
		req.Tools = tools
	}
	return req
}

func (s *Session) executeToolCalls(ctx context.Context, runID string, opts RunOptions, messages []api.Message, calls []api.ToolCall) (toolBatchResult, error) {
	meta := newEventMetadata(runID, opts)
	batch := toolBatchResult{
		messages: make([]api.Message, 0, len(calls)),
	}
	projectedMessages := append([]api.Message(nil), messages...)
	// Pre-compute the full-history token estimate once per batch instead of
	// re-marshaling the entire history for each tool call. Per-call deltas
	// (tool messages already appended this batch) are tracked in batchTokens
	// and added to historyTokens for a lightweight running total.
	historyTokens := s.estimateRunPromptTokens(opts, messages)
	batchTokens := 0

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
		if ok && s.needsApproval(tool, toolName, args) {
			approvalReq.AddToolCall(call.ID, toolName, toolApprovalScope(tool, toolName, args), args)
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
				msg := s.toolMessageForContext(plan.toolName, plan.call.ID, content, opts, historyTokens+batchTokens)
				batch.messages = append(batch.messages, msg)
				projectedMessages = append(projectedMessages, msg)
				batchTokens += estimateMessagesTokens([]api.Message{msg})
				deniedContent := msg.Content
				if emitErr := s.emit(newToolFinished(meta, "denied", plan.call.ID, plan.toolName, "", plan.args, deniedContent, deniedContent)); emitErr != nil {
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
			msg := s.toolMessageForContext(toolName, call.ID, content, opts, historyTokens+batchTokens)
			batch.messages = append(batch.messages, msg)
			projectedMessages = append(projectedMessages, msg)
			batchTokens += estimateMessagesTokens([]api.Message{msg})
			content = msg.Content
			if toolOutputFullyOmitted(content) {
				batch.overflows = append(batch.overflows, toolOutputOverflow{toolName: toolName, toolCallID: call.ID, content: fmt.Sprintf("Error: unknown tool: %s", toolName)})
			}
			if emitErr := s.emit(newToolFinished(meta, "failed", call.ID, toolName, "", args, content, fmt.Sprintf("unknown tool: %s", toolName))); emitErr != nil {
				return toolBatchResult{}, emitErr
			}
			continue
		}

		if err := s.emit(newToolStarted(meta, call.ID, toolName, plan.workingDir, args)); err != nil {
			return toolBatchResult{}, err
		}

		result, err := s.Tools.Execute(ctx, ToolContext{WorkingDir: plan.workingDir}, call)
		if err != nil {
			rawContent := fmt.Sprintf("Error: %v", err)
			msg := s.toolMessageForContext(toolName, call.ID, rawContent, opts, historyTokens+batchTokens)
			batch.messages = append(batch.messages, msg)
			projectedMessages = append(projectedMessages, msg)
			batchTokens += estimateMessagesTokens([]api.Message{msg})
			content := msg.Content
			if toolOutputFullyOmitted(content) {
				batch.overflows = append(batch.overflows, toolOutputOverflow{toolName: toolName, toolCallID: call.ID, content: rawContent})
			}
			if emitErr := s.emitIgnoringCanceled(ctx, newToolFinished(meta, "failed", call.ID, toolName, "", args, content, err.Error())); emitErr != nil {
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

		msg := s.toolMessageForContext(toolName, call.ID, rawContent, opts, historyTokens+batchTokens)
		batch.messages = append(batch.messages, msg)
		projectedMessages = append(projectedMessages, msg)
		batchTokens += estimateMessagesTokens([]api.Message{msg})
		content := msg.Content

		if toolOutputFullyOmitted(content) {
			batch.overflows = append(batch.overflows, toolOutputOverflow{toolName: toolName, toolCallID: call.ID, content: rawContent})
		}
		if err := s.emitIgnoringCanceled(ctx, newToolFinished(meta, "done", call.ID, toolName, eventWorkingDir, args, content, "")); err != nil {
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

func (s *Session) disabledToolCalls(ctx context.Context, runID string, opts RunOptions, messages []api.Message, calls []api.ToolCall) (toolBatchResult, error) {
	meta := newEventMetadata(runID, opts)
	batch := toolBatchResult{
		messages: make([]api.Message, 0, len(calls)),
	}
	projectedMessages := append([]api.Message(nil), messages...)
	historyTokens := s.estimateRunPromptTokens(opts, messages)
	batchTokens := 0
	for _, call := range calls {
		toolName := call.Function.Name
		args := call.Function.Arguments.ToMap()
		msg := s.toolMessageForContext(toolName, call.ID, toolExecutionDisabledMessage, opts, historyTokens+batchTokens)
		batch.messages = append(batch.messages, msg)
		projectedMessages = append(projectedMessages, msg)
		batchTokens += estimateMessagesTokens([]api.Message{msg})
		if emitErr := s.emitIgnoringCanceled(ctx, newToolFinished(meta, "disabled", call.ID, toolName, "", args, msg.Content, msg.Content)); emitErr != nil {
			return toolBatchResult{}, emitErr
		}
	}
	return batch, nil
}

func (s *Session) skipToolCalls(ctx context.Context, runID string, opts RunOptions, calls []api.ToolCall, content string) ([]api.Message, error) {
	meta := newEventMetadata(runID, opts)
	toolMessages := make([]api.Message, 0, len(calls))
	for _, call := range calls {
		toolName := call.Function.Name
		args := call.Function.Arguments.ToMap()
		msg := toolMessage(toolName, call.ID, content)
		toolMessages = append(toolMessages, msg)
		if emitErr := s.emitIgnoringCanceled(ctx, newToolFinished(meta, "skipped", call.ID, toolName, "", args, msg.Content, msg.Content)); emitErr != nil {
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

	historyTokens := s.estimateRunPromptTokens(opts, compacted)
	batchTokens := 0
	for _, msg := range toolMessages {
		content := msg.Content
		toolName := msg.ToolName
		if overflow, ok := overflowByID[msg.ToolCallID]; ok {
			content = overflow.content
			if overflow.toolName != "" {
				toolName = overflow.toolName
			}
		}
		refit := s.toolMessageForPostCompactionContext(toolName, msg.ToolCallID, content, opts, historyTokens+batchTokens)
		compacted = append(compacted, refit)
		batchTokens += estimateMessagesTokens([]api.Message{refit})
	}

	s.emitCompacted(runID, opts, compacted, "tool_output", result.Summary)
	if err := s.checkPostCompactionPromptBudget(opts, compacted); err != nil {
		return compacted, skipNotified, err
	}
	return compacted, skipNotified, nil
}

func (s *Session) compactionRequest(runID string, opts RunOptions, messages []api.Message, latest api.ChatResponse) CompactionRequest {
	meta := newEventMetadata(runID, opts)
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
			_ = s.emit(newCompactionProgress(meta, progress.Tokens))
		},
	}
}

func (s *Session) emitCompactionStarted(runID string, opts RunOptions, status string) {
	_ = s.emit(newCompactionStarted(newEventMetadata(runID, opts), status))
}

func (s *Session) emitCompactionSkipped(runID string, opts RunOptions, status, reason string) {
	_ = s.emit(newCompactionSkipped(newEventMetadata(runID, opts), status, compactionSkippedMessage(reason)))
}

func (s *Session) emitCompacted(runID string, opts RunOptions, messages []api.Message, status, summary string) {
	_ = s.emit(newCompacted(newEventMetadata(runID, opts), messages, status, summary))
}

func (s *Session) autoCompactionTrigger(req CompactionRequest) string {
	if s.Compactor == nil {
		return ""
	}
	trigger, should := s.Compactor.ShouldCompact(req)
	if should {
		return trigger
	}
	return ""
}

func compactionSkippedMessage(reason string) string {
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

// toolMessageWithBudget sizes a tool result message to fit within a token
// budget (compaction threshold or context window). baseTokens is the
// pre-computed estimate of everything before this message; budgetTokens is
// the ceiling. If the message already fits, it is returned with only the
// small-context rune cap applied.
func (s *Session) toolMessageWithBudget(toolName, toolCallID, content string, opts RunOptions, baseTokens, budgetTokens int) api.Message {
	maxRunes := maxToolResultRunes
	if limit := smallContextToolResultLimitRunes(s.contextWindowTokens(opts)); limit > 0 {
		maxRunes = min(maxRunes, limit)
	}

	if budgetTokens <= 0 {
		return toolMessageWithLimit(toolName, toolCallID, content, maxRunes)
	}

	msg := toolMessageWithLimit(toolName, toolCallID, content, maxRunes)
	projectedTokens := baseTokens + estimateMessagesTokens([]api.Message{msg})
	if projectedTokens < budgetTokens {
		return msg
	}

	overheadTokens := estimateMessagesTokens([]api.Message{{
		Role:       "tool",
		ToolName:   toolName,
		ToolCallID: toolCallID,
	}})
	// Keep oversized tool output below the budget before it is appended to
	// history. This is especially important for <=8k contexts: the next step
	// must still have enough room to compact and continue the same user
	// request instead of asking the user to prompt again.
	availableRunes := (budgetTokens - baseTokens - overheadTokens - toolTruncationMarkerReserveTokens) * 4
	maxRunes = min(maxRunes, max(0, availableRunes))
	msg.Content = truncateToolResultContentTo(content, maxRunes)
	return msg
}

func (s *Session) toolMessageForContext(toolName, toolCallID, content string, opts RunOptions, baseTokens int) api.Message {
	return s.toolMessageWithBudget(toolName, toolCallID, content, opts, baseTokens, s.compactionThresholdTokens(opts))
}

func (s *Session) toolMessageForPostCompactionContext(toolName, toolCallID, content string, opts RunOptions, baseTokens int) api.Message {
	return s.toolMessageWithBudget(toolName, toolCallID, content, opts, baseTokens, s.contextWindowTokens(opts))
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
	if s.Compactor != nil {
		configuredThreshold = s.Compactor.Threshold()
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
	return s.Compactor.ContextWindowTokens(opts.Options)
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
		sanitized[i] = sanitizeMessageForRun(msg)
	}
	return sanitized
}

func truncateToolResultContent(content string) string {
	return truncateToolResultContentTo(content, maxToolResultRunes)
}

func truncateToolResultContentTo(content string, maxRunes int) string {
	return Truncate(content, TruncateConfig{
		MaxRunes:           maxRunes,
		HeadTail:           true,
		HeadPct:            75,
		Label:              "tool output",
		Hint:               "Use a narrower command, line range, or search query if more detail is needed.",
		FullOmissionPrefix: toolOutputFullOmissionPrefix,
	})
}

// TruncateConfig configures content truncation via Truncate.
type TruncateConfig struct {
	MaxRunes           int    // rune limit; <= 0 means full omission
	HeadTail           bool   // true = head + tail split; false = head only
	HeadPct            int    // percentage of MaxRunes for head (e.g. 75); tail gets the rest
	Label              string // e.g. "tool output", "summary", "stdout"
	Hint               string // guidance text appended to marker (optional)
	FullOmissionPrefix string // marker prefix when MaxRunes <= 0
}

// Truncate truncates content to at most cfg.MaxRunes runes. When HeadTail is
// true, it preserves the first HeadPct% and last (100-HeadPct)% of the budget
// with a marker between; otherwise it keeps only the head. MaxRunes <= 0
// triggers full omission using FullOmissionPrefix. All token counts in
// markers use ApproximateTokens.
func Truncate(content string, cfg TruncateConfig) string {
	runes := []rune(content)
	total := len(runes)

	if cfg.MaxRunes <= 0 {
		return fmt.Sprintf("%s omitted ~%d tokens.%s]", cfg.FullOmissionPrefix, ApproximateTokens(total), truncHint(cfg.Hint))
	}
	if total <= cfg.MaxRunes {
		return content
	}

	if !cfg.HeadTail {
		head := cfg.MaxRunes
		omitted := total - head
		return string(runes[:head]) + TruncMarker(cfg.Label, head, 0, omitted, false, cfg.Hint)
	}

	head := cfg.MaxRunes * cfg.HeadPct / 100
	tail := cfg.MaxRunes - head
	omitted := total - head - tail
	return string(runes[:head]) + TruncMarker(cfg.Label, head, tail, omitted, true, cfg.Hint) + string(runes[len(runes)-tail:])
}

func truncHint(hint string) string {
	hint = strings.TrimSpace(hint)
	if hint == "" {
		return ""
	}
	if !strings.HasSuffix(hint, ".") {
		hint += "."
	}
	return " " + hint
}

// TruncMarker formats a truncation marker with consistent wording. head and
// tail are rune counts; omitted is the count of runes removed. headTail
// selects the head+tail vs head-only format. hint is optional guidance text.
func TruncMarker(label string, head, tail, omitted int, headTail bool, hint string) string {
	var b strings.Builder
	b.WriteString("\n\n[")
	b.WriteString(label)
	b.WriteString(" truncated: ")
	if headTail {
		fmt.Fprintf(&b, "showing first ~%d tokens and last ~%d tokens; ", ApproximateTokens(head), ApproximateTokens(tail))
	} else {
		fmt.Fprintf(&b, "showing first ~%d tokens; ", ApproximateTokens(head))
	}
	fmt.Fprintf(&b, "omitted ~%d tokens.%s]", ApproximateTokens(omitted), truncHint(hint))
	if headTail {
		b.WriteString("\n\n")
	}
	return b.String()
}

func toolOutputFullyOmitted(content string) bool {
	return strings.HasPrefix(content, toolOutputFullOmissionPrefix)
}

// ApproximateTokens estimates token count from a character/byte count using
// the standard ~4 chars-per-token heuristic. It is intentionally rough; all
// callers use it only for sizing/truncation decisions, not billing.
func ApproximateTokens(n int) int {
	if n <= 0 {
		return 0
	}
	return max(1, (n+3)/4)
}

func messageEmpty(msg api.Message) bool {
	return msg.Content == "" && msg.Thinking == "" && len(msg.ToolCalls) == 0
}
