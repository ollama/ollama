package chat

import (
	"context"
	"errors"
	"fmt"
	"hash/fnv"
	"slices"
	"strings"

	tea "github.com/charmbracelet/bubbletea"

	coreagent "github.com/ollama/ollama/agent"
	"github.com/ollama/ollama/agent/skills"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/cmd/internal/filedata"
)

var chatSpinnerFrames = []string{"...", ".", "..", "..."}

const (
	maxResumePickerItems      = 8
	maxInlineModelPickerItems = 5
	maxSlashCompletions       = 5
	maxPromptHistory          = 50
	waitingSpinnerTicks       = 4
)

var chatEmptyPrompts = []string{
	`read this repo and tell me where to start`,
	`what changed on this branch?`,
	`run the tests and summarize failures`,
	`find the riskiest code path in this folder`,
	`search the web and compare the latest docs with this implementation`,
	`summarize this file and suggest edits`,
}

type ModelOption struct {
	Name        string
	Description string
	Recommended bool
}

type Options struct {
	Model                       string
	ChatID                      string
	Messages                    []api.Message
	Client                      coreagent.ChatClient
	Store                       coreagent.ChatStore
	Tools                       *coreagent.Registry
	ToolRegistryForModel        func(context.Context, string) *coreagent.Registry
	MultiModalForModel          func(context.Context, string) bool
	ModelOptions                func(context.Context) ([]ModelOption, error)
	OnModelSelected             func(context.Context, string) error
	SystemPromptForModel        func(context.Context, string, *coreagent.Registry) string
	Clipboard                   func(context.Context, string) error
	Approval                    coreagent.ApprovalHandler
	EventSink                   coreagent.EventSink
	AutoApproveTools            bool
	WorkingDir                  string
	RootDir                     string
	Format                      string
	Options                     map[string]any
	Think                       *api.ThinkValue
	KeepAlive                   *api.Duration
	Images                      []api.ImageData
	MultiModal                  bool
	HideThinking                bool
	Verbose                     bool
	Compactor                   coreagent.Compactor
	ContextWindowTokens         int
	ContextWindowTokensForModel func(context.Context, string, int) int
	PreloadModel                func(context.Context, string, *api.ThinkValue) error
	CompactionThreshold         float64
	NewChat                     func(context.Context) (string, error)
	Skills                      *skills.Catalog
	SystemPrompt                string
}

type Result struct {
	ChatID   string
	Messages []api.Message
}

//nolint:containedctx // chatModel is a Bubble Tea session model; the context is the run-scoped cancellation root.
type chatModel struct {
	ctx          context.Context
	opts         Options
	chatID       string
	messages     []api.Message
	liveMessages []api.Message
	entries      []chatEntry
	workingDir   string

	input             []rune
	inputCursor       int
	inputCursorSet    bool
	inputAttachments  []chatInputAttachment
	inputPastedTexts  []chatInputPastedText
	nextImageID       int
	nextAudioID       int
	nextPastedTextID  int
	queued            []string
	queuedAttachments [][]chatInputAttachment
	queuedPastedTexts [][]chatInputPastedText
	promptHistory     []string
	promptCursor      int
	promptDraft       []rune
	promptActive      bool
	running           bool
	awaitingModel     bool
	compacting        bool
	cancel            context.CancelFunc
	events            <-chan tea.Msg
	compactEvents     <-chan tea.Msg
	scroll            int
	toolOutputMode    bool
	toolOutputOpen    bool
	flowPrintedLines  int
	thinking          bool
	thinkingTokens    int
	compactingTokens  int
	contextTokens     int
	contextEstimate   bool
	resumePicker      *chatResumePicker
	modelPicker       *chatModelPicker
	thinkPicker       *chatThinkPicker
	historyPopup      *chatHistoryPopup
	approvalPrompt    *chatApprovalPrompt
	reviewApproval    coreagent.ApprovalHandler
	permissionMode    *chatPermissionMode
	permissionNotice  string
	selection         chatSelection

	width              int
	height             int
	boundedFrame       bool
	fullScreen         bool
	status             string
	spinner            int
	tickActive         bool
	preloadingModel    string
	complete           int
	quitting           bool
	quitArmed          bool
	quitArmedKey       string
	escArmed           bool
	eventErrorRendered bool
	err                error
}

type chatSelectionPoint struct {
	line int
	col  int
}

type chatSelection struct {
	active bool
	anchor chatSelectionPoint
	cursor chatSelectionPoint
}

func startChatSelection(selection *chatSelection, msg tea.MouseMsg, contains func(tea.MouseMsg) bool, point func(tea.MouseMsg) chatSelectionPoint) {
	if !contains(msg) {
		*selection = chatSelection{}
		return
	}
	p := point(msg)
	*selection = chatSelection{active: true, anchor: p, cursor: p}
}

func dragChatSelection(selection *chatSelection, msg tea.MouseMsg, point func(tea.MouseMsg) chatSelectionPoint, scrollEdge func(tea.MouseMsg)) {
	if !selection.active {
		return
	}
	selection.cursor = point(msg)
	scrollEdge(msg)
}

func finishChatSelection(m chatModel, selection *chatSelection, msg tea.MouseMsg, point func(tea.MouseMsg) chatSelectionPoint, selectedText func() string) (tea.Model, tea.Cmd) {
	if !selection.active {
		return m, nil
	}
	selection.cursor = point(msg)
	selected := selectedText()
	if strings.TrimSpace(selected) == "" {
		*selection = chatSelection{}
		return m, nil
	}
	return m, func() tea.Msg {
		if m.opts.Clipboard == nil {
			return nil
		}
		if err := m.opts.Clipboard(m.ctx, selected); err != nil {
			return chatClipboardErrorMsg{err: err}
		}
		return nil
	}
}

type chatInputAttachment struct {
	placeholder string
	kind        string
	data        api.ImageData
}

func Run(ctx context.Context, opts Options) (*Result, error) {
	if opts.Approval == nil {
		opts.Approval = coreagent.NewApprovalManager(coreagent.ApprovalManagerOptions{})
	}
	if opts.RootDir == "" {
		opts.RootDir = opts.WorkingDir
	}
	if opts.Clipboard == nil {
		opts.Clipboard = writeClipboard
	}
	reviewApproval := chatReviewApprovalHandler(opts.Approval)
	autoApproveTools := opts.AutoApproveTools || approvalHandlerAutoApproves(opts.Approval)
	opts.Approval = reviewApproval

	m := chatModel{
		ctx:            ctx,
		opts:           opts,
		chatID:         opts.ChatID,
		messages:       slices.Clone(opts.Messages),
		workingDir:     opts.WorkingDir,
		reviewApproval: reviewApproval,
		permissionMode: newChatPermissionMode(autoApproveTools),
		promptHistory:  initialPromptHistory(ctx, opts),
		boundedFrame:   true,
		fullScreen:     true,
		status:         "ready",
	}
	m.nextImageID, m.nextAudioID = nextInputAttachmentIDsFromMessages(m.messages)
	m.nextPastedTextID = nextInputPastedTextIDFromMessages(m.messages)
	m.entries = entriesFromMessages(m.messages)
	m.contextTokens = m.estimatePromptTokens(m.messages, "")
	m.contextEstimate = true
	if opts.PreloadModel != nil && strings.TrimSpace(opts.Model) != "" {
		m.preloadingModel = strings.TrimSpace(opts.Model)
	}

	p := tea.NewProgram(m, tea.WithReportFocus(), tea.WithMouseCellMotion())
	finalModel, err := p.Run()
	if err != nil {
		return nil, err
	}

	fm := finalModel.(chatModel)
	if fm.err != nil {
		return nil, fm.err
	}
	return &Result{ChatID: fm.chatID, Messages: fm.messages}, nil
}

func (m chatModel) Init() tea.Cmd {
	cmds := []tea.Cmd{tea.EnterAltScreen}
	if m.preloadingModel != "" && m.opts.PreloadModel != nil {
		cmds = append(cmds, preloadModelCmd(m.ctx, m.opts.PreloadModel, m.preloadingModel, m.opts.Think), chatTickCmd())
	}
	return tea.Batch(cmds...)
}

func (m chatModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	if m.canEditInput() && isShiftEnterCSI(msg) {
		m.insertInputNewline()
		return m, nil
	}

	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		wasSet := m.width > 0 || m.height > 0
		resized := m.width != msg.Width || m.height != msg.Height
		m.width = msg.Width
		m.height = msg.Height
		if wasSet && resized {
			m.resetRenderAfterResize()
			return m, tea.Batch(tea.EnterAltScreen, tea.ClearScreen)
		}
		return m.withFlowTranscriptFlush(nil)

	case tea.FocusMsg:
		return m, nil

	case chatTickMsg:
		m.tickActive = false
		if !m.running && !m.compacting && m.preloadingModel == "" {
			return m, nil
		}
		m.spinner++
		cmd := m.scheduleTick()
		return m, cmd

	case chatModelPreloadDoneMsg:
		if msg.model != "" && msg.model != m.preloadingModel {
			return m, nil
		}
		m.preloadingModel = ""
		if msg.err != nil {
			if isUnsupportedThinkingError(msg.err) && thinkRequestsThinking(m.opts.Think) {
				m.opts.Think = &api.ThinkValue{Value: false}
				m.status = fmt.Sprintf("Thinking disabled for %s", msg.model)
				if msg.model != "" && m.opts.PreloadModel != nil {
					m.preloadingModel = msg.model
					return m, tea.Batch(preloadModelCmd(m.ctx, m.opts.PreloadModel, msg.model, m.opts.Think), m.scheduleTick())
				}
				return m, nil
			}
			if !errors.Is(msg.err, context.Canceled) {
				m.status = "error"
				m.entries = append(m.entries, newChatEntry(chatEntry{role: "error", content: fmt.Sprintf("Could not load model: %v", msg.err), err: msg.err.Error()}))
			}
			return m, nil
		}
		m.refreshContextWindowTokens(msg.model)
		return m, nil

	case chatAgentMsg:
		m.applyAgentEvent(msg.event)
		return m.withFlowTranscriptFlush(waitForChatMsg(m.events))

	case chatClipboardErrorMsg:
		if msg.err != nil {
			m.status = "clipboard error: " + msg.err.Error()
		}
		return m, nil

	case chatApprovalPromptMsg:
		m.resumePicker = nil
		m.modelPicker = nil
		m.historyPopup = nil
		m.openApprovalPrompt(msg)
		return m, nil

	case chatRunDoneMsg:
		wasCanceling := m.status == "canceling"
		m.running = false
		m.awaitingModel = false
		m.compacting = false
		m.compactingTokens = 0
		m.cancel = nil
		m.events = nil
		m.thinking = false
		m.thinkingTokens = 0
		m.approvalPrompt = nil
		if msg.result != nil {
			m.messages = msg.result.Messages
			m.liveMessages = nil
			if msg.result.WorkingDir != "" {
				m.workingDir = msg.result.WorkingDir
			}
			m.refreshContextWindowTokens(m.responseModelName(&msg.result.Latest))
			m.contextTokens = m.estimatePromptTokens(m.messages, "")
			m.contextEstimate = true
			if !messagesEndWithCompactionResult(m.messages) {
				m.applyResponseMetrics(&msg.result.Latest)
			}
		}
		m.groupCompletedToolHistory()
		if msg.result == nil {
			m.finishLiveMessagesForStoppedRun(msg.newMessagesPersisted, msg.persistedMessages)
		}
		if wasCanceling || isChatContextCanceledError(msg.err) {
			m.status = "Tell the model what to do instead."
			return m.withFlowTranscriptFlush(m.startNextQueued())
		}
		if msg.err != nil {
			if !m.eventErrorRendered {
				m.entries = append(m.entries, newChatEntry(chatEntry{role: "error", content: msg.err.Error(), err: msg.err.Error()}))
			}
			m.status = "error"
			return m.withFlowTranscriptFlush(nil)
		}
		if msg.result != nil {
			m.attachVerboseMetrics(msg.result.Latest.Metrics)
		}
		m.status = "ready"
		return m.withFlowTranscriptFlush(m.startNextQueued())

	case chatCompactDoneMsg:
		return m.finishManualCompaction(msg)

	case chatCompactProgressMsg:
		if msg.tokens > m.compactingTokens {
			m.compactingTokens = msg.tokens
		}
		return m, waitForChatMsg(m.compactEvents)

	case chatEventsClosedMsg:
		if m.compacting {
			return m.Update(chatCompactDoneMsg{err: context.Canceled})
		}
		if m.running {
			return m.Update(chatRunDoneMsg{err: context.Canceled, newMessagesPersisted: true})
		}
		return m, nil

	case chatEditorDoneMsg:
		m.applyEditorResult(msg)
		return m, nil

	case tea.MouseMsg:
		return m.updateMouse(msg)

	case tea.KeyMsg:
		return m.updateKey(msg)
	}
	return m, nil
}

func (m *chatModel) resetRenderAfterResize() {
	m.enterFullScreen()
	m.scroll = m.maxScroll()
}

func (m *chatModel) enterFullScreen() {
	m.boundedFrame = true
	m.fullScreen = true
	m.flowPrintedLines = 0
	m.selection = chatSelection{}
}

func (m chatModel) updateMouse(msg tea.MouseMsg) (tea.Model, tea.Cmd) {
	if m.modelPicker != nil {
		switch msg.Type {
		case tea.MouseWheelUp:
			m.modelPicker.move(-3)
		case tea.MouseWheelDown:
			m.modelPicker.move(3)
		}
		return m, nil
	}
	if m.historyPopup != nil {
		return m.updateHistoryPopupMouse(msg)
	}
	if !m.mouseInTranscript(msg) && !m.selection.active {
		switch msg.Type {
		case tea.MouseWheelUp:
			m.scrollBy(3)
		case tea.MouseWheelDown:
			m.scrollBy(-3)
		}
		return m, nil
	}
	switch msg.Type {
	case tea.MouseWheelUp:
		m.scrollBy(3)
	case tea.MouseWheelDown:
		m.scrollBy(-3)
	case tea.MouseLeft:
		switch msg.Action {
		case tea.MouseActionPress:
			m.startTranscriptSelection(msg)
		case tea.MouseActionMotion:
			m.dragTranscriptSelection(msg)
		default:
			if msg.Action == 0 {
				m.startTranscriptSelection(msg)
			}
		}
	case tea.MouseMotion:
		m.dragTranscriptSelection(msg)
	case tea.MouseRelease:
		return m.finishTranscriptSelection(msg)
	}
	return m, nil
}

func (m chatModel) mouseInTranscript(msg tea.MouseMsg) bool {
	top, height := m.transcriptLayout()
	if msg.X < 0 || msg.X >= m.viewWidth() {
		return false
	}
	return msg.Y >= top && msg.Y < top+height
}

func (m chatModel) mouseTranscriptPoint(msg tea.MouseMsg) chatSelectionPoint {
	top, height := m.transcriptLayout()
	visibleY := clamp(msg.Y-top, 0, max(0, height-1))
	line := m.visibleTranscriptStartLine(m.viewWidth(), height) + visibleY
	col := max(0, msg.X)
	return chatSelectionPoint{line: line, col: col}
}

func (m *chatModel) startTranscriptSelection(msg tea.MouseMsg) {
	startChatSelection(&m.selection, msg, m.mouseInTranscript, m.mouseTranscriptPoint)
}

func (m *chatModel) dragTranscriptSelection(msg tea.MouseMsg) {
	dragChatSelection(&m.selection, msg, m.mouseTranscriptPoint, func(msg tea.MouseMsg) {
		top, height := m.transcriptLayout()
		if msg.Y <= top {
			m.scrollBy(1)
		} else if msg.Y >= top+height-1 {
			m.scrollBy(-1)
		}
	})
}

func (m chatModel) finishTranscriptSelection(msg tea.MouseMsg) (tea.Model, tea.Cmd) {
	return finishChatSelection(m, &m.selection, msg, m.mouseTranscriptPoint, func() string {
		return m.selectedTranscriptText(m.viewWidth())
	})
}

func (m chatModel) updateKey(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	if msg.Type == tea.KeyCtrlO {
		m.toggleInlineToolOutput()
		m.disarmQuit()
		m.disarmEsc()
		// Toggling tool output changes the transcript height; force a full
		// repaint so the input box and model-status footer aren't left stale.
		return m, tea.ClearScreen
	}
	if msg.Type == tea.KeyShiftTab {
		return m.togglePermissionMode()
	}
	if m.approvalPrompt != nil {
		return m.updateApprovalPrompt(msg)
	}
	if m.modelPicker != nil {
		return m.updateModelPicker(msg)
	}
	if m.thinkPicker != nil {
		return m.updateThinkPicker(msg)
	}
	if m.resumePicker != nil {
		return m.updateResumePicker(msg)
	}
	if m.historyPopup != nil {
		return m.updateHistoryPopup(msg)
	}
	if msg.Type != tea.KeyCtrlC && msg.Type != tea.KeyCtrlD {
		m.disarmQuit()
	}
	if msg.Type != tea.KeyEsc {
		m.disarmEsc()
	}

	switch msg.Type {
	case tea.KeyCtrlC:
		return m.updateCtrlC()
	case tea.KeyCtrlD:
		return m.updateCtrlD()
	case tea.KeyEsc:
		return m.updateEsc()
	case tea.KeyEnter:
		if msg.Alt {
			m.insertInputNewline()
			return m, nil
		}
		return m.handleSubmit()
	case tea.KeyCtrlJ:
		m.insertInputNewline()
		return m, nil
	case tea.KeyCtrlG:
		return m.openInputEditor()
	case tea.KeyUp:
		return m.updateUpKey()
	case tea.KeyDown:
		return m.updateDownKey()
	case tea.KeyLeft:
		m.moveInputCursorHorizontal(-1)
	case tea.KeyRight:
		m.moveInputCursorHorizontal(1)
	case tea.KeyCtrlA:
		m.moveInputCursorToLineStart()
	case tea.KeyCtrlE:
		m.moveInputCursorToLineEnd()
	case tea.KeyCtrlB:
		m.moveInputCursorHorizontal(-1)
	case tea.KeyCtrlF:
		m.moveInputCursorHorizontal(1)
	case tea.KeyCtrlLeft:
		m.moveInputCursorWord(-1)
	case tea.KeyCtrlRight:
		m.moveInputCursorWord(1)
	case tea.KeyPgUp:
		m.scrollBy(max(1, m.transcriptHeight()-1))
	case tea.KeyPgDown:
		m.scrollBy(-max(1, m.transcriptHeight()-1))
	case tea.KeyHome:
		m.moveInputCursorToLineStart()
	case tea.KeyEnd:
		m.moveInputCursorToLineEnd()
	case tea.KeyCtrlHome:
		m.scroll = m.maxScroll()
	case tea.KeyCtrlEnd:
		m.scroll = 0
	case tea.KeyBackspace:
		m.resetPromptHistoryCursor()
		if msg.Alt {
			m.deleteInputWordBackward()
		} else {
			m.deleteInputBackward()
		}
	case tea.KeyCtrlW:
		m.resetPromptHistoryCursor()
		m.deleteInputWordBackward()
	case tea.KeyCtrlU:
		m.resetPromptHistoryCursor()
		m.clearInput()
	case tea.KeyCtrlK:
		m.resetPromptHistoryCursor()
		m.deleteInputForward()
	case tea.KeyTab:
		m.applyCompletion()
	case tea.KeySpace:
		m.insertInputRunes([]rune{' '})
	case tea.KeyRunes:
		m.insertInputRunesFromKey(msg.Runes, msg.Paste)
	default:
		if m.canEditInput() && isShiftEnterCSI(msg) {
			m.insertInputNewline()
		}
	}
	return m, nil
}

func (m *chatModel) toggleInlineToolOutput() {
	m.toolOutputMode = true
	m.toolOutputOpen = !m.toolOutputOpen
	m.applyToolOutputMode()
	m.selection = chatSelection{}
	m.scroll = clamp(m.scroll, 0, m.maxScroll())
}

func (m chatModel) updateCtrlC() (tea.Model, tea.Cmd) {
	if len(m.input) > 0 {
		m.clearInput()
		m.resetPromptHistoryCursor()
		m.disarmQuit()
		m.status = "ready"
		return m, nil
	}
	if (m.running || m.compacting) && m.cancel != nil {
		m.cancel()
		m.disarmQuit()
		m.status = "canceling"
		return m, nil
	}
	if !m.quitArmed || m.quitArmedKey != "ctrl+c" {
		m.armQuit("ctrl+c", "press ctrl+c again to quit")
		return m, nil
	}
	m.quitting = true
	return m, m.quitCmd()
}

func (m chatModel) updateCtrlD() (tea.Model, tea.Cmd) {
	if len(m.input) > 0 {
		return m, nil
	}
	if !m.quitArmed || m.quitArmedKey != "ctrl+d" {
		m.armQuit("ctrl+d", "press ctrl+d again to quit")
		return m, nil
	}
	m.quitting = true
	if (m.running || m.compacting) && m.cancel != nil {
		m.cancel()
	}
	return m, m.quitCmd()
}

func (m chatModel) quitCmd() tea.Cmd {
	if m.fullScreen {
		return tea.Batch(tea.ExitAltScreen, tea.Quit)
	}
	return tea.Quit
}

func (m *chatModel) armQuit(key, status string) {
	m.quitArmed = true
	m.quitArmedKey = key
	m.status = status
}

func (m chatModel) updateEsc() (tea.Model, tea.Cmd) {
	if !m.escArmed {
		m.escArmed = true
		switch {
		case len(m.input) > 0 && (m.running || m.compacting):
			m.status = "press esc again to clear input and cancel"
		case len(m.input) > 0:
			m.status = "press esc again to clear input"
		case (m.running || m.compacting) && m.cancel != nil:
			m.status = "press esc again to cancel"
		default:
			m.status = "ready"
		}
		return m, nil
	}

	m.escArmed = false
	cleared := false
	if len(m.input) > 0 {
		m.clearInput()
		m.resetPromptHistoryCursor()
		cleared = true
	}
	if (m.running || m.compacting) && m.cancel != nil {
		m.cancel()
		m.disarmQuit()
		m.status = "canceling"
		return m, nil
	}
	if cleared {
		m.status = "ready"
	} else {
		m.status = "ready"
	}
	return m, nil
}

func (m *chatModel) clearInput() {
	m.input = nil
	m.inputCursor = 0
	m.inputCursorSet = false
	m.inputAttachments = nil
	m.inputPastedTexts = nil
	m.complete = 0
}

func (m chatModel) updateUpKey() (tea.Model, tea.Cmd) {
	if m.promptActive && m.movePromptHistory(-1) {
		return m, nil
	}
	if m.moveCompletion(-1) {
		return m, nil
	}
	if m.moveInputCursorVertical(-1) {
		return m, nil
	}
	m.movePromptHistory(-1)
	return m, nil
}

func (m chatModel) updateDownKey() (tea.Model, tea.Cmd) {
	if m.promptActive && m.movePromptHistory(1) {
		return m, nil
	}
	if m.moveCompletion(1) {
		return m, nil
	}
	if m.moveInputCursorVertical(1) {
		return m, nil
	}
	m.movePromptHistory(1)
	return m, nil
}

func (m chatModel) View() string {
	if m.quitting {
		return ""
	}

	width := m.width
	if width <= 0 {
		width = 80
	}
	height := m.height
	if height <= 0 {
		height = 24
	}

	if m.resumePicker != nil && m.shouldRenderPickerFullFrame(width, height) {
		return renderFullFrame(m.renderResumePicker(width), width, height)
	}
	if m.modelPicker != nil && m.shouldRenderPickerFullFrame(width, height) {
		return renderFullFrame(m.renderModelPicker(width), width, height)
	}
	if m.thinkPicker != nil {
		return renderFullFrame(m.renderThinkPicker(width), width, height)
	}
	if m.historyPopup != nil {
		return renderFullFrame(m.renderHistoryPopup(width, height), width, height)
	}
	if !m.boundedFrame {
		return m.flowView(width)
	}

	headerLines := m.headerLines()
	allTranscriptLines := m.transcriptLines(width)
	contentLineCount := len(allTranscriptLines)

	bottomLines := m.bottomLines(width, height-len(headerLines))
	bottomGap := transcriptInputGap(height-len(headerLines), len(bottomLines), contentLineCount)
	available := height - len(headerLines) - len(bottomLines) - bottomGap
	if available < 0 {
		available = 0
	}

	transcriptLines := m.visibleTranscriptLinesForLines(allTranscriptLines, available)
	if len(allTranscriptLines) > available {
		for len(transcriptLines) < available {
			transcriptLines = append(transcriptLines, "")
		}
	}

	lines := append(headerLines, transcriptLines...)
	for range bottomGap {
		lines = append(lines, "")
	}
	lines = append(lines, bottomLines...)
	return renderFrameLines(lines, width, height)
}

func (m chatModel) flowView(width int) string {
	allTranscriptLines := m.transcriptLines(width)
	bottomLines := m.bottomLines(width, 0)
	bottomGap := transcriptInputGap(0, len(bottomLines), len(allTranscriptLines))

	printed := clamp(m.flowPrintedLines, 0, len(allTranscriptLines))
	lines := slices.Clone(allTranscriptLines[printed:])
	for range bottomGap {
		lines = append(lines, "")
	}
	lines = append(lines, bottomLines...)
	return strings.Join(lines, "\n")
}

func (m chatModel) withFlowTranscriptFlush(cmd tea.Cmd) (tea.Model, tea.Cmd) {
	next, printCmd := m.flowTranscriptFlushCmd()
	return next, tea.Batch(printCmd, cmd)
}

func (m chatModel) flowTranscriptFlushCmd() (chatModel, tea.Cmd) {
	if m.boundedFrame || m.resumePicker != nil || m.modelPicker != nil || m.thinkPicker != nil || m.historyPopup != nil {
		return m, nil
	}
	width := m.viewWidth()
	lines := m.transcriptLines(width)
	if len(lines) == 0 {
		m.flowPrintedLines = 0
		return m, nil
	}
	m.flowPrintedLines = clamp(m.flowPrintedLines, 0, len(lines))
	flushCount := m.flowTranscriptFlushCount(lines, width)
	if flushCount <= m.flowPrintedLines {
		return m, nil
	}
	pending := strings.Join(lines[m.flowPrintedLines:flushCount], "\n")
	m.flowPrintedLines = flushCount
	return m, tea.Println(pending)
}

func (m chatModel) flowTranscriptFlushCount(lines []string, width int) int {
	holdFrom := m.flowTranscriptHoldEntryIndex()
	if holdFrom < 0 {
		return len(lines)
	}
	clone := m
	clone.entries = slices.Clone(m.entries[:holdFrom])
	clone.selection = chatSelection{}
	return len(clone.transcriptLines(width))
}

func (m chatModel) flowTranscriptHoldEntryIndex() int {
	if !m.running && !m.compacting {
		return -1
	}
	if len(m.entries) == 0 {
		return -1
	}
	index := len(m.entries) - 1
	entry := m.entries[index]
	switch entry.role {
	case "assistant":
		if entry.content != "" {
			return index
		}
	case "tool":
		if isToolActiveStatus(entry.status) {
			return index
		}
	}
	return -1
}

func (m chatModel) emptyChatHint() string {
	if len(chatEmptyPrompts) == 0 {
		return "Try asking the agent to inspect files, run tools, or explain a repo."
	}
	h := fnv.New32a()
	_, _ = h.Write([]byte(m.chatID))
	prompt := chatEmptyPrompts[int(h.Sum32())%len(chatEmptyPrompts)]
	return `Try: "` + prompt + `"`
}

func (m chatModel) headerLines() []string {
	return nil
}

func (m *chatModel) resetChat(status string) (tea.Model, tea.Cmd) {
	m.messages = nil
	m.liveMessages = nil
	m.entries = nil
	m.queued = nil
	m.queuedAttachments = nil
	m.queuedPastedTexts = nil
	m.inputAttachments = nil
	m.inputPastedTexts = nil
	m.nextImageID = 0
	m.nextAudioID = 0
	m.nextPastedTextID = 1
	m.resetPromptHistoryCursor()
	m.resetWorkingDir()
	m.thinking = false
	m.thinkingTokens = 0
	m.contextTokens = 0
	m.contextEstimate = true
	m.scroll = 0
	m.flowPrintedLines = 0
	if m.opts.NewChat != nil {
		chatID, err := m.opts.NewChat(m.ctx)
		if err != nil {
			m.entries = append(m.entries, newChatEntry(chatEntry{role: "error", content: err.Error(), err: err.Error()}))
			m.status = "error"
			return *m, nil
		}
		m.chatID = chatID
	}
	m.status = status
	return *m, nil
}

func (m *chatModel) resetWorkingDir() {
	if m.opts.RootDir != "" {
		m.workingDir = m.opts.RootDir
		return
	}
	m.workingDir = m.opts.WorkingDir
}

func (m *chatModel) startRun(input string) (tea.Model, tea.Cmd) {
	displayInput, message, err := m.userMessageFromInput(input, input)
	if err != nil {
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "error", content: err.Error(), err: err.Error()}))
		m.status = "error"
		return *m, nil
	}
	return m.startRunWithMessages(displayInput, message.Content, []api.Message{message}, "")
}

func (m *chatModel) userMessageFromInput(displayInput, userInput string) (string, api.Message, error) {
	content := m.expandPastedTextPlaceholders(userInput)
	images := slices.Clone(m.opts.Images)
	preloaded := len(images)
	placeholderAttachments := m.activeInputAttachmentsFor(userInput)
	for _, attachment := range placeholderAttachments {
		images = append(images, attachment.data)
	}
	extracted := 0

	if m.opts.MultiModal {
		cleaned, files, err := filedata.ExtractWithFiles(content)
		if err != nil {
			return "", api.Message{}, err
		}
		content = cleaned
		extracted = len(files)
		for _, file := range files {
			images = append(images, file.Data)
		}
	}
	m.opts.Images = nil
	m.inputAttachments = nil
	m.inputPastedTexts = nil

	if len(images) > 0 {
		base := displayInput
		if extracted > 0 {
			base = content
		}
		if len(placeholderAttachments) > 0 && extracted == 0 && preloaded == 0 {
			displayInput = strings.TrimSpace(content)
		} else {
			displayInput = chatDisplayInputWithAttachments(base, len(images))
		}
	}

	return displayInput, api.Message{Role: "user", Content: content, Images: images}, nil
}

func chatDisplayInputWithAttachments(input string, count int) string {
	note := fmt.Sprintf("[attached %d file%s]", count, pluralSuffix(count))
	input = strings.TrimSpace(input)
	if input == "" {
		return note
	}
	return input + "\n\n" + note
}

func pluralSuffix(count int) string {
	if count == 1 {
		return ""
	}
	return "s"
}

func (m *chatModel) startRunWithMessages(displayInput, historyInput string, newMessages []api.Message, extraSystemPrompt string) (tea.Model, tea.Cmd) {
	m.ensurePermissionMode()
	m.refreshContextWindowTokens(m.opts.Model)
	m.addPromptHistory(historyInput)
	m.entries = append(m.entries, newChatEntry(chatEntry{role: "user", content: displayInput}))
	if len(newMessages) > 1 {
		m.entries = append(m.entries, entriesFromMessages(newMessages[1:])...)
	}
	m.running = true
	m.awaitingModel = false
	m.status = "running"
	m.spinner = 0
	m.scroll = 0
	m.thinking = false
	m.thinkingTokens = 0
	m.eventErrorRendered = false
	systemPrompt := m.systemPrompt(extraSystemPrompt)
	m.liveMessages = append(slices.Clone(m.messages), newMessages...)
	m.contextTokens = m.estimatePromptTokens(m.liveMessages, systemPrompt)
	m.contextEstimate = true

	runCtx, cancel := context.WithCancel(m.ctx)
	m.cancel = cancel
	events := make(chan tea.Msg, 128)
	m.events = events

	var newMessagesPersisted bool
	eventSink := coreagent.EventSink(chatEventSink{ctx: runCtx, ch: events, newMessagesPersisted: &newMessagesPersisted})
	if m.opts.EventSink != nil {
		eventSink = coreagent.MultiEventSink{eventSink, m.opts.EventSink}
	}

	session := &coreagent.Session{
		Client:     m.opts.Client,
		Store:      m.opts.Store,
		Events:     eventSink,
		Tools:      m.opts.Tools,
		Approval:   m.approvalHandlerForRun(events),
		WorkingDir: m.currentWorkingDir(),
		Compactor:  m.opts.Compactor,
	}
	opts := coreagent.RunOptions{
		ChatID:       m.chatID,
		Model:        m.opts.Model,
		SystemPrompt: systemPrompt,
		Messages:     slices.Clone(m.messages),
		NewMessages:  slices.Clone(newMessages),
		Format:       m.opts.Format,
		Options:      m.opts.Options,
		Think:        m.opts.Think,
		KeepAlive:    m.opts.KeepAlive,
		UseTools:     m.opts.Tools != nil,
	}

	persistedMessages := make([]api.Message, 0, len(m.messages)+len(newMessages))
	persistedMessages = append(persistedMessages, slices.Clone(m.messages)...)
	persistedMessages = append(persistedMessages, slices.Clone(newMessages)...)
	go func() {
		defer close(events)
		result, err := session.Run(runCtx, opts)
		events <- chatRunDoneMsg{result: result, err: err, newMessagesPersisted: newMessagesPersisted, persistedMessages: persistedMessages}
	}()

	tickCmd := m.scheduleTick()
	flushModel, flushCmd := m.flowTranscriptFlushCmd()
	*m = flushModel
	return *m, tea.Batch(flushCmd, waitForChatMsg(events), tickCmd)
}

func (m *chatModel) startNextQueued() tea.Cmd {
	for len(m.queued) > 0 && !m.running && !m.compacting && !m.quitting && m.resumePicker == nil && m.modelPicker == nil && m.thinkPicker == nil && m.historyPopup == nil {
		input := m.queued[0]
		m.queued = m.queued[1:]
		if len(m.queuedAttachments) > 0 {
			m.inputAttachments = cloneInputAttachments(m.queuedAttachments[0])
			m.queuedAttachments = m.queuedAttachments[1:]
		} else {
			m.inputAttachments = nil
		}
		if len(m.queuedPastedTexts) > 0 {
			m.inputPastedTexts = cloneInputPastedTexts(m.queuedPastedTexts[0])
			m.queuedPastedTexts = m.queuedPastedTexts[1:]
		} else {
			m.inputPastedTexts = nil
		}
		_, cmd := m.submitInput(input)
		if cmd != nil || m.running || m.compacting || m.quitting {
			return cmd
		}
	}
	return nil
}

func (m *chatModel) finishLiveMessagesForStoppedRun(promote bool, persistedMessages []api.Message) {
	if len(m.liveMessages) == 0 {
		return
	}
	if promote {
		if len(persistedMessages) > 0 {
			m.messages = slices.Clone(persistedMessages)
		} else if !messagesHavePendingToolCalls(m.liveMessages) {
			m.messages = slices.Clone(m.liveMessages)
		}
	}
	m.liveMessages = nil
	m.contextTokens = m.estimatePromptTokens(m.messages, "")
	m.contextEstimate = true
}

func messagesHavePendingToolCalls(messages []api.Message) bool {
	pending := map[string]struct{}{}
	for _, msg := range messages {
		if msg.Role == "assistant" {
			for _, call := range msg.ToolCalls {
				pending[call.ID] = struct{}{}
			}
		}
		if msg.Role == "tool" && msg.ToolCallID != "" {
			delete(pending, msg.ToolCallID)
		}
	}
	return len(pending) > 0
}

func (m *chatModel) disarmQuit() {
	if !m.quitArmed {
		return
	}
	m.quitArmed = false
	m.quitArmedKey = ""
	if strings.HasPrefix(m.status, "press ctrl+") && strings.HasSuffix(m.status, "again to quit") {
		m.status = "ready"
	}
}

func (m *chatModel) disarmEsc() {
	if !m.escArmed {
		return
	}
	m.escArmed = false
	if strings.HasPrefix(m.status, "press esc again") {
		m.status = "ready"
	}
}

func (m chatModel) canEditInput() bool {
	return m.approvalPrompt == nil && m.modelPicker == nil && m.thinkPicker == nil && m.resumePicker == nil && m.historyPopup == nil
}

func isChatContextCanceledError(err error) bool {
	return err != nil && (errors.Is(err, context.Canceled) || strings.Contains(err.Error(), "context canceled"))
}

const (
	markdownBlockProse markdownBlockKind = iota
	markdownBlockTable
	markdownBlockCodeFence
	markdownBlockDiffFence
)
