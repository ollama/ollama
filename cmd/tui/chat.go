package tui

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

var chatSpinnerFrames = []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}

const (
	maxResumePickerItems = 8
	maxSlashCompletions  = 5
	maxPromptHistory     = 50
)
const chatCompactionSummaryPrefix = "Conversation summary:\n"

var chatEmptyPrompts = []string{
	`read this repo and tell me where to start`,
	`what changed on this branch?`,
	`run the tests and summarize failures`,
	`find the riskiest code path in this folder`,
	`search the web and compare the latest docs with this implementation`,
	`summarize this file and suggest edits`,
}

type ChatModelOption struct {
	Name        string
	Description string
	Recommended bool
}

type ChatOptions struct {
	Model                       string
	ChatID                      string
	Messages                    []api.Message
	Client                      coreagent.ChatClient
	Store                       coreagent.ChatStore
	Tools                       *coreagent.Registry
	ToolRegistryForModel        func(context.Context, string) *coreagent.Registry
	ModelOptions                func(context.Context) ([]ChatModelOption, error)
	OnModelSelected             func(context.Context, string) error
	SystemPromptForModel        func(context.Context, string, *coreagent.Registry) string
	Clipboard                   func(context.Context, string) error
	Approval                    coreagent.ApprovalHandler
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
	CompactionThreshold         float64
	NewChat                     func(context.Context) (string, error)
	Skills                      *skills.Catalog
	SystemPrompt                string
}

type ChatResult struct {
	ChatID          string
	Messages        []api.Message
	LaunchRequested bool
}

//nolint:containedctx // chatModel is a Bubble Tea session model; the context is the run-scoped cancellation root.
type chatModel struct {
	ctx          context.Context
	opts         ChatOptions
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
	compacting        bool
	cancel            context.CancelFunc
	events            <-chan tea.Msg
	compactEvents     <-chan tea.Msg
	scroll            int
	toolOutputMode    bool
	toolOutputOpen    bool
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
	selection         chatSelection

	width                int
	height               int
	status               string
	spinner              int
	complete             int
	systemPromptDisabled bool
	quitting             bool
	launchRequested      bool
	quitArmed            bool
	escArmed             bool
	eventErrorRendered   bool
	err                  error
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

type chatInputAttachment struct {
	placeholder string
	kind        string
	data        api.ImageData
}

func RunAgentChat(ctx context.Context, opts ChatOptions) (*ChatResult, error) {
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
		status:         "ready",
	}
	m.nextImageID, m.nextAudioID = nextInputAttachmentIDsFromMessages(m.messages)
	m.nextPastedTextID = nextInputPastedTextIDFromMessages(m.messages)
	m.entries = entriesFromMessages(m.messages)
	m.contextTokens = m.estimatePromptTokens(m.messages, "")
	m.contextEstimate = true

	p := tea.NewProgram(m, tea.WithAltScreen(), tea.WithReportFocus(), tea.WithMouseCellMotion())
	finalModel, err := p.Run()
	if err != nil {
		return nil, err
	}

	fm := finalModel.(chatModel)
	if fm.err != nil {
		return nil, fm.err
	}
	return &ChatResult{ChatID: fm.chatID, Messages: fm.messages, LaunchRequested: fm.launchRequested}, nil
}

func (m chatModel) Init() tea.Cmd {
	return nil
}

func (m chatModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		return m, tea.ClearScreen

	case tea.FocusMsg:
		return m, tea.ClearScreen

	case chatTickMsg:
		if !m.running && !m.compacting {
			return m, nil
		}
		m.spinner++
		return m, chatTickCmd()

	case chatAgentMsg:
		m.applyAgentEvent(msg.event)
		return m, waitForChatMsg(m.events)

	case chatApprovalPromptMsg:
		m.resumePicker = nil
		m.modelPicker = nil
		m.historyPopup = nil
		m.openApprovalPrompt(msg)
		return m, nil

	case chatRunDoneMsg:
		wasCanceling := m.status == "canceling"
		m.running = false
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
			m.applyResponseMetrics(&msg.result.Latest)
		}
		m.groupCompletedToolHistory()
		if wasCanceling || isChatContextCanceledError(msg.err) {
			m.status = "Tell the model what to do instead."
			return m, m.startNextQueued()
		}
		if msg.err != nil {
			if !m.eventErrorRendered {
				m.entries = append(m.entries, newChatEntry(chatEntry{role: "error", content: msg.err.Error(), err: msg.err.Error()}))
			}
			m.status = "error"
			return m, nil
		}
		if msg.result != nil {
			m.attachVerboseMetrics(msg.result.Latest.Metrics)
		}
		m.status = "ready"
		return m, m.startNextQueued()

	case chatCompactDoneMsg:
		return m.finishManualCompaction(msg)

	case chatCompactProgressMsg:
		if msg.tokens > m.compactingTokens {
			m.compactingTokens = msg.tokens
		}
		return m, waitForChatMsg(m.compactEvents)

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
	if !m.mouseInTranscript(msg) {
		m.selection = chatSelection{}
		return
	}
	point := m.mouseTranscriptPoint(msg)
	m.selection = chatSelection{active: true, anchor: point, cursor: point}
}

func (m *chatModel) dragTranscriptSelection(msg tea.MouseMsg) {
	if !m.selection.active {
		return
	}
	point := m.mouseTranscriptPoint(msg)
	m.selection.cursor = point
	top, height := m.transcriptLayout()
	if msg.Y <= top {
		m.scrollBy(1)
	} else if msg.Y >= top+height-1 {
		m.scrollBy(-1)
	}
}

func (m chatModel) finishTranscriptSelection(msg tea.MouseMsg) (tea.Model, tea.Cmd) {
	if !m.selection.active {
		return m, nil
	}
	point := m.mouseTranscriptPoint(msg)
	m.selection.cursor = point
	selected := m.selectedTranscriptText(m.viewWidth())
	if strings.TrimSpace(selected) == "" {
		m.selection = chatSelection{}
		return m, nil
	}
	m.status = "selection copied"
	return m, func() tea.Msg {
		if m.opts.Clipboard == nil {
			return nil
		}
		if err := m.opts.Clipboard(m.ctx, selected); err != nil {
			return chatAgentMsg{event: coreagent.Event{Type: coreagent.EventError, Error: err.Error()}}
		}
		return nil
	}
}

func (m chatModel) updateKey(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
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
	if msg.Type != tea.KeyCtrlC {
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
	case tea.KeyPgUp:
		m.scrollBy(m.transcriptHeight() - 1)
	case tea.KeyPgDown:
		m.scrollBy(-(m.transcriptHeight() - 1))
	case tea.KeyHome, tea.KeyCtrlHome:
		m.scroll = m.maxScroll()
	case tea.KeyEnd, tea.KeyCtrlEnd:
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
	case tea.KeyTab:
		m.applyCompletion()
	case tea.KeyCtrlO:
		m.toggleAllToolOutputs()
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

func (m chatModel) updateCtrlC() (tea.Model, tea.Cmd) {
	if len(m.input) > 0 {
		m.clearInput()
		m.resetPromptHistoryCursor()
		m.disarmQuit()
		m.status = "input cleared"
		return m, nil
	}
	if (m.running || m.compacting) && m.cancel != nil {
		m.cancel()
		m.quitArmed = false
		m.status = "canceling"
		return m, nil
	}
	if !m.quitArmed {
		m.quitArmed = true
		m.status = "press ctrl+c again to quit"
		return m, nil
	}
	m.quitting = true
	return m, tea.Quit
}

func (m chatModel) updateCtrlD() (tea.Model, tea.Cmd) {
	if len(m.input) > 0 {
		return m, nil
	}
	m.quitting = true
	if (m.running || m.compacting) && m.cancel != nil {
		m.cancel()
	}
	return m, tea.Quit
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
		m.quitArmed = false
		m.status = "canceling"
		return m, nil
	}
	if cleared {
		m.status = "input cleared"
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

	if m.resumePicker != nil {
		return renderFullFrame(m.renderResumePicker(width), width, height)
	}
	if m.modelPicker != nil {
		return renderFullFrame(m.renderModelPicker(width), width, height)
	}
	if m.thinkPicker != nil {
		return renderFullFrame(m.renderThinkPicker(width), width, height)
	}
	if m.historyPopup != nil {
		return renderFullFrame(m.renderHistoryPopup(width, height), width, height)
	}

	headerLines := m.headerLines()

	bottomLines := m.bottomLines(width, height-len(headerLines))
	available := height - len(headerLines) - len(bottomLines)
	if available < 0 {
		available = 0
	}

	transcriptLines := m.visibleTranscriptLines(width, available)
	if len(transcriptLines) == 0 {
		if available > 0 {
			transcriptLines = []string{chatMetaStyle.Render(m.emptyChatHint())}
		}
	}
	for len(transcriptLines) < available {
		transcriptLines = append(transcriptLines, "")
	}

	lines := append(headerLines, transcriptLines...)
	lines = append(lines, bottomLines...)
	return renderFullFrame(strings.Join(lines, "\n"), width, height)
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
	header := chatHeaderStyle.Render("Ollama")
	if m.opts.Model != "" {
		header += chatMetaStyle.Render("  " + m.opts.Model)
	}
	lines := []string{header}
	if status := m.statusLine(); status != "" {
		lines = append(lines, chatMetaStyle.Render(status))
	}
	return append(lines, "")
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

func (m *chatModel) startManualCompaction() (tea.Model, tea.Cmd) {
	if m.running || m.compacting {
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "system", content: "Wait for the current response to finish before compacting."}))
		return *m, nil
	}
	m.refreshContextWindowTokens(m.opts.Model)
	if m.opts.Compactor == nil {
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "system", content: coreagent.CompactionSkippedMessage("compaction is unavailable")}))
		m.status = "compact skipped"
		return *m, nil
	}

	ctx := m.ctx
	if ctx == nil {
		ctx = context.Background()
	}
	runCtx, cancel := context.WithCancel(ctx)
	compactor := m.opts.Compactor
	events := make(chan tea.Msg, 128)
	m.compacting = true
	m.compactingTokens = 0
	m.cancel = cancel
	m.compactEvents = events
	m.status = "compacting"
	messages := slices.Clone(m.messages)
	var tools api.Tools
	if m.opts.Tools != nil {
		tools = m.opts.Tools.Tools()
	}
	req := coreagent.CompactionRequest{
		ChatID:       m.chatID,
		Model:        m.opts.Model,
		SystemPrompt: m.systemPrompt(""),
		Messages:     messages,
		Tools:        tools,
		Format:       m.opts.Format,
		Options:      m.opts.Options,
		KeepAlive:    m.opts.KeepAlive,
		Force:        true,
		Progress: func(progress coreagent.CompactionProgress) {
			select {
			case events <- chatCompactProgressMsg{tokens: progress.Tokens}:
			case <-runCtx.Done():
			}
		},
	}
	go func() {
		defer close(events)
		result, err := compactor.MaybeCompact(runCtx, req)
		events <- chatCompactDoneMsg{result: result, err: err}
	}()
	return *m, tea.Batch(waitForChatMsg(events), chatTickCmd())
}

func (m chatModel) finishManualCompaction(msg chatCompactDoneMsg) (tea.Model, tea.Cmd) {
	wasCanceling := m.status == "canceling"
	m.compacting = false
	m.compactEvents = nil
	m.cancel = nil
	m.compactingTokens = 0
	if wasCanceling || isChatContextCanceledError(msg.err) {
		m.status = "compact canceled"
		return m, m.startNextQueued()
	}
	if msg.err != nil {
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "system", content: coreagent.CompactionSkippedMessage(msg.err.Error())}))
		m.status = "compact skipped"
		return m, m.startNextQueued()
	}
	if !msg.result.Compacted {
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "system", content: coreagent.CompactionSkippedMessage(msg.result.Reason)}))
		m.status = "compact skipped"
		return m, m.startNextQueued()
	}

	m.messages = msg.result.Messages
	m.liveMessages = nil
	m.entries = entriesFromMessages(m.messages)
	m.contextTokens = m.estimatePromptTokens(m.messages, "")
	m.contextEstimate = true
	m.scroll = 0
	m.status = "compacted"
	return m, m.startNextQueued()
}

func (m *chatModel) startRun(input string) (tea.Model, tea.Cmd) {
	displayInput, message, err := m.userMessageFromInput(input, input)
	if err != nil {
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "error", content: err.Error(), err: err.Error()}))
		m.status = "error"
		return *m, nil
	}
	return m.startRunWithMessages(displayInput, []api.Message{message}, "")
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

func (m *chatModel) startRunWithMessages(displayInput string, newMessages []api.Message, extraSystemPrompt string) (tea.Model, tea.Cmd) {
	m.ensurePermissionMode()
	m.refreshContextWindowTokens(m.opts.Model)
	m.addPromptHistory(displayInput)
	m.entries = append(m.entries, newChatEntry(chatEntry{role: "user", content: displayInput}))
	if len(newMessages) > 1 {
		m.entries = append(m.entries, entriesFromMessages(newMessages[1:])...)
	}
	m.running = true
	m.status = "running"
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

	session := &coreagent.Session{
		Client:     m.opts.Client,
		Store:      m.opts.Store,
		Events:     chatEventSink{ctx: runCtx, ch: events},
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

	go func() {
		defer close(events)
		result, err := session.Run(runCtx, opts)
		events <- chatRunDoneMsg{result: result, err: err}
	}()

	return *m, tea.Batch(waitForChatMsg(events), chatTickCmd())
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

func (m *chatModel) disarmQuit() {
	if !m.quitArmed {
		return
	}
	m.quitArmed = false
	if m.status == "press ctrl+c again to quit" {
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
