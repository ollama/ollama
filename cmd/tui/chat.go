package tui

import (
	"context"
	"errors"
	"slices"
	"strings"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	coreagent "github.com/ollama/ollama/agent"
	"github.com/ollama/ollama/agent/skills"
	"github.com/ollama/ollama/api"
)

var (
	chatHeaderStyle = lipgloss.NewStyle().
			Bold(true)

	chatMetaStyle = lipgloss.NewStyle().
			Foreground(lipgloss.AdaptiveColor{Light: "242", Dark: "246"})

	chatUserStyle = lipgloss.NewStyle().
			Foreground(lipgloss.AdaptiveColor{Light: "236", Dark: "252"})

	chatAssistantStyle = lipgloss.NewStyle().
				Foreground(lipgloss.AdaptiveColor{Light: "236", Dark: "252"})

	chatThinkingStyle = lipgloss.NewStyle().
				Foreground(lipgloss.AdaptiveColor{Light: "244", Dark: "244"}).
				Italic(true)

	chatToolStyle = lipgloss.NewStyle().
			Foreground(lipgloss.AdaptiveColor{Light: "240", Dark: "250"})

	chatToolRunningStyle = lipgloss.NewStyle().
				Foreground(lipgloss.AdaptiveColor{Light: "178", Dark: "222"})

	chatToolDoneStyle = lipgloss.NewStyle().
				Foreground(lipgloss.AdaptiveColor{Light: "28", Dark: "114"})

	chatDiffMetaStyle = lipgloss.NewStyle().
				Foreground(lipgloss.AdaptiveColor{Light: "242", Dark: "246"})

	chatDiffFileStyle = lipgloss.NewStyle().
				Bold(true).
				Foreground(lipgloss.AdaptiveColor{Light: "31", Dark: "117"})

	chatDiffHunkStyle = lipgloss.NewStyle().
				Foreground(lipgloss.AdaptiveColor{Light: "25", Dark: "111"})

	chatDiffAddStyle = lipgloss.NewStyle().
				Foreground(lipgloss.AdaptiveColor{Light: "28", Dark: "114"})

	chatDiffDeleteStyle = lipgloss.NewStyle().
				Foreground(lipgloss.AdaptiveColor{Light: "160", Dark: "203"})

	chatErrorStyle = lipgloss.NewStyle().
			Foreground(lipgloss.AdaptiveColor{Light: "160", Dark: "203"})

	chatFullAccessStyle = lipgloss.NewStyle().
				Bold(true).
				Foreground(lipgloss.AdaptiveColor{Light: "160", Dark: "203"})

	chatInputStyle = lipgloss.NewStyle().
			Foreground(lipgloss.AdaptiveColor{Light: "236", Dark: "252"})

	chatInputBorderStyle = lipgloss.NewStyle().
				Foreground(lipgloss.AdaptiveColor{Light: "248", Dark: "244"})

	chatCommandNameStyle = lipgloss.NewStyle().
				Foreground(lipgloss.AdaptiveColor{Light: "0", Dark: "15"})

	chatResumeTextStyle = lipgloss.NewStyle().
				Foreground(lipgloss.AdaptiveColor{Light: "0", Dark: "15"})

	chatResumeTitleStyle = lipgloss.NewStyle().
				Bold(true).
				Foreground(lipgloss.AdaptiveColor{Light: "0", Dark: "15"})

	chatResumeSelectedStyle = lipgloss.NewStyle().
				Bold(true).
				Foreground(lipgloss.AdaptiveColor{Light: "0", Dark: "15"})

	chatResumeMetaStyle = lipgloss.NewStyle().
				Foreground(lipgloss.AdaptiveColor{Light: "8", Dark: "7"})

	chatResumeBorderStyle = lipgloss.NewStyle().
				Foreground(lipgloss.AdaptiveColor{Light: "8", Dark: "7"})

	chatHistoryTitleStyle = lipgloss.NewStyle().
				Bold(true).
				Foreground(lipgloss.AdaptiveColor{Light: "0", Dark: "15"})

	chatHistorySystemRoleStyle = lipgloss.NewStyle().
					Bold(true).
					Foreground(lipgloss.AdaptiveColor{Light: "242", Dark: "248"})

	chatHistoryUserRoleStyle = lipgloss.NewStyle().
					Bold(true).
					Foreground(lipgloss.AdaptiveColor{Light: "25", Dark: "117"})

	chatHistoryAssistantRoleStyle = lipgloss.NewStyle().
					Bold(true).
					Foreground(lipgloss.AdaptiveColor{Light: "136", Dark: "222"})

	chatHistoryToolRoleStyle = lipgloss.NewStyle().
					Bold(true).
					Foreground(lipgloss.AdaptiveColor{Light: "28", Dark: "114"})

	chatHistoryLabelStyle = lipgloss.NewStyle().
				Foreground(lipgloss.AdaptiveColor{Light: "244", Dark: "244"})

	chatHistoryTextStyle = lipgloss.NewStyle().
				Foreground(lipgloss.AdaptiveColor{Light: "236", Dark: "252"})

	chatHistoryCodeStyle = lipgloss.NewStyle().
				Foreground(lipgloss.AdaptiveColor{Light: "31", Dark: "117"})
)

var chatSpinnerFrames = []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}

const (
	maxResumePickerItems = 8
	maxPromptHistory     = 50
)
const chatCompactionSummaryPrefix = "Conversation summary:\n"

type ChatOptions struct {
	Model                       string
	ChatID                      string
	Messages                    []api.Message
	Client                      coreagent.ChatClient
	Store                       coreagent.ChatStore
	Tools                       *coreagent.Registry
	ToolRegistryForModel        func(context.Context, string) *coreagent.Registry
	Approval                    coreagent.ApprovalHandler
	AutoApproveTools            bool
	WorkingDir                  string
	RootDir                     string
	Format                      string
	Options                     map[string]any
	Think                       *api.ThinkValue
	KeepAlive                   *api.Duration
	HideThinking                bool
	Compactor                   coreagent.Compactor
	ContextWindowTokens         int
	ContextWindowTokensForModel func(context.Context, string, int) int
	CompactionThreshold         float64
	NewChat                     func(context.Context) (string, error)
	Skills                      *skills.Catalog
	SystemPrompt                string
}

type ChatResult struct {
	ChatID   string
	Messages []api.Message
}

type chatModel struct {
	ctx        context.Context
	opts       ChatOptions
	chatID     string
	messages   []api.Message
	entries    []chatEntry
	workingDir string

	input            []rune
	queued           []string
	promptHistory    []string
	promptCursor     int
	promptDraft      []rune
	promptActive     bool
	running          bool
	compacting       bool
	cancel           context.CancelFunc
	events           <-chan tea.Msg
	compactEvents    <-chan tea.Msg
	scroll           int
	toolOutputMode   bool
	toolOutputOpen   bool
	thinking         bool
	thinkingTokens   int
	compactingTokens int
	contextTokens    int
	contextEstimate  bool
	resumePicker     *chatResumePicker
	historyPopup     *chatHistoryPopup
	approvalPrompt   *chatApprovalPrompt
	reviewApproval   coreagent.ApprovalHandler
	permissionMode   *chatPermissionMode

	width    int
	height   int
	status   string
	spinner  int
	complete int
	quitting bool
	err      error
}

func RunAgentChat(ctx context.Context, opts ChatOptions) (*ChatResult, error) {
	if opts.Approval == nil {
		opts.Approval = coreagent.NewApprovalManager(coreagent.ApprovalManagerOptions{})
	}
	if opts.RootDir == "" {
		opts.RootDir = opts.WorkingDir
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
	return &ChatResult{ChatID: fm.chatID, Messages: fm.messages}, nil
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
		m.historyPopup = nil
		m.openApprovalPrompt(msg)
		return m, nil

	case chatRunDoneMsg:
		wasCanceling := m.status == "canceling"
		m.running = false
		m.cancel = nil
		m.events = nil
		m.thinking = false
		m.thinkingTokens = 0
		m.approvalPrompt = nil
		if msg.result != nil {
			m.messages = msg.result.Messages
			if msg.result.WorkingDir != "" {
				m.workingDir = msg.result.WorkingDir
			}
			m.refreshContextWindowTokens(m.responseModelName(&msg.result.Latest))
			m.contextTokens = m.estimatePromptTokens(m.messages, "")
			m.contextEstimate = true
			m.applyResponseMetrics(&msg.result.Latest)
		}
		m.groupCompletedToolHistory()
		if wasCanceling || errors.Is(msg.err, context.Canceled) {
			m.status = "Tell the model what to do instead."
			return m, m.startNextQueued()
		}
		if msg.err != nil {
			m.entries = append(m.entries, newChatEntry(chatEntry{role: "error", content: msg.err.Error(), err: msg.err.Error()}))
			m.status = "error"
			return m, nil
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

	case tea.MouseMsg:
		if m.historyPopup != nil {
			switch msg.Type {
			case tea.MouseWheelUp:
				m.moveHistoryPopup(-3)
			case tea.MouseWheelDown:
				m.moveHistoryPopup(3)
			}
			return m, nil
		}
		switch msg.Type {
		case tea.MouseWheelUp:
			m.scrollBy(3)
		case tea.MouseWheelDown:
			m.scrollBy(-3)
		}
		return m, nil

	case tea.KeyMsg:
		if msg.Type == tea.KeyShiftTab {
			return m.togglePermissionMode()
		}
		if m.approvalPrompt != nil {
			return m.updateApprovalPrompt(msg)
		}
		if m.resumePicker != nil {
			return m.updateResumePicker(msg)
		}
		if m.historyPopup != nil {
			return m.updateHistoryPopup(msg)
		}

		switch msg.Type {
		case tea.KeyCtrlC:
			if (m.running || m.compacting) && m.cancel != nil {
				m.cancel()
				m.status = "canceling"
				return m, nil
			}
			m.quitting = true
			return m, tea.Quit
		case tea.KeyEsc:
			if (m.running || m.compacting) && m.cancel != nil {
				m.cancel()
				m.status = "canceling"
				return m, nil
			}
			return m, nil
		case tea.KeyEnter:
			return m.handleSubmit()
		case tea.KeyUp:
			if m.running || m.compacting {
				m.scrollBy(1)
				return m, nil
			}
			if m.promptActive && m.movePromptHistory(-1) {
				return m, nil
			}
			if m.moveCompletion(-1) {
				return m, nil
			}
			if m.movePromptHistory(-1) {
				return m, nil
			}
			m.scrollBy(1)
			return m, nil
		case tea.KeyDown:
			if m.running || m.compacting {
				m.scrollBy(-1)
				return m, nil
			}
			if m.promptActive && m.movePromptHistory(1) {
				return m, nil
			}
			if m.moveCompletion(1) {
				return m, nil
			}
			if m.movePromptHistory(1) {
				return m, nil
			}
			m.scrollBy(-1)
			return m, nil
		case tea.KeyPgUp:
			m.scrollBy(m.transcriptHeight() - 1)
			return m, nil
		case tea.KeyPgDown:
			m.scrollBy(-(m.transcriptHeight() - 1))
			return m, nil
		case tea.KeyHome, tea.KeyCtrlHome:
			m.scroll = m.maxScroll()
			return m, nil
		case tea.KeyEnd, tea.KeyCtrlEnd:
			m.scroll = 0
			return m, nil
		case tea.KeyBackspace:
			m.resetPromptHistoryCursor()
			if len(m.input) > 0 {
				m.input = m.input[:len(m.input)-1]
				m.complete = 0
			}
			return m, nil
		case tea.KeyCtrlU:
			m.resetPromptHistoryCursor()
			m.input = nil
			m.complete = 0
			return m, nil
		case tea.KeyTab:
			if m.applyCompletion() {
				return m, nil
			}
			return m, nil
		case tea.KeyCtrlO:
			m.toggleAllToolOutputs()
			return m, nil
		case tea.KeySpace:
			m.resetPromptHistoryCursor()
			m.input = append(m.input, ' ')
			m.complete = 0
			return m, nil
		case tea.KeyRunes:
			m.resetPromptHistoryCursor()
			m.input = append(m.input, msg.Runes...)
			m.complete = 0
			return m, nil
		}
	}
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
	if m.historyPopup != nil {
		return renderFullFrame(m.renderHistoryPopup(width, height), width, height)
	}

	header := chatHeaderStyle.Render("Ollama")
	if m.opts.Model != "" {
		header += chatMetaStyle.Render("  " + m.opts.Model)
	}
	headerLines := []string{header}
	if status := m.statusLine(); status != "" {
		headerLines = append(headerLines, chatMetaStyle.Render(status))
	}
	headerLines = append(headerLines, "")

	bottomLines := m.bottomLines(width)
	available := height - len(headerLines) - len(bottomLines)
	if available < 0 {
		available = 0
	}

	transcriptLines := m.visibleTranscriptLines(width, available)
	if len(transcriptLines) == 0 {
		if available > 0 {
			transcriptLines = []string{chatMetaStyle.Render("Start a conversation. Use /help for commands.")}
		}
	}
	for len(transcriptLines) < available {
		transcriptLines = append(transcriptLines, "")
	}

	lines := append(headerLines, transcriptLines...)
	lines = append(lines, bottomLines...)
	return renderFullFrame(strings.Join(lines, "\n"), width, height)
}

func (m *chatModel) resetChat(status string) (tea.Model, tea.Cmd) {
	m.messages = nil
	m.entries = nil
	m.queued = nil
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
	req := coreagent.CompactionRequest{
		ChatID:    m.chatID,
		Model:     m.opts.Model,
		Messages:  messages,
		Options:   m.opts.Options,
		KeepAlive: m.opts.KeepAlive,
		Force:     true,
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
	if wasCanceling || errors.Is(msg.err, context.Canceled) {
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
	m.entries = entriesFromMessages(m.messages)
	m.contextTokens = m.estimatePromptTokens(m.messages, "")
	m.contextEstimate = true
	m.scroll = 0
	m.status = "compacted"
	return m, m.startNextQueued()
}

func (m *chatModel) startRun(input string) (tea.Model, tea.Cmd) {
	return m.startRunWithPrompt(input, input, "")
}

func (m *chatModel) startRunWithPrompt(displayInput, userInput, extraSystemPrompt string) (tea.Model, tea.Cmd) {
	m.ensurePermissionMode()
	m.refreshContextWindowTokens(m.opts.Model)
	m.addPromptHistory(displayInput)
	userMsg := api.Message{Role: "user", Content: userInput}
	m.entries = append(m.entries, newChatEntry(chatEntry{role: "user", content: displayInput}))
	m.running = true
	m.status = "running"
	m.scroll = 0
	m.thinking = false
	m.thinkingTokens = 0
	systemPrompt := m.systemPrompt(extraSystemPrompt)
	m.contextTokens = m.estimatePromptTokens(append(slices.Clone(m.messages), userMsg), systemPrompt)
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
		NewMessages:  []api.Message{userMsg},
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
	for len(m.queued) > 0 && !m.running && !m.compacting && !m.quitting && m.resumePicker == nil && m.historyPopup == nil {
		input := m.queued[0]
		m.queued = m.queued[1:]
		_, cmd := m.submitInput(input)
		if cmd != nil || m.running || m.compacting || m.quitting {
			return cmd
		}
	}
	return nil
}

const (
	markdownBlockProse markdownBlockKind = iota
	markdownBlockTable
	markdownBlockCodeFence
	markdownBlockDiffFence
)
