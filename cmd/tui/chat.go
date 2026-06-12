package tui

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/glamour"
	glamouransi "github.com/charmbracelet/glamour/ansi"
	"github.com/charmbracelet/glamour/styles"
	"github.com/charmbracelet/lipgloss"

	coreagent "github.com/ollama/ollama/agent"
	"github.com/ollama/ollama/agent/chatstore"
	"github.com/ollama/ollama/agent/skills"
	agenttools "github.com/ollama/ollama/agent/tools"
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

var markdownLinkPattern = regexp.MustCompile(`\[([^\]]+)\]\((https?://[^)\s]+)\)`)
var markdownTableSeparatorPattern = regexp.MustCompile(`^:?-{3,}:?$`)

var chatMarkdownRenderers sync.Map

const (
	maxResumePickerItems = 8
	maxPromptHistory     = 50
)
const chatCompactionSummaryPrefix = "Conversation summary:\n"

type cachedMarkdownRenderer struct {
	renderer *glamour.TermRenderer
	mu       sync.Mutex
}

type chatApprovalChoice struct {
	label    string
	key      string
	decision coreagent.ApprovalDecision
	reason   string
}

var chatApprovalChoices = []chatApprovalChoice{
	{label: "Approve once", key: "o", decision: coreagent.ApprovalAllowOnce},
	{label: "Approve session", key: "s", decision: coreagent.ApprovalAllowSession},
	{label: "Deny", key: "d", decision: coreagent.ApprovalDeny, reason: "Tool execution denied."},
}

type chatSlashCommand struct {
	name        string
	description string
}

type chatCompletion struct {
	value       string
	label       string
	description string
	directory   bool
}

var chatSlashCommands = []chatSlashCommand{
	{name: "/clear", description: "clear this chat"},
	{name: "/tools", description: "show available tools"},
	{name: "/history", description: "show prompt message history"},
	{name: "/skills", description: "show or import installed skills"},
	{name: "/new", description: "start a new chat"},
	{name: "/resume", description: "resume a saved chat"},
	{name: "/compact", description: "summarize older context"},
	{name: "/help", description: "show commands"},
	{name: "/bye", description: "exit"},
}

type chatResumeStore interface {
	ListChats(context.Context, int) ([]chatstore.ChatSummary, error)
	Chat(context.Context, string) (*chatstore.Chat, error)
}

type chatPromptHistoryStore interface {
	ListUserMessages(context.Context, int) ([]string, error)
}

type chatResumePicker struct {
	chats  []chatstore.ChatSummary
	filter string
	cursor int
	scroll int
}

type chatHistoryPopup struct {
	content       string
	scroll        int
	stickToBottom bool
}

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

type chatEntry struct {
	role       string
	content    string
	label      string
	detail     string
	status     string
	err        string
	toolID     string
	args       map[string]any
	expanded   bool
	startedAt  time.Time
	finishedAt time.Time
	tools      []chatEntry

	version     int
	renderKey   chatEntryRenderKey
	renderLines []string
}

type chatEntryRenderKey struct {
	width   int
	version int
	hash    string
}

type chatAgentMsg struct {
	event coreagent.Event
}

type chatApprovalPromptMsg struct {
	request coreagent.ApprovalRequest
	reply   chan<- coreagent.ApprovalResult
}

type chatRunDoneMsg struct {
	result *coreagent.RunResult
	err    error
}

type chatCompactDoneMsg struct {
	result coreagent.CompactionResult
	err    error
}

type chatCompactProgressMsg struct {
	tokens int
}

type chatTickMsg struct{}

type chatModel struct {
	ctx      context.Context
	opts     ChatOptions
	chatID   string
	messages []api.Message
	entries  []chatEntry

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

type chatApprovalPrompt struct {
	request coreagent.ApprovalRequest
	reply   chan<- coreagent.ApprovalResult
	cursor  int
}

type chatPermissionMode struct {
	mu          sync.Mutex
	autoApprove bool
}

func newChatPermissionMode(autoApprove bool) *chatPermissionMode {
	return &chatPermissionMode{autoApprove: autoApprove}
}

func (m *chatPermissionMode) AutoApprove() bool {
	if m == nil {
		return false
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.autoApprove
}

func (m *chatPermissionMode) SetAutoApprove(autoApprove bool) {
	if m == nil {
		return
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.autoApprove = autoApprove
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
		reviewApproval: reviewApproval,
		permissionMode: newChatPermissionMode(autoApproveTools),
		promptHistory:  initialPromptHistory(ctx, opts),
		status:         "ready",
	}
	m.entries = entriesFromMessages(m.messages)
	m.contextTokens = m.estimatePromptTokens(m.messages, "")
	m.contextEstimate = true

	p := tea.NewProgram(m, tea.WithAltScreen(), tea.WithReportFocus())
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
				m.opts.WorkingDir = msg.result.WorkingDir
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

func (m *chatModel) handleSubmit() (tea.Model, tea.Cmd) {
	input := strings.TrimSpace(string(m.input))
	if selected, ok := m.selectedSlashCommand(); ok {
		input = selected
	}
	m.input = nil
	m.complete = 0
	m.resetPromptHistoryCursor()
	if input == "" {
		return *m, nil
	}

	if m.running || m.compacting {
		m.queued = append(m.queued, input)
		m.status = "queued"
		return *m, nil
	}

	return m.submitInput(input)
}

func (m chatModel) selectedSlashCommand() (string, bool) {
	input := strings.TrimSpace(string(m.input))
	if !strings.HasPrefix(input, "/") {
		return "", false
	}
	completions := m.slashCompletions()
	if len(completions) == 0 || !completionIsSelectable(completions) {
		return "", false
	}
	return completions[clamp(m.complete, 0, len(completions)-1)].value, true
}

func (m *chatModel) submitInput(input string) (tea.Model, tea.Cmd) {
	switch {
	case input == "/bye" || input == "/exit":
		m.quitting = true
		return *m, tea.Quit
	case input == "/?" || input == "/help":
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "system", content: m.helpSummary()}))
		return *m, nil
	case input == "/clear":
		return m.resetChat("cleared")
	case input == "/tools":
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "system", content: m.toolsSummary()}))
		return *m, nil
	case input == "/history":
		return m.openHistoryPopup()
	case input == "/skills" || strings.HasPrefix(input, "/skills "):
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "system", content: m.handleSkillsCommand(input)}))
		return *m, nil
	case input == "/new":
		return m.resetChat("new chat")
	case input == "/resume":
		return m.openResumePicker()
	case input == "/compact":
		return m.startManualCompaction()
	case strings.HasPrefix(input, "/"):
		if skill, request, ok := m.skillTrigger(input); ok {
			manualPrompt, err := skills.ManualSystemPrompt(skill)
			if err != nil {
				m.entries = append(m.entries, newChatEntry(chatEntry{role: "error", content: err.Error(), err: err.Error()}))
				return *m, nil
			}
			return m.startRunWithPrompt(input, skills.ManualUserPrompt(skill.Name, request), manualPrompt)
		}
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "error", content: fmt.Sprintf("Unknown command %q", strings.Fields(input)[0])}))
		return *m, nil
	}

	return m.startRun(input)
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
		m.opts.WorkingDir = m.opts.RootDir
	}
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
		WorkingDir: m.opts.WorkingDir,
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

func (m chatModel) approvalHandlerForRun(events chan<- tea.Msg) coreagent.ApprovalHandler {
	return chatPermissionApprovalHandler{
		mode:   m.permissionMode,
		review: approvalHandlerForRun(m.reviewApproval, events),
	}
}

func approvalHandlerForRun(handler coreagent.ApprovalHandler, events chan<- tea.Msg) coreagent.ApprovalHandler {
	prompter := chatApprovalPrompter{ch: events}
	if manager, ok := handler.(*coreagent.ApprovalManager); ok {
		return manager.WithPrompter(prompter)
	}
	if handler != nil {
		return handler
	}
	return coreagent.NewApprovalManager(coreagent.ApprovalManagerOptions{Prompter: prompter})
}

func chatReviewApprovalHandler(handler coreagent.ApprovalHandler) coreagent.ApprovalHandler {
	if handler == nil || approvalHandlerAutoApproves(handler) {
		return coreagent.NewApprovalManager(coreagent.ApprovalManagerOptions{})
	}
	return handler
}

func approvalHandlerAutoApproves(handler coreagent.ApprovalHandler) bool {
	switch h := handler.(type) {
	case coreagent.AutoAllowApproval:
		return true
	case *coreagent.AutoAllowApproval:
		return h != nil
	case interface{ AutoApproveEnabled() bool }:
		return h.AutoApproveEnabled()
	default:
		return false
	}
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

func initialPromptHistory(ctx context.Context, opts ChatOptions) []string {
	if ctx == nil {
		ctx = context.Background()
	}
	if store, ok := opts.Store.(chatPromptHistoryStore); ok && store != nil {
		prompts, err := store.ListUserMessages(ctx, maxPromptHistory)
		if err == nil {
			return normalizePromptHistory(prompts)
		}
	}

	var prompts []string
	for _, msg := range opts.Messages {
		if msg.Role == "user" {
			prompts = append(prompts, msg.Content)
		}
	}
	return normalizePromptHistory(prompts)
}

func normalizePromptHistory(prompts []string) []string {
	history := make([]string, 0, min(len(prompts), maxPromptHistory))
	for _, prompt := range prompts {
		prompt = strings.TrimSpace(prompt)
		if prompt == "" || strings.HasPrefix(prompt, chatCompactionSummaryPrefix) {
			continue
		}
		history = append(history, prompt)
	}
	if len(history) > maxPromptHistory {
		history = history[len(history)-maxPromptHistory:]
	}
	return history
}

func (m *chatModel) addPromptHistory(prompt string) {
	prompt = strings.TrimSpace(prompt)
	if prompt == "" {
		return
	}
	m.promptHistory = append(m.promptHistory, prompt)
	if len(m.promptHistory) > maxPromptHistory {
		m.promptHistory = m.promptHistory[len(m.promptHistory)-maxPromptHistory:]
	}
	m.resetPromptHistoryCursor()
}

func (m *chatModel) movePromptHistory(delta int) bool {
	if len(m.promptHistory) == 0 || delta == 0 {
		return false
	}
	if !m.promptActive {
		if delta > 0 {
			return false
		}
		m.promptDraft = slices.Clone(m.input)
		m.promptCursor = len(m.promptHistory) - 1
		m.promptActive = true
	} else {
		m.promptCursor += delta
		if m.promptCursor >= len(m.promptHistory) {
			m.input = slices.Clone(m.promptDraft)
			m.resetPromptHistoryCursor()
			m.complete = 0
			return true
		}
		if m.promptCursor < 0 {
			m.promptCursor = 0
		}
	}

	m.input = []rune(m.promptHistory[m.promptCursor])
	m.complete = 0
	return true
}

func (m *chatModel) resetPromptHistoryCursor() {
	m.promptActive = false
	m.promptCursor = 0
	m.promptDraft = nil
}

func (m *chatModel) openHistoryPopup() (tea.Model, tea.Cmd) {
	m.historyPopup = &chatHistoryPopup{content: m.historySummary(), stickToBottom: true}
	m.status = "history"
	return *m, nil
}

func (m chatModel) updateHistoryPopup(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch msg.Type {
	case tea.KeyCtrlC, tea.KeyEsc:
		m.historyPopup = nil
		m.status = "ready"
		return m, nil
	case tea.KeyUp:
		m.moveHistoryPopup(-1)
	case tea.KeyDown:
		m.moveHistoryPopup(1)
	case tea.KeyPgUp:
		m.moveHistoryPopup(-m.historyPopupVisibleHeight())
	case tea.KeyPgDown:
		m.moveHistoryPopup(m.historyPopupVisibleHeight())
	case tea.KeyHome, tea.KeyCtrlHome:
		if m.historyPopup != nil {
			m.historyPopup.scroll = 0
			m.historyPopup.stickToBottom = false
		}
	case tea.KeyEnd, tea.KeyCtrlEnd:
		if m.historyPopup != nil {
			m.historyPopup.scroll = m.historyPopupMaxScroll()
			m.historyPopup.stickToBottom = true
		}
	}
	return m, nil
}

func (m *chatModel) moveHistoryPopup(delta int) {
	if m.historyPopup == nil || delta == 0 {
		return
	}
	if m.historyPopup.stickToBottom {
		m.historyPopup.scroll = m.historyPopupMaxScroll()
		m.historyPopup.stickToBottom = false
	}
	m.historyPopup.scroll = clamp(m.historyPopup.scroll+delta, 0, m.historyPopupMaxScroll())
}

func (m chatModel) historyPopupMaxScroll() int {
	return max(0, len(m.historyPopupBodyLines())-m.historyPopupVisibleHeight())
}

func (m chatModel) historyPopupVisibleHeight() int {
	height := m.height
	if height <= 0 {
		height = 24
	}
	return max(1, height-4)
}

func (m chatModel) historyPopupBodyLines() []string {
	width := m.width
	if width <= 0 {
		width = 80
	}
	return m.historyPopupBodyLinesForWidth(width)
}

func (m chatModel) historyPopupBodyLinesForWidth(width int) []string {
	if m.historyPopup == nil {
		return nil
	}
	if width <= 0 {
		width = 80
	}
	content := strings.TrimPrefix(m.historyPopup.content, "**Message History**\n\n")
	return renderHistoryLines(content, width)
}

func (m *chatModel) openResumePicker() (tea.Model, tea.Cmd) {
	store, ok := m.opts.Store.(chatResumeStore)
	if !ok || store == nil {
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "error", content: "Chat resume is unavailable because persistence is disabled."}))
		m.status = "error"
		return *m, nil
	}

	chats, err := store.ListChats(m.ctx, 50)
	if err != nil {
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "error", content: fmt.Sprintf("Could not list saved chats: %v", err)}))
		m.status = "error"
		return *m, nil
	}
	if len(chats) == 0 {
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "system", content: "No saved chats to resume."}))
		m.status = "ready"
		return *m, nil
	}

	m.resumePicker = newChatResumePicker(chats, m.chatID)
	m.status = "resume"
	return *m, nil
}

func newChatResumePicker(chats []chatstore.ChatSummary, currentChatID string) *chatResumePicker {
	picker := &chatResumePicker{chats: slices.Clone(chats)}
	for i, chat := range picker.chats {
		if chat.ID == currentChatID {
			picker.cursor = i
			picker.updateScroll()
			break
		}
	}
	return picker
}

func (m chatModel) updateResumePicker(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch msg.Type {
	case tea.KeyCtrlC, tea.KeyEsc:
		m.resumePicker = nil
		m.status = "ready"
		return m, nil
	case tea.KeyEnter:
		return m.resumeSelectedChat()
	case tea.KeyUp:
		m.resumePicker.move(-1)
	case tea.KeyDown:
		m.resumePicker.move(1)
	case tea.KeyPgUp:
		m.resumePicker.move(-maxResumePickerItems)
	case tea.KeyPgDown:
		m.resumePicker.move(maxResumePickerItems)
	case tea.KeyBackspace:
		if len(m.resumePicker.filter) > 0 {
			runes := []rune(m.resumePicker.filter)
			m.resumePicker.filter = string(runes[:len(runes)-1])
			m.resumePicker.cursor = 0
			m.resumePicker.scroll = 0
		}
	case tea.KeyCtrlU:
		m.resumePicker.filter = ""
		m.resumePicker.cursor = 0
		m.resumePicker.scroll = 0
	case tea.KeySpace:
		m.resumePicker.filter += " "
		m.resumePicker.cursor = 0
		m.resumePicker.scroll = 0
	case tea.KeyRunes:
		m.resumePicker.filter += string(msg.Runes)
		m.resumePicker.cursor = 0
		m.resumePicker.scroll = 0
	}
	return m, nil
}

func (m *chatModel) resumeSelectedChat() (tea.Model, tea.Cmd) {
	if m.resumePicker == nil {
		return *m, nil
	}
	selected, ok := m.resumePicker.selected()
	if !ok {
		return *m, nil
	}
	store, ok := m.opts.Store.(chatResumeStore)
	if !ok || store == nil {
		m.resumePicker = nil
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "error", content: "Chat resume is unavailable because persistence is disabled."}))
		m.status = "error"
		return *m, nil
	}

	chat, err := store.Chat(m.ctx, selected.ID)
	if err != nil {
		m.resumePicker = nil
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "error", content: fmt.Sprintf("Could not resume chat: %v", err)}))
		m.status = "error"
		return *m, nil
	}

	m.resumePicker = nil
	m.chatID = chat.ID
	if chat.Model != "" && chat.Model != m.opts.Model {
		m.opts.Model = chat.Model
		if m.opts.ToolRegistryForModel != nil {
			m.opts.Tools = m.opts.ToolRegistryForModel(m.ctx, chat.Model)
		}
	}
	m.refreshContextWindowTokens(m.opts.Model)
	m.messages = slices.Clone(chat.Messages)
	m.entries = entriesFromMessages(m.messages)
	m.input = nil
	m.queued = nil
	m.resetWorkingDir()
	m.complete = 0
	m.thinking = false
	m.thinkingTokens = 0
	m.contextTokens = m.estimatePromptTokens(m.messages, "")
	m.contextEstimate = true
	m.scroll = 0
	m.status = "resumed"
	return *m, nil
}

func (m *chatModel) openApprovalPrompt(msg chatApprovalPromptMsg) {
	m.approvalPrompt = &chatApprovalPrompt{request: msg.request, reply: msg.reply}
	m.status = "approval required"
	m.thinking = false
	m.thinkingTokens = 0
	m.upsertApprovalToolEntry(msg.request)
}

func (m *chatModel) togglePermissionMode() (tea.Model, tea.Cmd) {
	m.ensurePermissionMode()
	autoApprove := !m.permissionMode.AutoApprove()
	m.permissionMode.SetAutoApprove(autoApprove)
	m.opts.AutoApproveTools = autoApprove
	if autoApprove {
		m.status = "ready"
		if m.approvalPrompt != nil {
			return m.resolveApprovalPrompt(coreagent.ApprovalAllowOnce, "")
		}
		return *m, nil
	}
	m.status = "ready"
	return *m, nil
}

func (m *chatModel) ensurePermissionMode() {
	if m.permissionMode == nil {
		m.permissionMode = newChatPermissionMode(m.opts.AutoApproveTools || approvalHandlerAutoApproves(m.opts.Approval))
	}
	if m.reviewApproval == nil {
		m.reviewApproval = chatReviewApprovalHandler(m.opts.Approval)
	}
}

func (m chatModel) autoApproveTools() bool {
	if m.permissionMode != nil {
		return m.permissionMode.AutoApprove()
	}
	return m.opts.AutoApproveTools || approvalHandlerAutoApproves(m.opts.Approval)
}

func (m *chatModel) upsertApprovalToolEntry(request coreagent.ApprovalRequest) {
	idx := m.findToolEntry(request.ToolCallID)
	if idx < 0 {
		m.groupCompletedToolHistory()
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "tool"}))
		idx = len(m.entries) - 1
	}
	m.entries[idx].detail = request.ToolName
	m.entries[idx].label = toolInvocationLabel(request.ToolName, request.Args)
	m.entries[idx].status = "approval"
	m.entries[idx].toolID = request.ToolCallID
	m.entries[idx].args = request.Args
	m.entries[idx].startedAt = time.Now()
	m.applyToolOutputModeTo(idx)
	m.markEntryDirty(idx)
}

func (m chatModel) updateApprovalPrompt(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch msg.Type {
	case tea.KeyLeft, tea.KeyUp:
		m.moveApprovalChoice(-1)
	case tea.KeyRight, tea.KeyDown, tea.KeyTab:
		m.moveApprovalChoice(1)
	case tea.KeyRunes:
		switch strings.ToLower(string(msg.Runes)) {
		case "o":
			return m.resolveApprovalPrompt(coreagent.ApprovalAllowOnce, "")
		case "s":
			return m.resolveApprovalPrompt(coreagent.ApprovalAllowSession, "")
		case "d":
			return m.resolveApprovalPrompt(coreagent.ApprovalDeny, "Tool execution denied.")
		}
	case tea.KeyEnter:
		choice := chatApprovalChoices[clamp(m.approvalPrompt.cursor, 0, len(chatApprovalChoices)-1)]
		return m.resolveApprovalPrompt(choice.decision, choice.reason)
	case tea.KeyEsc, tea.KeyCtrlC:
		return m.resolveApprovalPrompt(coreagent.ApprovalDeny, "Tool execution denied.")
	}
	return m, nil
}

func (m *chatModel) moveApprovalChoice(delta int) {
	if m.approvalPrompt == nil {
		return
	}
	m.approvalPrompt.cursor = (m.approvalPrompt.cursor + delta) % len(chatApprovalChoices)
	if m.approvalPrompt.cursor < 0 {
		m.approvalPrompt.cursor += len(chatApprovalChoices)
	}
	m.markApprovalPromptEntryDirty()
}

func (m *chatModel) markApprovalPromptEntryDirty() {
	if m.approvalPrompt == nil {
		return
	}
	if idx := m.findToolEntry(m.approvalPrompt.request.ToolCallID); idx >= 0 {
		m.markEntryDirty(idx)
	}
}

func (m chatModel) resolveApprovalPrompt(decision coreagent.ApprovalDecision, reason string) (tea.Model, tea.Cmd) {
	if m.approvalPrompt == nil {
		return m, nil
	}
	prompt := m.approvalPrompt
	m.approvalPrompt = nil
	m.status = "running"
	if decision == coreagent.ApprovalDeny {
		m.status = "denied"
	}
	if idx := m.findToolEntry(prompt.request.ToolCallID); idx >= 0 && m.entries[idx].status == "approval" {
		if decision == coreagent.ApprovalDeny {
			m.entries[idx].status = "error"
			m.entries[idx].err = reason
			if m.entries[idx].err == "" {
				m.entries[idx].err = "Tool execution denied."
			}
		} else {
			m.entries[idx].status = "queued"
		}
		m.markEntryDirty(idx)
	}
	prompt.reply <- coreagent.ApprovalResult{Decision: decision, Reason: reason}
	return m, waitForChatMsg(m.events)
}

func (m *chatModel) applyAgentEvent(event coreagent.Event) {
	m.applyResponseMetrics(event.Response)

	switch event.Type {
	case coreagent.EventMessageStarted:
		m.thinking = false
		m.thinkingTokens = 0
	case coreagent.EventThinkingDelta:
		if event.Thinking != "" {
			m.thinking = true
			m.thinkingTokens = max(m.thinkingTokens, eventEvalCount(event))
			if eventEvalCount(event) <= 0 {
				m.thinkingTokens += estimateTokenCount(event.Thinking)
			}
		}
	case coreagent.EventMessageDelta:
		m.thinking = false
		m.thinkingTokens = 0
		m.groupCompletedToolHistory()
		idx := m.ensureAssistantEntry()
		m.entries[idx].content += event.Content
		m.markEntryDirty(idx)
	case coreagent.EventToolCallDetected:
		m.thinking = false
		m.thinkingTokens = 0
	case coreagent.EventToolStarted:
		m.thinking = false
		m.thinkingTokens = 0
		idx := m.findActiveToolEntry(event.ToolCallID)
		if idx < 0 {
			m.groupCompletedToolHistory()
			m.entries = append(m.entries, newChatEntry(chatEntry{role: "tool"}))
			idx = len(m.entries) - 1
		}
		m.entries[idx].detail = event.ToolName
		m.entries[idx].label = toolInvocationLabel(event.ToolName, event.Args)
		m.entries[idx].status = "running"
		m.entries[idx].toolID = event.ToolCallID
		m.entries[idx].args = event.Args
		m.entries[idx].startedAt = event.StartedAt
		m.applyToolOutputModeTo(idx)
		m.markEntryDirty(idx)
	case coreagent.EventToolFinished:
		m.thinking = false
		m.thinkingTokens = 0
		if event.WorkingDir != "" {
			m.opts.WorkingDir = event.WorkingDir
		}
		startedAt := m.toolStartedAt(event.ToolCallID)
		status := "done"
		if event.Error != "" {
			status = "error"
		}
		idx := m.findToolEntry(event.ToolCallID)
		if idx < 0 {
			m.entries = append(m.entries, newChatEntry(chatEntry{role: "tool"}))
			idx = len(m.entries) - 1
		}
		m.entries[idx].content = event.Content
		m.entries[idx].label = toolInvocationLabel(event.ToolName, event.Args)
		m.entries[idx].detail = event.ToolName
		m.entries[idx].status = status
		m.entries[idx].err = event.Error
		m.entries[idx].toolID = event.ToolCallID
		m.entries[idx].args = event.Args
		m.entries[idx].startedAt = startedAt
		m.entries[idx].finishedAt = event.FinishedAt
		m.applyToolOutputModeTo(idx)
		m.markEntryDirty(idx)
	case coreagent.EventToolsUnavailable:
		m.thinking = false
		m.thinkingTokens = 0
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "system", content: "Tools are unavailable for this model."}))
	case coreagent.EventCompacted:
		m.thinking = false
		m.thinkingTokens = 0
		m.status = "compacted"
	case coreagent.EventCompactionSkipped:
		m.thinking = false
		m.thinkingTokens = 0
		message := event.Content
		if strings.TrimSpace(message) == "" {
			message = coreagent.CompactionSkippedMessage(event.Error)
		}
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "system", content: message}))
		m.status = "compact skipped"
	case coreagent.EventError:
		m.thinking = false
		m.thinkingTokens = 0
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "error", content: event.Error, err: event.Error}))
	}
}

func (m chatModel) findToolEntry(toolID string) int {
	if toolID == "" {
		return -1
	}
	for i := len(m.entries) - 1; i >= 0; i-- {
		if m.entries[i].role == "tool" && m.entries[i].toolID == toolID {
			return i
		}
	}
	return -1
}

func (m chatModel) findActiveToolEntry(toolID string) int {
	idx := m.findToolEntry(toolID)
	if idx < 0 || !isToolActiveStatus(m.entries[idx].status) {
		return -1
	}
	return idx
}

func (m chatModel) toolStartedAt(toolID string) time.Time {
	idx := m.findToolEntry(toolID)
	if idx < 0 {
		return time.Time{}
	}
	return m.entries[idx].startedAt
}

func (m *chatModel) toggleAllToolOutputs() {
	toolIndexes := m.toolOutputIndexes()
	if len(toolIndexes) == 0 {
		if !m.running {
			return
		}
		m.setToolOutputMode(!m.toolOutputOpen || !m.toolOutputMode)
		return
	}

	expand := false
	for _, index := range toolIndexes {
		if !m.entries[index].expanded {
			expand = true
			break
		}
	}

	m.setToolOutputMode(expand)
}

func (m chatModel) lastExpandableToolEntry() int {
	for i := len(m.entries) - 1; i >= 0; i-- {
		if m.isExpandableTool(i) {
			return i
		}
	}
	return -1
}

func (m chatModel) expandableToolIndexes() []int {
	var indexes []int
	for i := range m.entries {
		if m.isExpandableTool(i) {
			indexes = append(indexes, i)
		}
	}
	return indexes
}

func (m chatModel) isExpandableTool(index int) bool {
	if index < 0 || index >= len(m.entries) {
		return false
	}
	return entryHasExpandableOutput(m.entries[index])
}

func (m chatModel) toolOutputIndexes() []int {
	var indexes []int
	for i := range m.entries {
		if entryHasToolOutputMode(m.entries[i]) {
			indexes = append(indexes, i)
		}
	}
	return indexes
}

func entryHasExpandableOutput(entry chatEntry) bool {
	return (entry.role == "tool" && isToolResultStatus(entry.status) && entry.content != "") ||
		(entry.role == "tool_group" && len(entry.tools) > 0) ||
		(entry.role == "compaction_summary" && strings.TrimSpace(entry.content) != "")
}

func entryHasToolOutputMode(entry chatEntry) bool {
	return (entry.role == "tool" && (isToolActiveStatus(entry.status) || isToolResultStatus(entry.status) || entry.content != "")) ||
		(entry.role == "tool_group" && len(entry.tools) > 0) ||
		(entry.role == "compaction_summary" && strings.TrimSpace(entry.content) != "")
}

func (m *chatModel) setToolOutputMode(open bool) {
	m.toolOutputMode = true
	m.toolOutputOpen = open
	m.applyToolOutputMode()
}

func (m *chatModel) applyToolOutputMode() {
	if !m.toolOutputMode {
		return
	}
	for i := range m.entries {
		m.applyToolOutputModeTo(i)
	}
}

func (m *chatModel) applyToolOutputModeTo(index int) {
	if !m.toolOutputMode || index < 0 || index >= len(m.entries) {
		return
	}
	if !entryHasToolOutputMode(m.entries[index]) {
		return
	}
	if m.entries[index].expanded == m.toolOutputOpen {
		return
	}
	m.entries[index].expanded = m.toolOutputOpen
	m.markEntryDirty(index)
}

func (m *chatModel) groupCompletedToolHistory() {
	m.entries = groupCompletedToolEntries(m.entries)
	m.applyToolOutputMode()
}

func (m *chatModel) ensureAssistantEntry() int {
	if len(m.entries) > 0 && m.entries[len(m.entries)-1].role == "assistant" {
		return len(m.entries) - 1
	}
	m.entries = append(m.entries, newChatEntry(chatEntry{role: "assistant"}))
	return len(m.entries) - 1
}

func (m chatModel) renderTranscript(width int) string {
	var b strings.Builder
	for index, entry := range m.entries {
		if b.Len() > 0 {
			b.WriteByte('\n')
		}
		prefix, body := m.renderEntry(entry)
		prefixWidth := lipgloss.Width(prefix)
		continuation := ""
		if prefixWidth > 0 {
			continuation = strings.Repeat(" ", prefixWidth)
		}
		for i, line := range m.renderEntryLinesCached(index, entry, body, width-prefixWidth) {
			if i == 0 {
				b.WriteString(prefix)
				b.WriteString(line)
			} else {
				b.WriteString(continuation)
				b.WriteString(line)
			}
			b.WriteByte('\n')
		}
	}
	return b.String()
}

func (m chatModel) renderEntryLinesCached(index int, entry chatEntry, body string, width int) []string {
	key := entryRenderKey(entry, body, width)
	if index >= 0 && index < len(m.entries) {
		cached := m.entries[index]
		if cached.renderKey == key && cached.renderLines != nil {
			return cached.renderLines
		}
	}

	lines := m.renderEntryLines(entry, body, width)
	if index >= 0 && index < len(m.entries) {
		m.entries[index].renderKey = key
		m.entries[index].renderLines = slices.Clone(lines)
	}
	return lines
}

func (m chatModel) transcriptLines(width int) []string {
	transcript := m.renderTranscript(width)
	transcript = strings.TrimRight(transcript, "\n")
	if transcript == "" {
		return nil
	}
	return strings.Split(transcript, "\n")
}

func (m chatModel) visibleTranscriptLines(width, available int) []string {
	if available <= 0 {
		return nil
	}
	lines := m.transcriptLines(width)
	if len(lines) > available {
		maxScroll := len(lines) - available
		scroll := clamp(m.scroll, 0, maxScroll)
		start := maxScroll - scroll
		lines = lines[start : start+available]
	}
	return lines
}

func (m chatModel) transcriptHeight() int {
	width := m.width
	if width <= 0 {
		width = 80
	}
	height := m.height
	if height <= 0 {
		height = 24
	}
	return max(0, height-2-len(m.bottomLines(width)))
}

func (m chatModel) maxScroll() int {
	width := m.width
	if width <= 0 {
		width = 80
	}
	return max(0, len(m.transcriptLines(width))-m.transcriptHeight())
}

func (m chatModel) bottomLines(width int) []string {
	var lines []string
	lines = append(lines, m.completionLines(width)...)
	lines = append(lines, m.queuedLines(width)...)
	if activity := m.activityLine(); activity != "" {
		lines = append(lines, chatMetaStyle.Render(activity))
	}
	lines = append(lines, strings.Split(renderInputBox(string(m.input), width), "\n")...)
	lines = append(lines, m.renderFooterLine())
	return lines
}

func (m *chatModel) scrollBy(lines int) {
	if lines == 0 {
		return
	}
	m.scroll = clamp(m.scroll+lines, 0, m.maxScroll())
}

func (m chatModel) renderEntry(entry chatEntry) (string, string) {
	switch entry.role {
	case "user":
		return "", entry.content
	case "assistant":
		return chatAssistantStyle.Render("●") + " ", entry.content
	case "compaction_summary":
		prefix := toolStatusStyle(entry.status).Render("●") + " "
		return prefix, compactionSummaryStatusLine(entry)
	case "tool":
		prefix := toolStatusStyle(entry.status).Render("⏺") + " "
		return prefix, toolStatusLine(entry)
	case "tool_group":
		prefix := toolStatusStyle(entry.status).Render("●") + " "
		return prefix, toolGroupStatusLine(entry)
	case "error":
		return chatErrorStyle.Render("err ") + " ", entry.content
	case "system":
		return "", entry.content
	case "history":
		return "", entry.content
	default:
		return "", entry.content
	}
}

func (m chatModel) renderEntryLines(entry chatEntry, body string, width int) []string {
	if width < 20 {
		width = 20
	}
	switch entry.role {
	case "assistant", "system":
		return splitRenderedBody(renderMarkdownForView(body, width))
	case "history":
		return renderHistoryLines(body, width)
	case "user":
		return renderUserMessageLines(body, width)
	case "compaction_summary":
		return renderCompactionSummaryLines(entry, width)
	case "tool":
		if entry.status == "approval" {
			return m.renderApprovalEntryLines(entry, body, width)
		}
		if isToolResultStatus(entry.status) {
			return renderToolResultLines(entry, width)
		}
		return wrapChatText(body, width)
	case "tool_group":
		return renderToolGroupLines(entry, width)
	default:
		return wrapChatText(body, width)
	}
}

func renderUserMessageLines(content string, width int) []string {
	return renderPromptRow("> "+content, width)
}

func renderHistoryLines(history string, width int) []string {
	if width < 20 {
		width = 20
	}
	history = strings.TrimRight(history, "\n")
	if history == "" {
		return []string{""}
	}

	var lines []string
	inFence := false
	fence := ""
	fenceIndent := ""

	for _, rawLine := range strings.Split(history, "\n") {
		line := strings.TrimRight(rawLine, "\r")
		if inFence {
			if fenceEnd(line, fence) {
				inFence = false
				fence = ""
				fenceIndent = ""
				continue
			}
			code := strings.TrimPrefix(line, fenceIndent)
			lines = append(lines, renderHistoryCodeLine(fenceIndent, code, width)...)
			continue
		}

		if nextFence, indent, ok := historyFenceStart(line); ok {
			inFence = true
			fence = nextFence
			fenceIndent = indent
			continue
		}

		lines = append(lines, renderHistoryLine(line, width)...)
	}
	if len(lines) == 0 {
		return []string{""}
	}
	return lines
}

func renderHistoryLine(line string, width int) []string {
	if strings.TrimSpace(line) == "" {
		return []string{""}
	}

	if bold, ok := historyBoldLine(line); ok {
		if bold == "Message History" {
			return renderHistoryStyledLine("", bold, width, chatHistoryTitleStyle)
		}
		return renderHistoryStyledLine("", bold, width, historyRoleStyle(bold))
	}

	indent := leadingWhitespace(line)
	text := strings.TrimSpace(line)
	if strings.Contains(text, " · ") {
		return []string{indent + renderHistoryMetaLine(text)}
	}
	if label, value, ok := historyLabelValue(text); ok {
		return renderHistoryLabelValue(indent, label, value, width)
	}
	return renderHistoryStyledLine(indent, text, width, chatHistoryTextStyle)
}

func renderHistoryLabelValue(indent, label, value string, width int) []string {
	labelText := label + ":"
	value = strings.TrimSpace(value)
	if value == "" {
		return []string{indent + chatHistoryLabelStyle.Render(labelText)}
	}

	prefixWidth := lipgloss.Width(indent) + lipgloss.Width(labelText) + 1
	wrapped := wrapChatText(value, max(10, width-prefixWidth))
	lines := make([]string, 0, len(wrapped))
	for i, line := range wrapped {
		if i == 0 {
			lines = append(lines, indent+chatHistoryLabelStyle.Render(labelText)+" "+renderHistoryInline(line, chatHistoryTextStyle))
			continue
		}
		lines = append(lines, indent+strings.Repeat(" ", lipgloss.Width(labelText)+1)+renderHistoryInline(line, chatHistoryTextStyle))
	}
	return lines
}

func renderHistoryStyledLine(indent, text string, width int, style lipgloss.Style) []string {
	wrapped := wrapChatText(text, max(10, width-lipgloss.Width(indent)))
	for i, line := range wrapped {
		wrapped[i] = indent + renderHistoryInline(line, style)
	}
	return wrapped
}

func renderHistoryCodeLine(indent, code string, width int) []string {
	codeIndent := indent + "  "
	wrapped := wrapChatText(code, max(10, width-lipgloss.Width(codeIndent)))
	for i, line := range wrapped {
		wrapped[i] = codeIndent + chatHistoryCodeStyle.Render(line)
	}
	return wrapped
}

func renderHistoryMetaLine(text string) string {
	parts := strings.Split(text, " · ")
	for i, part := range parts {
		if label, value, ok := historyLabelValue(part); ok {
			parts[i] = chatHistoryLabelStyle.Render(label+":") + " " + renderHistoryInline(value, chatHistoryTextStyle)
			continue
		}
		parts[i] = renderHistoryInline(part, chatHistoryTextStyle)
	}
	return strings.Join(parts, chatHistoryLabelStyle.Render(" · "))
}

func renderHistoryInline(text string, style lipgloss.Style) string {
	text = strings.ReplaceAll(text, "**", "")
	var b strings.Builder
	for {
		before, rest, ok := strings.Cut(text, "`")
		b.WriteString(style.Render(before))
		if !ok {
			break
		}
		code, after, ok := strings.Cut(rest, "`")
		if !ok {
			b.WriteString(style.Render("`" + rest))
			break
		}
		b.WriteString(chatHistoryCodeStyle.Render(code))
		text = after
	}
	return b.String()
}

func historyRoleStyle(role string) lipgloss.Style {
	switch role {
	case "system":
		return chatHistorySystemRoleStyle
	case "user":
		return chatHistoryUserRoleStyle
	case "assistant":
		return chatHistoryAssistantRoleStyle
	case "tool":
		return chatHistoryToolRoleStyle
	default:
		return chatHistoryTitleStyle
	}
}

func historyBoldLine(line string) (string, bool) {
	trimmed := strings.TrimSpace(line)
	if !strings.HasPrefix(trimmed, "**") || !strings.HasSuffix(trimmed, "**") || len(trimmed) <= 4 {
		return "", false
	}
	return strings.TrimSpace(strings.TrimSuffix(strings.TrimPrefix(trimmed, "**"), "**")), true
}

func historyLabelValue(text string) (string, string, bool) {
	label, value, ok := strings.Cut(text, ":")
	if !ok {
		return "", "", false
	}
	label = strings.TrimSpace(label)
	if !historyLabel(label) {
		return "", "", false
	}
	return label, strings.TrimSpace(value), true
}

func historyLabel(label string) bool {
	switch label {
	case "args", "content", "thinking", "tool", "tool call", "tool calls":
		return true
	default:
		return false
	}
}

func historyFenceStart(line string) (string, string, bool) {
	indent := leadingWhitespace(line)
	fence, _, ok := markdownFenceStart(line)
	return fence, indent, ok
}

func leadingWhitespace(line string) string {
	for i, r := range line {
		if r != ' ' && r != '\t' {
			return line[:i]
		}
	}
	return line
}

func renderInputBox(input string, width int) string {
	if width < 1 {
		width = 1
	}
	lines := []string{
		chatInputBorderStyle.Render(strings.Repeat("─", width)),
		renderPromptRow("> "+input+"█", width)[0],
		chatInputBorderStyle.Render(strings.Repeat("─", width)),
	}
	return strings.Join(lines, "\n")
}

func renderPromptRow(text string, width int) []string {
	if width < 20 {
		width = 20
	}
	lines := wrapChatText(text, width)
	for i, line := range lines {
		lines[i] = chatUserStyle.Render(line)
	}
	return lines
}

func (m chatModel) slashCommandLines(width int) []string {
	return m.renderCompletions(m.slashCompletions(), width)
}

func (m chatModel) completionLines(width int) []string {
	return m.renderCompletions(m.completions(), width)
}

func (m chatModel) renderCompletions(completions []chatCompletion, width int) []string {
	if len(completions) == 0 {
		return nil
	}
	selected := clamp(m.complete, 0, len(completions)-1)
	nameWidth := 0
	for _, completion := range completions {
		nameWidth = max(nameWidth, lipgloss.Width(completion.label))
	}

	lines := make([]string, 0, len(completions))
	for i, completion := range completions {
		marker := "  "
		if i == selected {
			marker = "› "
		}
		name := chatCommandNameStyle.Render(completion.label)
		padding := strings.Repeat(" ", max(1, nameWidth-lipgloss.Width(completion.label)+2))
		line := marker + name + padding + chatMetaStyle.Render(completion.description)
		lines = append(lines, truncateRenderedLine(line, width))
	}
	return lines
}

func (m chatModel) completions() []chatCompletion {
	if completions := m.slashCompletions(); len(completions) > 0 {
		return completions
	}
	return m.mentionCompletions()
}

func (m chatModel) slashCompletions() []chatCompletion {
	input := strings.TrimSpace(string(m.input))
	if !strings.HasPrefix(input, "/") {
		return nil
	}

	commands := matchingSlashCommands(input)
	skillCompletions := m.skillSlashCompletions(input)
	if len(commands) == 0 && len(skillCompletions) == 0 {
		return []chatCompletion{{label: "No matching commands"}}
	}

	completions := make([]chatCompletion, 0, len(commands)+len(skillCompletions))
	for _, command := range commands {
		completions = append(completions, chatCompletion{
			value:       command.name,
			label:       command.name,
			description: command.description,
		})
	}
	completions = append(completions, skillCompletions...)
	return completions
}

func (m chatModel) queuedLines(width int) []string {
	if len(m.queued) == 0 {
		return nil
	}
	limit := 2
	if width < 40 {
		limit = 1
	}
	lines := make([]string, 0, min(len(m.queued), limit)+1)
	for i, queued := range m.queued {
		if i >= limit {
			lines = append(lines, chatMetaStyle.Render(fmt.Sprintf("queued +%d more", len(m.queued)-i)))
			break
		}
		label := fmt.Sprintf("queued %d: %s", i+1, queued)
		lines = append(lines, chatMetaStyle.Render(truncateRunes(label, max(20, width))))
	}
	return lines
}

func (m chatModel) renderResumePicker(width int) string {
	picker := m.resumePicker
	if picker == nil {
		return ""
	}
	if width <= 0 {
		width = 80
	}

	var b strings.Builder
	b.WriteString(chatResumeTitleStyle.Render("Resume session"))
	b.WriteString("\n\n")
	b.WriteString(renderResumeSearchBox(picker.filter, width))
	b.WriteString("\n\n")

	filtered := picker.filtered()
	if len(filtered) == 0 {
		b.WriteString(chatResumeMetaStyle.Render("No matching chats"))
		b.WriteString("\n")
	} else {
		start := clamp(picker.scroll, 0, max(0, len(filtered)-1))
		end := min(len(filtered), start+maxResumePickerItems)
		for i := start; i < end; i++ {
			chat := filtered[i]
			selected := i == picker.cursor
			if selected {
				b.WriteString(chatResumeSelectedStyle.Render("› " + resumeChatTitle(chat)))
			} else {
				b.WriteString("  ")
				b.WriteString(chatResumeTextStyle.Render(resumeChatTitle(chat)))
			}
			b.WriteByte('\n')
			b.WriteString(chatResumeMetaStyle.Render("  " + resumeChatMeta(chat)))
			b.WriteByte('\n')
			if i < end-1 {
				b.WriteByte('\n')
			}
		}
		if end < len(filtered) {
			b.WriteString(chatResumeMetaStyle.Render(fmt.Sprintf("\n  +%d more", len(filtered)-end)))
			b.WriteByte('\n')
		}
	}

	b.WriteString("\n")
	b.WriteString(chatResumeMetaStyle.Render("↑/↓ move • enter resume • type search • esc cancel"))
	return b.String()
}

func (m chatModel) renderHistoryPopup(width, height int) string {
	popup := m.historyPopup
	if popup == nil {
		return ""
	}
	if width <= 0 {
		width = 80
	}
	if height <= 0 {
		height = 24
	}

	bodyLines := m.historyPopupBodyLinesForWidth(width)
	visibleHeight := max(1, height-4)
	maxScroll := max(0, len(bodyLines)-visibleHeight)
	scroll := clamp(popup.scroll, 0, maxScroll)
	if popup.stickToBottom {
		scroll = maxScroll
	}
	end := min(len(bodyLines), scroll+visibleHeight)

	var b strings.Builder
	b.WriteString(chatResumeTitleStyle.Render("Message history"))
	if len(bodyLines) > visibleHeight {
		b.WriteString(chatResumeMetaStyle.Render(fmt.Sprintf("  %d/%d", scroll+1, maxScroll+1)))
	}
	b.WriteString("\n\n")
	if len(bodyLines) == 0 {
		b.WriteString(chatResumeMetaStyle.Render("No messages yet."))
		b.WriteByte('\n')
	} else {
		for _, line := range bodyLines[scroll:end] {
			b.WriteString(line)
			b.WriteByte('\n')
		}
	}
	b.WriteString("\n")
	help := "↑/↓ scroll • pgup/pgdn page • home/end jump • esc close"
	b.WriteString(chatResumeMetaStyle.Render(truncateRenderedLine(help, width)))
	return b.String()
}

func (m chatModel) renderApprovalEntryLines(entry chatEntry, body string, width int) []string {
	prompt := m.approvalPrompt
	if prompt == nil || prompt.request.ToolCallID != entry.toolID {
		return wrapChatText(body, width)
	}
	if width <= 0 {
		width = 80
	}
	bodyWidth := max(20, width-2)

	request := prompt.request
	var lines []string
	lines = append(lines, wrapChatText(body, width)...)
	if request.Summary != "" {
		lines = append(lines, indentLines(wrapChatText(request.Summary, bodyWidth), "  ")...)
	} else {
		lines = append(lines, indentLines(wrapChatText(fmt.Sprintf("%s wants to run", toolDisplayName(request.ToolName)), bodyWidth), "  ")...)
	}
	if detail := approvalRequestDetail(request, bodyWidth); detail != "" {
		lines = append(lines, indentLines(splitRenderedBody(detail), "  ")...)
	}

	risk := request.Risk
	if risk == "" {
		risk = coreagent.ApprovalRiskMedium
	}
	lines = append(lines, "  "+approvalRiskStyle(risk).Render("Risk: "+string(risk)))
	for _, reason := range request.Reasons {
		lines = append(lines, "  "+chatMetaStyle.Render("- "+reason))
	}
	if strings.TrimSpace(request.WorkingDir) != "" {
		lines = append(lines, "  "+chatMetaStyle.Render("cwd: "+request.WorkingDir))
	}

	lines = append(lines, "")
	lines = append(lines, indentLines([]string{renderApprovalChoices(prompt.cursor, bodyWidth)}, "  ")...)
	lines = append(lines, "  "+chatMetaStyle.Render("enter select • ←/→ move • o once • s session • d deny • esc deny"))
	return lines
}

func approvalRequestDetail(request coreagent.ApprovalRequest, width int) string {
	switch request.ToolName {
	case "bash":
		command, ok := stringArg(request.Args, "command")
		if !ok {
			return ""
		}
		return strings.Join(wrapChatText("$ "+command, width), "\n")
	case "edit":
		path, ok := stringArg(request.Args, "path")
		if !ok {
			return ""
		}
		var lines []string
		lines = append(lines, "path: "+path)
		if oldText, ok := stringArg(request.Args, "old_text"); ok {
			lines = append(lines, fmt.Sprintf("old_text: %d chars", len([]rune(oldText))))
		}
		if newText, ok := stringArg(request.Args, "new_text"); ok {
			lines = append(lines, fmt.Sprintf("new_text: %d chars", len([]rune(newText))))
		}
		return chatMetaStyle.Render(strings.Join(lines, "\n"))
	default:
		if len(request.Args) == 0 {
			return ""
		}
		return chatMetaStyle.Render(formatToolArgs(request.Args))
	}
}

func renderApprovalChoices(cursor, width int) string {
	var parts []string
	for i, choice := range chatApprovalChoices {
		label := choice.label
		if choice.key != "" {
			label = choice.label + " (" + choice.key + ")"
		}
		if i == clamp(cursor, 0, len(chatApprovalChoices)-1) {
			parts = append(parts, chatResumeSelectedStyle.Render("› "+label))
		} else {
			parts = append(parts, chatResumeTextStyle.Render("  "+label))
		}
	}
	return truncateRenderedLine(strings.Join(parts, "   "), width)
}

func approvalRiskStyle(risk coreagent.ApprovalRisk) lipgloss.Style {
	switch risk {
	case coreagent.ApprovalRiskHigh:
		return chatErrorStyle
	case coreagent.ApprovalRiskMedium:
		return chatToolRunningStyle
	default:
		return chatMetaStyle
	}
}

func renderResumeSearchBox(filter string, width int) string {
	if width < 20 {
		width = 20
	}
	placeholder := "Search..."
	value := filter
	if value == "" {
		value = placeholder
	}
	line := "⌕ " + value
	lines := []string{
		chatResumeBorderStyle.Render(strings.Repeat("─", width)),
		chatResumeTextStyle.Render(truncateRunes(line, max(20, width))),
		chatResumeBorderStyle.Render(strings.Repeat("─", width)),
	}
	return strings.Join(lines, "\n")
}

func (p *chatResumePicker) filtered() []chatstore.ChatSummary {
	if p == nil {
		return nil
	}
	filter := strings.ToLower(strings.TrimSpace(p.filter))
	if filter == "" {
		return p.chats
	}
	var out []chatstore.ChatSummary
	for _, chat := range p.chats {
		haystack := strings.ToLower(strings.Join([]string{
			chat.Title,
			chat.Model,
			chat.ID,
		}, " "))
		if strings.Contains(haystack, filter) {
			out = append(out, chat)
		}
	}
	return out
}

func (p *chatResumePicker) move(delta int) {
	if p == nil {
		return
	}
	filtered := p.filtered()
	if len(filtered) == 0 {
		p.cursor = 0
		p.scroll = 0
		return
	}
	p.cursor = clamp(p.cursor+delta, 0, len(filtered)-1)
	p.updateScroll()
}

func (p *chatResumePicker) updateScroll() {
	if p == nil {
		return
	}
	if p.cursor < p.scroll {
		p.scroll = p.cursor
	}
	if p.cursor >= p.scroll+maxResumePickerItems {
		p.scroll = p.cursor - maxResumePickerItems + 1
	}
	if p.scroll < 0 {
		p.scroll = 0
	}
}

func (p *chatResumePicker) selected() (chatstore.ChatSummary, bool) {
	if p == nil {
		return chatstore.ChatSummary{}, false
	}
	filtered := p.filtered()
	if len(filtered) == 0 || p.cursor < 0 || p.cursor >= len(filtered) {
		return chatstore.ChatSummary{}, false
	}
	return filtered[p.cursor], true
}

func resumeChatTitle(chat chatstore.ChatSummary) string {
	title := strings.TrimSpace(chat.Title)
	if title == "" {
		title = shortChatID(chat.ID)
	}
	return truncateRunes(title, 96)
}

func resumeChatMeta(chat chatstore.ChatSummary) string {
	var parts []string
	if !chat.UpdatedAt.IsZero() {
		parts = append(parts, relativeTime(chat.UpdatedAt))
	}
	if chat.Model != "" {
		parts = append(parts, chat.Model)
	}
	if chat.MessageCount > 0 {
		parts = append(parts, fmt.Sprintf("%d messages", chat.MessageCount))
	}
	if chat.ApproxBytes > 0 {
		parts = append(parts, formatByteSize(chat.ApproxBytes))
	}
	return strings.Join(parts, " · ")
}

func relativeTime(t time.Time) string {
	if t.IsZero() {
		return ""
	}
	elapsed := time.Since(t)
	if elapsed < 0 {
		elapsed = 0
	}
	switch {
	case elapsed < time.Minute:
		return "just now"
	case elapsed < time.Hour:
		return fmt.Sprintf("%d min ago", int(elapsed/time.Minute))
	case elapsed < 24*time.Hour:
		hours := int(elapsed / time.Hour)
		if hours == 1 {
			return "1 hour ago"
		}
		return fmt.Sprintf("%d hours ago", hours)
	case elapsed < 14*24*time.Hour:
		days := int(elapsed / (24 * time.Hour))
		if days == 1 {
			return "1 day ago"
		}
		return fmt.Sprintf("%d days ago", days)
	default:
		return t.Format("Jan 2")
	}
}

func formatByteSize(bytes int64) string {
	if bytes < 1024 {
		return fmt.Sprintf("%dB", bytes)
	}
	kb := float64(bytes) / 1024
	if kb < 1024 {
		return fmt.Sprintf("%.1fKB", kb)
	}
	return fmt.Sprintf("%.1fMB", kb/1024)
}

func shortChatID(id string) string {
	if len(id) <= 8 {
		return id
	}
	return id[:8]
}

func matchingSlashCommands(input string) []chatSlashCommand {
	prefix := strings.ToLower(strings.TrimSpace(input))
	if prefix == "" {
		return nil
	}

	var commands []chatSlashCommand
	for _, command := range chatSlashCommands {
		if strings.HasPrefix(command.name, prefix) {
			commands = append(commands, command)
		}
	}
	return commands
}

func (m chatModel) mentionCompletions() []chatCompletion {
	input := string(m.input)
	_, query, ok := activeMentionToken(input)
	if !ok {
		return nil
	}

	workingDir := m.opts.WorkingDir
	if strings.TrimSpace(workingDir) == "" {
		var err error
		workingDir, err = os.Getwd()
		if err != nil {
			return []chatCompletion{{label: "No working directory"}}
		}
	}

	dirPart, prefix := splitMentionQuery(query)
	dir, err := resolveCompletionDir(workingDir, dirPart)
	if err != nil {
		return []chatCompletion{{label: "No matching files"}}
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return []chatCompletion{{label: "No matching files"}}
	}
	sort.SliceStable(entries, func(i, j int) bool {
		if entries[i].IsDir() != entries[j].IsDir() {
			return entries[i].IsDir()
		}
		return strings.ToLower(entries[i].Name()) < strings.ToLower(entries[j].Name())
	})

	includeHidden := strings.HasPrefix(prefix, ".")
	completions := make([]chatCompletion, 0, 8)
	for _, entry := range entries {
		name := entry.Name()
		if !includeHidden && strings.HasPrefix(name, ".") {
			continue
		}
		if !strings.HasPrefix(strings.ToLower(name), strings.ToLower(prefix)) {
			continue
		}
		value := filepath.ToSlash(filepath.Join(dirPart, name))
		label := "@" + value
		description := "file"
		if entry.IsDir() {
			value += "/"
			label += "/"
			description = "directory"
		}
		completions = append(completions, chatCompletion{
			value:       value,
			label:       label,
			description: description,
			directory:   entry.IsDir(),
		})
		if len(completions) >= 8 {
			break
		}
	}
	if len(completions) == 0 {
		return []chatCompletion{{label: "No matching files"}}
	}
	return completions
}

func activeMentionToken(input string) (int, string, bool) {
	runes := []rune(input)
	start := len(runes)
	for start > 0 && !unicode.IsSpace(runes[start-1]) {
		start--
	}
	token := string(runes[start:])
	if !strings.HasPrefix(token, "@") {
		return 0, "", false
	}
	return start, token[1:], true
}

func splitMentionQuery(query string) (string, string) {
	query = filepath.ToSlash(query)
	index := strings.LastIndex(query, "/")
	if index < 0 {
		return ".", query
	}
	return query[:index+1], query[index+1:]
}

func resolveCompletionDir(workingDir, dir string) (string, error) {
	if filepath.IsAbs(dir) {
		return "", fmt.Errorf("absolute paths are not allowed")
	}
	base := workingDir
	if base == "" {
		base = "."
	}
	base, err := filepath.Abs(base)
	if err != nil {
		return "", err
	}
	resolved := filepath.Clean(filepath.Join(base, dir))
	rel, err := filepath.Rel(base, resolved)
	if err != nil {
		return "", err
	}
	if rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("path escapes working directory")
	}
	return resolved, nil
}

func (m *chatModel) moveCompletion(delta int) bool {
	completions := m.completions()
	if len(completions) == 0 || !completionIsSelectable(completions) {
		return false
	}
	m.complete = (m.complete + delta) % len(completions)
	if m.complete < 0 {
		m.complete += len(completions)
	}
	return true
}

func (m *chatModel) applyCompletion() bool {
	completions := m.completions()
	if len(completions) == 0 || !completionIsSelectable(completions) {
		return false
	}
	m.resetPromptHistoryCursor()
	selected := completions[clamp(m.complete, 0, len(completions)-1)]
	input := string(m.input)
	if strings.HasPrefix(strings.TrimSpace(input), "/") {
		m.input = []rune(selected.value)
		m.complete = 0
		return true
	}

	start, _, ok := activeMentionToken(input)
	if !ok {
		return false
	}
	suffix := ""
	if !selected.directory {
		suffix = " "
	}
	next := string([]rune(input)[:start]) + "@" + selected.value + suffix
	m.input = []rune(next)
	m.complete = 0
	return true
}

func completionIsSelectable(completions []chatCompletion) bool {
	return len(completions) > 0 && completions[0].value != ""
}

func renderMarkdownForView(markdown string, width int) string {
	if strings.TrimSpace(markdown) == "" {
		return markdown
	}
	if width < 20 {
		width = 20
	}
	return renderMarkdownBlocks(markdownBlocks(markdown), width)
}

func renderMarkdownPlain(markdown string, width int) string {
	return renderMarkdownBlocks(markdownBlocks(markdown), width)
}

func exposeMarkdownLinks(markdown string) string {
	return markdownLinkPattern.ReplaceAllString(markdown, "$1 ($2)")
}

func renderMarkdownChunk(markdown string, width int) string {
	renderer, err := markdownRendererForWidth(width)
	if err != nil {
		return strings.Join(wrapChatText(markdown, width), "\n")
	}
	renderer.mu.Lock()
	rendered, err := renderer.renderer.Render(markdown)
	renderer.mu.Unlock()
	if err != nil {
		return strings.Join(wrapChatText(markdown, width), "\n")
	}
	return trimRenderedLines(rendered)
}

type markdownBlockKind int

const (
	markdownBlockProse markdownBlockKind = iota
	markdownBlockTable
	markdownBlockCodeFence
	markdownBlockDiffFence
)

type markdownBlock struct {
	kind  markdownBlockKind
	text  string
	table markdownTable
}

type markdownTable struct {
	headers []string
	rows    [][]string
}

func markdownBlocks(markdown string) []markdownBlock {
	lines := strings.Split(markdown, "\n")
	blocks := make([]markdownBlock, 0, 1)
	var prose []string

	flushProse := func() {
		if len(prose) == 0 {
			return
		}
		blocks = append(blocks, markdownBlock{kind: markdownBlockProse, text: strings.Join(prose, "\n")})
		prose = nil
	}

	for i := 0; i < len(lines); {
		if fence, info, ok := markdownFenceStart(lines[i]); ok {
			flushProse()
			if markdownFenceIsDiff(info) {
				var diffLines []string
				i++
				for ; i < len(lines); i++ {
					if fenceEnd(lines[i], fence) {
						break
					}
					diffLines = append(diffLines, lines[i])
				}
				if i < len(lines) {
					i++
				}
				blocks = append(blocks, markdownBlock{kind: markdownBlockDiffFence, text: strings.Join(diffLines, "\n")})
				continue
			}

			codeLines := []string{lines[i]}
			i++
			for ; i < len(lines); i++ {
				codeLines = append(codeLines, lines[i])
				if fenceEnd(lines[i], fence) {
					i++
					break
				}
			}
			blocks = append(blocks, markdownBlock{kind: markdownBlockCodeFence, text: strings.Join(codeLines, "\n")})
			continue
		}

		if table, next, ok := markdownTableAt(lines, i); ok {
			flushProse()
			blocks = append(blocks, markdownBlock{kind: markdownBlockTable, table: table})
			i = next
			continue
		}

		prose = append(prose, lines[i])
		i++
	}
	flushProse()
	return blocks
}

func renderMarkdownBlocks(blocks []markdownBlock, width int) string {
	if width < 20 {
		width = 20
	}
	var rendered []string
	for _, block := range blocks {
		switch block.kind {
		case markdownBlockProse:
			chunk := renderMarkdownChunk(exposeMarkdownLinks(block.text), width)
			if strings.TrimSpace(chunk) != "" {
				rendered = append(rendered, chunk)
			}
		case markdownBlockTable:
			rendered = append(rendered, renderMarkdownTable(block.table, width))
		case markdownBlockCodeFence:
			chunk := renderMarkdownChunk(block.text, width)
			if strings.TrimSpace(chunk) != "" {
				rendered = append(rendered, chunk)
			}
		case markdownBlockDiffFence:
			rendered = append(rendered, renderDiffForView(block.text, width))
		}
	}
	return trimRenderedLines(strings.Join(rendered, "\n"))
}

func renderMarkdownTables(markdown string, width int) (string, bool) {
	blocks := markdownBlocks(markdown)
	found := false
	for _, block := range blocks {
		if block.kind == markdownBlockTable {
			found = true
			break
		}
	}
	if !found {
		return "", false
	}
	return renderMarkdownBlocks(blocks, width), true
}

func markdownTableAt(lines []string, start int) (markdownTable, int, bool) {
	if start+1 >= len(lines) {
		return markdownTable{}, start, false
	}
	if markdownLineIsIndentedCode(lines[start]) || markdownLineIsIndentedCode(lines[start+1]) {
		return markdownTable{}, start, false
	}
	headers, ok := parseMarkdownTableRow(lines[start])
	if !ok {
		return markdownTable{}, start, false
	}
	separator, ok := parseMarkdownTableRow(lines[start+1])
	if !ok || !isMarkdownTableSeparator(separator) {
		return markdownTable{}, start, false
	}

	table := markdownTable{headers: headers}
	columns := len(headers)
	i := start + 2
	for ; i < len(lines); i++ {
		if strings.TrimSpace(lines[i]) == "" {
			break
		}
		row, ok := parseMarkdownTableRow(lines[i])
		if !ok || isMarkdownTableSeparator(row) {
			break
		}
		if len(row) > columns {
			columns = len(row)
		}
		table.rows = append(table.rows, row)
	}
	table.headers = padMarkdownTableRow(table.headers, columns)
	for row := range table.rows {
		table.rows[row] = padMarkdownTableRow(table.rows[row], columns)
	}
	return table, i, true
}

func parseMarkdownTableRow(line string) ([]string, bool) {
	trimmed := strings.TrimSpace(line)
	if !strings.Contains(trimmed, "|") {
		return nil, false
	}
	if strings.HasPrefix(trimmed, "|") {
		trimmed = strings.TrimPrefix(trimmed, "|")
	}
	if strings.HasSuffix(trimmed, "|") {
		trimmed = strings.TrimSuffix(trimmed, "|")
	}

	var cells []string
	var b strings.Builder
	escaped := false
	for _, r := range trimmed {
		switch {
		case escaped:
			if r != '|' {
				b.WriteRune('\\')
			}
			b.WriteRune(r)
			escaped = false
		case r == '\\':
			escaped = true
		case r == '|':
			cells = append(cells, strings.TrimSpace(b.String()))
			b.Reset()
		default:
			b.WriteRune(r)
		}
	}
	if escaped {
		b.WriteRune('\\')
	}
	cells = append(cells, strings.TrimSpace(b.String()))
	if len(cells) < 2 {
		return nil, false
	}
	return cells, true
}

func isMarkdownTableSeparator(cells []string) bool {
	if len(cells) < 2 {
		return false
	}
	for _, cell := range cells {
		normalized := strings.ReplaceAll(strings.TrimSpace(cell), " ", "")
		if !markdownTableSeparatorPattern.MatchString(normalized) {
			return false
		}
	}
	return true
}

func padMarkdownTableRow(row []string, columns int) []string {
	if len(row) >= columns {
		return row
	}
	out := slices.Clone(row)
	for len(out) < columns {
		out = append(out, "")
	}
	return out
}

func renderMarkdownTable(table markdownTable, width int) string {
	if width < 20 {
		width = 20
	}
	rows := make([][]string, 0, len(table.rows)+1)
	rows = append(rows, table.headers)
	rows = append(rows, table.rows...)

	widths := markdownTableColumnWidths(rows, width)
	var rendered []string
	rendered = append(rendered, renderMarkdownTableRow(table.headers, widths, true)...)
	rendered = append(rendered, strings.Repeat("─", min(width, markdownTableRenderedWidth(widths))))
	for _, row := range table.rows {
		rendered = append(rendered, renderMarkdownTableRow(row, widths, false)...)
	}
	return strings.Join(rendered, "\n")
}

func markdownTableColumnWidths(rows [][]string, width int) []int {
	columns := 0
	for _, row := range rows {
		columns = max(columns, len(row))
	}
	if columns == 0 {
		return nil
	}
	gapWidth := 2 * (columns - 1)
	available := max(columns, width-gapWidth)
	minWidth := 4
	if available < columns*minWidth {
		minWidth = max(1, available/columns)
	}

	maxWidths := make([]int, columns)
	for _, row := range rows {
		for col := 0; col < columns; col++ {
			cell := ""
			if col < len(row) {
				cell = cleanMarkdownTableCell(row[col])
			}
			for _, line := range strings.Split(cell, "\n") {
				maxWidths[col] = max(maxWidths[col], lipgloss.Width(line))
			}
		}
	}

	widths := make([]int, columns)
	even := max(minWidth, available/columns)
	for col := range widths {
		widths[col] = min(max(maxWidths[col], minWidth), even)
	}

	for leftover := available - sumInts(widths); leftover > 0; {
		grew := false
		for col := columns - 1; col >= 0 && leftover > 0; col-- {
			if widths[col] >= maxWidths[col] {
				continue
			}
			widths[col]++
			leftover--
			grew = true
		}
		if !grew {
			widths[columns-1] += leftover
			break
		}
	}

	for sumInts(widths) > available {
		col := widestColumn(widths)
		if widths[col] <= minWidth {
			break
		}
		widths[col]--
	}
	return widths
}

func renderMarkdownTableRow(row []string, widths []int, header bool) []string {
	wrapped := make([][]string, len(widths))
	height := 1
	for col := range widths {
		cell := ""
		if col < len(row) {
			cell = cleanMarkdownTableCell(row[col])
		}
		wrapped[col] = wrapTableCell(cell, widths[col])
		height = max(height, len(wrapped[col]))
	}

	lines := make([]string, 0, height)
	for lineIndex := 0; lineIndex < height; lineIndex++ {
		var b strings.Builder
		for col := range widths {
			if col > 0 {
				b.WriteString("  ")
			}
			part := ""
			if lineIndex < len(wrapped[col]) {
				part = wrapped[col][lineIndex]
			}
			if header || (col == 0 && strings.TrimSpace(part) != "") {
				part = chatHeaderStyle.Render(part)
			}
			if col < len(widths)-1 {
				part = padRenderedCell(part, widths[col])
			}
			b.WriteString(part)
		}
		lines = append(lines, strings.TrimRight(b.String(), " "))
	}
	return lines
}

func cleanMarkdownTableCell(cell string) string {
	cell = strings.TrimSpace(cell)
	cell = strings.ReplaceAll(cell, "<br>", "\n")
	cell = strings.ReplaceAll(cell, "<br/>", "\n")
	cell = strings.ReplaceAll(cell, "<br />", "\n")
	cell = exposeMarkdownLinks(cell)
	cell = strings.ReplaceAll(cell, "**", "")
	cell = strings.ReplaceAll(cell, "__", "")
	cell = strings.Trim(cell, "`")
	return strings.ReplaceAll(cell, `\|`, "|")
}

func wrapTableCell(text string, width int) []string {
	if width <= 0 {
		return []string{""}
	}
	var out []string
	for _, rawLine := range strings.Split(text, "\n") {
		line := strings.TrimRight(rawLine, "\r")
		for lipgloss.Width(line) > width {
			runes := []rune(line)
			cut := min(width, len(runes))
			for i := cut; i > max(1, cut/2); i-- {
				if unicode.IsSpace(runes[i-1]) {
					cut = i
					break
				}
			}
			out = append(out, strings.TrimSpace(string(runes[:cut])))
			line = strings.TrimSpace(string(runes[cut:]))
		}
		out = append(out, line)
	}
	if len(out) == 0 {
		return []string{""}
	}
	return out
}

func markdownTableRenderedWidth(widths []int) int {
	if len(widths) == 0 {
		return 0
	}
	return sumInts(widths) + 2*(len(widths)-1)
}

func padRenderedCell(cell string, width int) string {
	padding := width - lipgloss.Width(cell)
	if padding <= 0 {
		return cell
	}
	return cell + strings.Repeat(" ", padding)
}

func sumInts(values []int) int {
	total := 0
	for _, value := range values {
		total += value
	}
	return total
}

func widestColumn(widths []int) int {
	widest := 0
	for i := 1; i < len(widths); i++ {
		if widths[i] > widths[widest] {
			widest = i
		}
	}
	return widest
}

func markdownRendererForWidth(width int) (*cachedMarkdownRenderer, error) {
	if cached, ok := chatMarkdownRenderers.Load(width); ok {
		return cached.(*cachedMarkdownRenderer), nil
	}

	renderer, err := glamour.NewTermRenderer(
		glamour.WithStyles(compactMarkdownStyle()),
		glamour.WithWordWrap(width),
		glamour.WithTableWrap(true),
		glamour.WithInlineTableLinks(true),
	)
	if err != nil {
		return nil, err
	}
	cached := &cachedMarkdownRenderer{renderer: renderer}
	actual, _ := chatMarkdownRenderers.LoadOrStore(width, cached)
	return actual.(*cachedMarkdownRenderer), nil
}

func renderMarkdownDiffFences(markdown string, width int) (string, bool) {
	blocks := markdownBlocks(markdown)
	found := false
	for _, block := range blocks {
		if block.kind == markdownBlockDiffFence {
			found = true
			break
		}
	}
	if !found {
		return "", false
	}
	return renderMarkdownBlocks(blocks, width), true
}

func diffFenceStart(line string) (string, bool) {
	fence, info, ok := markdownFenceStart(line)
	if !ok || !markdownFenceIsDiff(info) {
		return "", false
	}
	return fence, true
}

func markdownFenceStart(line string) (string, string, bool) {
	trimmed := strings.TrimSpace(line)
	if strings.HasPrefix(trimmed, "```") {
		return markdownFence(trimmed, '`')
	}
	if strings.HasPrefix(trimmed, "~~~") {
		return markdownFence(trimmed, '~')
	}
	return "", "", false
}

func markdownFence(line string, marker rune) (string, string, bool) {
	count := 0
	for _, r := range line {
		if r != marker {
			break
		}
		count++
	}
	if count < 3 {
		return "", "", false
	}
	info := strings.TrimSpace(line[count:])
	return strings.Repeat(string(marker), count), info, true
}

func markdownFenceIsDiff(info string) bool {
	fields := strings.Fields(strings.ToLower(strings.TrimSpace(info)))
	if len(fields) == 0 {
		return false
	}
	first := strings.Trim(fields[0], "{}.")
	return first == "diff" || first == "patch"
}

func fenceEnd(line string, fence string) bool {
	trimmed := strings.TrimSpace(line)
	return strings.HasPrefix(trimmed, fence)
}

func markdownLineIsIndentedCode(line string) bool {
	if strings.HasPrefix(line, "\t") {
		return true
	}
	spaces := 0
	for _, r := range line {
		if r != ' ' {
			break
		}
		spaces++
	}
	return spaces >= 4
}

func compactMarkdownStyle() glamouransi.StyleConfig {
	style := styles.ASCIIStyleConfig
	style.Document.Margin = uintPtr(0)
	style.Document.BlockPrefix = ""
	style.Document.BlockSuffix = ""
	style.CodeBlock.Margin = uintPtr(0)
	style.Table.Margin = uintPtr(0)
	style.Strong.BlockPrefix = ""
	style.Strong.BlockSuffix = ""
	style.Strong.Bold = boolPtr(true)
	style.Emph.BlockPrefix = ""
	style.Emph.BlockSuffix = ""
	style.Emph.Italic = boolPtr(true)
	return style
}

func trimRenderedLines(rendered string) string {
	rendered = strings.TrimRight(rendered, "\n")
	lines := strings.Split(rendered, "\n")
	for i, line := range lines {
		lines[i] = strings.TrimRight(line, " \t")
	}
	return strings.Join(lines, "\n")
}

func uintPtr(value uint) *uint {
	return &value
}

func boolPtr(value bool) *bool {
	return &value
}

func splitRenderedBody(body string) []string {
	body = strings.TrimRight(body, "\n")
	if body == "" {
		return []string{""}
	}
	return strings.Split(body, "\n")
}

func renderToolResultLines(entry chatEntry, width int) []string {
	lines := wrapChatText(toolStatusLine(entry), width)
	if !entry.expanded {
		return lines
	}

	if strings.TrimSpace(entry.content) == "" {
		return lines
	}
	lines = append(lines, "")
	lines = append(lines, renderToolOutputLines(entry, entry.content, width)...)
	return lines
}

func renderToolGroupLines(entry chatEntry, width int) []string {
	lines := wrapChatText(toolGroupStatusLine(entry), width)
	if !entry.expanded {
		return lines
	}

	for i, tool := range entry.tools {
		if i > 0 {
			lines = append(lines, "")
		}
		lines = append(lines, "  "+toolGroupChildStatusLine(tool))
		if strings.TrimSpace(tool.content) == "" {
			continue
		}
		lines = append(lines, indentLines(renderToolOutputLines(tool, tool.content, width-4), "    ")...)
	}
	return lines
}

func renderCompactionSummaryLines(entry chatEntry, width int) []string {
	lines := wrapChatText(compactionSummaryStatusLine(entry), width)
	if !entry.expanded || strings.TrimSpace(entry.content) == "" {
		return lines
	}
	lines = append(lines, "")
	lines = append(lines, indentLines(splitRenderedBody(renderMarkdownForView(entry.content, width-2)), "  ")...)
	return lines
}

func compactionSummaryStatusLine(entry chatEntry) string {
	status := toolStatusStyle(entry.status).Render(toolStatusLabel(entry))
	if entry.expanded {
		return fmt.Sprintf("▾ Compacted summary %s", status)
	}
	return fmt.Sprintf("▸ Compacted summary %s", status)
}

func toolGroupChildStatusLine(entry chatEntry) string {
	label := entry.label
	if label == "" {
		label = toolDisplayName(entry.detail)
	}

	status := toolStatusLabel(entry)
	if suffix := toolElapsedSuffix(entry.startedAt, entry.finishedAt); suffix != "" && isToolResultStatus(entry.status) {
		status += suffix
	}

	return fmt.Sprintf("%s %s", boldToolInvocationName(label), toolStatusStyle(entry.status).Render(status))
}

func boldToolInvocationName(label string) string {
	name, rest, ok := strings.Cut(label, "(")
	if !ok || name == "" {
		return chatHeaderStyle.Render(label)
	}
	return chatHeaderStyle.Render(name) + "(" + rest
}

func renderToolOutputLines(entry chatEntry, output string, width int) []string {
	if looksLikeUnifiedDiff(output) {
		return splitRenderedBody(renderDiffForView(output, width))
	}
	if toolOutputUsesMarkdown(entry.detail) {
		return splitRenderedBody(renderMarkdownForView(output, width))
	}
	return wrapChatText(output, width)
}

func looksLikeUnifiedDiff(output string) bool {
	lines := strings.Split(output, "\n")
	hasOldFile := false
	hasNewFile := false
	hasHunk := false
	for _, line := range lines {
		switch {
		case strings.HasPrefix(line, "diff --git "):
			return true
		case strings.HasPrefix(line, "--- "):
			hasOldFile = true
		case strings.HasPrefix(line, "+++ "):
			hasNewFile = true
		case strings.HasPrefix(line, "@@ "):
			hasHunk = true
		}
	}
	return hasHunk || (hasOldFile && hasNewFile)
}

func renderDiffForView(diff string, width int) string {
	if width < 20 {
		width = 20
	}
	lines := strings.Split(strings.TrimRight(diff, "\n"), "\n")
	rendered := make([]string, 0, len(lines))
	for _, line := range lines {
		style := diffLineStyle(line)
		for _, wrapped := range wrapChatText(line, width) {
			rendered = append(rendered, style.Render(wrapped))
		}
	}
	return strings.Join(rendered, "\n")
}

func diffLineStyle(line string) lipgloss.Style {
	switch {
	case strings.HasPrefix(line, "diff --git "),
		strings.HasPrefix(line, "--- "),
		strings.HasPrefix(line, "+++ "):
		return chatDiffFileStyle
	case strings.HasPrefix(line, "@@ "):
		return chatDiffHunkStyle
	case strings.HasPrefix(line, "+"):
		return chatDiffAddStyle
	case strings.HasPrefix(line, "-"):
		return chatDiffDeleteStyle
	case strings.HasPrefix(line, "index "),
		strings.HasPrefix(line, "new file "),
		strings.HasPrefix(line, "deleted file "),
		strings.HasPrefix(line, "similarity index "),
		strings.HasPrefix(line, "rename from "),
		strings.HasPrefix(line, "rename to "),
		strings.HasPrefix(line, "\\ "):
		return chatDiffMetaStyle
	default:
		return chatToolStyle
	}
}

func isToolActiveStatus(status string) bool {
	return status == "queued" || status == "running" || status == "approval"
}

func isToolResultStatus(status string) bool {
	return status == "done" || status == "error"
}

func toolStatusLine(entry chatEntry) string {
	return toolStatusLineWithArrow(entry, true)
}

func toolStatusLineWithArrow(entry chatEntry, arrow bool) string {
	label := entry.label
	if label == "" {
		label = toolDisplayName(entry.detail)
	}

	status := toolStatusLabel(entry)
	if suffix := toolElapsedSuffix(entry.startedAt, entry.finishedAt); suffix != "" && isToolResultStatus(entry.status) {
		status += suffix
	}

	if arrow && isToolResultStatus(entry.status) && entry.content != "" {
		if entry.expanded {
			return fmt.Sprintf("▾ %s %s", label, toolStatusStyle(entry.status).Render(status))
		}
		return fmt.Sprintf("▸ %s %s", label, toolStatusStyle(entry.status).Render(status))
	}
	return fmt.Sprintf("%s %s", label, toolStatusStyle(entry.status).Render(status))
}

func toolGroupStatusLine(entry chatEntry) string {
	label := entry.label
	if label == "" {
		label = fmt.Sprintf("Tool calls (%d)", len(entry.tools))
	}

	status := toolStatusLabel(entry)
	if suffix := toolElapsedSuffix(entry.startedAt, entry.finishedAt); suffix != "" {
		status += suffix
	}

	if entry.expanded {
		return fmt.Sprintf("▾ %s %s", label, toolStatusStyle(entry.status).Render(status))
	}
	return fmt.Sprintf("▸ %s %s", label, toolStatusStyle(entry.status).Render(status))
}

func toolStatusLabel(entry chatEntry) string {
	if entry.err != "" || entry.status == "error" {
		return "failed"
	}
	if entry.status == "done" {
		return "done"
	}
	if entry.status == "approval" {
		return "needs approval"
	}
	return "in progress"
}

func toolStatusStyle(status string) lipgloss.Style {
	switch status {
	case "done":
		return chatToolDoneStyle
	case "error":
		return chatErrorStyle
	default:
		return chatToolRunningStyle
	}
}

func toolInvocationLabel(name string, args map[string]any) string {
	displayName := toolDisplayName(name)
	switch name {
	case "web_search":
		if query, ok := stringArg(args, "query"); ok {
			return fmt.Sprintf("%s(%s)", displayName, strconv.Quote(query))
		}
	case "web_fetch":
		if targetURL, ok := stringArg(args, "url"); ok {
			return fmt.Sprintf("%s(%s)", displayName, strconv.Quote(targetURL))
		}
	case "bash":
		if command, ok := stringArg(args, "command"); ok {
			return fmt.Sprintf("%s(%s)", displayName, strconv.Quote(command))
		}
	case "read", "list":
		if path, ok := stringArg(args, "path"); ok {
			return fmt.Sprintf("%s(%s)", displayName, strconv.Quote(path))
		}
	case "edit":
		if path, ok := stringArg(args, "path"); ok {
			return fmt.Sprintf("%s(%s)", displayName, strconv.Quote(path))
		}
	}
	if len(args) == 0 {
		return displayName
	}
	return fmt.Sprintf("%s(%s)", displayName, formatToolArgs(args))
}

func toolDisplayName(name string) string {
	switch name {
	case "web_search":
		return "Web Search"
	case "web_fetch":
		return "Web Fetch"
	case "bash":
		return "Bash"
	case "read":
		return "Read"
	case "list":
		return "List"
	case "edit":
		return "Edit"
	default:
		if name == "" {
			return "Tool"
		}
		return name
	}
}

func toolElapsedSuffix(startedAt, finishedAt time.Time) string {
	if startedAt.IsZero() || finishedAt.IsZero() || finishedAt.Before(startedAt) {
		return ""
	}
	elapsed := finishedAt.Sub(startedAt)
	if elapsed < time.Second {
		return " in " + elapsed.Round(time.Millisecond).String()
	}
	return " in " + elapsed.Round(time.Second).String()
}

func toolOutputUsesMarkdown(name string) bool {
	switch name {
	case "read", "skill", "web_search", "web_fetch":
		return true
	default:
		return false
	}
}

func formatToolArgs(args map[string]any) string {
	keys := make([]string, 0, len(args))
	for key := range args {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	parts := make([]string, 0, len(keys))
	for _, key := range keys {
		parts = append(parts, fmt.Sprintf("%s=%s", key, quoteToolArg(args[key])))
	}
	return strings.Join(parts, ", ")
}

func quoteToolArg(value any) string {
	switch v := value.(type) {
	case string:
		return strconv.Quote(truncateRunes(v, 100))
	default:
		return fmt.Sprint(v)
	}
}

func stringArg(args map[string]any, key string) (string, bool) {
	value, ok := args[key].(string)
	if !ok || strings.TrimSpace(value) == "" {
		return "", false
	}
	return truncateRunes(value, 120), true
}

func truncateRunes(value string, limit int) string {
	runes := []rune(value)
	if len(runes) <= limit {
		return value
	}
	return string(runes[:limit]) + "..."
}

func (m chatModel) statusLine() string {
	var parts []string
	if scroll := m.scrollStatus(); scroll != "" {
		parts = append(parts, scroll)
	}
	return strings.Join(parts, "  ")
}

func (m chatModel) footerLine() string {
	return strings.Join(m.footerParts(), " • ")
}

func (m chatModel) footerParts() []string {
	var parts []string
	if !m.running && !m.compacting && m.approvalPrompt == nil && m.status != "" && m.status != "ready" {
		parts = append(parts, m.status)
	}

	if len(m.queued) > 0 {
		parts = append(parts, fmt.Sprintf("queued %d", len(m.queued)))
	}

	action := "enter send"
	if m.approvalPrompt != nil {
		action = "enter approve"
	} else if m.running || m.compacting {
		action = "enter queue"
	}
	controls := action
	if m.approvalPrompt != nil {
		controls += " • ←/→ choose • o once • s session • d deny • esc deny"
	} else {
		controls += " • ctrl+c cancel/quit"
	}
	controls += " • shift+tab"
	if m.lastExpandableToolEntry() >= 0 {
		controls += " • ctrl+o details"
	}
	parts = append(parts, controls)
	parts = append(parts, m.permissionModeStatus())
	if cwd := m.cwdStatus(); cwd != "" {
		parts = append(parts, cwd)
	}
	if contextStatus := m.contextStatus(); contextStatus != "" {
		parts = append(parts, contextStatus)
	}
	return parts
}

func (m chatModel) renderFooterLine() string {
	parts := m.footerParts()
	for i, part := range parts {
		if part == "full access" {
			parts[i] = chatFullAccessStyle.Render(part)
			continue
		}
		parts[i] = chatMetaStyle.Render(part)
	}
	return strings.Join(parts, chatMetaStyle.Render(" • "))
}

func (m chatModel) permissionModeStatus() string {
	if m.autoApproveTools() {
		return "full access"
	}
	return "review"
}

func (m *chatModel) refreshContextWindowTokens(modelName string) {
	if m == nil || m.opts.ContextWindowTokensForModel == nil {
		return
	}
	modelName = strings.TrimSpace(modelName)
	if modelName == "" {
		return
	}
	ctx := m.ctx
	if ctx == nil {
		ctx = context.Background()
	}
	tokens := m.opts.ContextWindowTokensForModel(ctx, modelName, m.opts.ContextWindowTokens)
	m.updateContextWindowTokens(tokens)
}

func (m *chatModel) updateContextWindowTokens(tokens int) {
	if tokens <= 0 || tokens == m.opts.ContextWindowTokens {
		return
	}
	m.opts.ContextWindowTokens = tokens
	if compactor, ok := m.opts.Compactor.(*coreagent.SimpleCompactor); ok && compactor != nil {
		compactor.Options.ContextWindowTokens = tokens
	}
}

func (m chatModel) responseModelName(response *api.ChatResponse) string {
	if response != nil {
		if strings.TrimSpace(response.Model) != "" {
			return response.Model
		}
		if strings.TrimSpace(response.RemoteModel) != "" {
			return response.RemoteModel
		}
	}
	return m.opts.Model
}

func (m chatModel) cwdStatus() string {
	workingDir := strings.TrimSpace(m.opts.WorkingDir)
	rootDir := strings.TrimSpace(m.opts.RootDir)
	if workingDir == "" || rootDir == "" {
		return ""
	}
	rel, err := filepath.Rel(rootDir, workingDir)
	if err != nil || rel == "." {
		return ""
	}
	if rel == ".." || strings.HasPrefix(rel, ".."+string(os.PathSeparator)) {
		return "cwd " + workingDir
	}
	return "cwd ./" + filepath.ToSlash(rel)
}

func (m chatModel) activityLine() string {
	if !m.running && !m.compacting && m.approvalPrompt == nil {
		return ""
	}
	status := m.spinnerFrame()
	if label := m.activityLabel(); label != "" {
		status += " " + label
	}
	return status
}

func (m chatModel) activityLabel() string {
	if m.status == "canceling" {
		return "canceling"
	}
	if m.approvalPrompt != nil {
		return "waiting for approval"
	}
	if m.compacting {
		if m.compactingTokens > 0 {
			return "compacting " + formatTokenCount(m.compactingTokens)
		}
		return "compacting"
	}
	if m.thinking {
		if m.thinkingTokens > 0 {
			return "thinking " + formatTokenCount(m.thinkingTokens)
		}
		return "thinking"
	}
	for i := len(m.entries) - 1; i >= 0; i-- {
		entry := m.entries[i]
		switch entry.role {
		case "tool":
			if isToolActiveStatus(entry.status) {
				active := m.activeToolLabels()
				if len(active) == 1 {
					return "using " + active[0]
				}
				if len(active) > 1 {
					return fmt.Sprintf("using %d tools", len(active))
				}
				return "using tools"
			}
		case "assistant":
			if entry.content != "" {
				return ""
			}
		}
	}
	return "thinking"
}

func (m chatModel) activeToolLabels() []string {
	var labels []string
	for _, entry := range m.entries {
		if entry.role != "tool" || !isToolActiveStatus(entry.status) {
			continue
		}
		label := entry.label
		if label == "" {
			label = toolDisplayName(entry.detail)
		}
		labels = append(labels, label)
	}
	return labels
}

func (m *chatModel) applyResponseMetrics(response *api.ChatResponse) {
	if response == nil {
		return
	}
	if response.PromptEvalCount > 0 {
		m.contextTokens = response.PromptEvalCount
		m.contextEstimate = false
	}
}

func eventEvalCount(event coreagent.Event) int {
	if event.Response == nil {
		return 0
	}
	return event.Response.EvalCount
}

func (m chatModel) estimatePromptTokens(messages []api.Message, systemPrompt string) int {
	if strings.TrimSpace(systemPrompt) == "" {
		systemPrompt = m.systemPrompt("")
	}
	var tools api.Tools
	if m.opts.Tools != nil {
		tools = m.opts.Tools.Tools()
	}
	return estimatePromptTokenCount(systemPrompt, messages, tools, m.opts.Format)
}

func estimatePromptTokenCount(systemPrompt string, messages []api.Message, tools api.Tools, format string) int {
	requestMessages := slices.Clone(messages)
	if strings.TrimSpace(systemPrompt) != "" {
		requestMessages = make([]api.Message, 0, len(messages)+1)
		requestMessages = append(requestMessages, api.Message{Role: "system", Content: strings.TrimSpace(systemPrompt)})
		requestMessages = append(requestMessages, messages...)
	}
	if len(requestMessages) == 0 && len(tools) == 0 && strings.TrimSpace(format) == "" {
		return 0
	}

	payload := struct {
		Messages []api.Message   `json:"messages,omitempty"`
		Tools    api.Tools       `json:"tools,omitempty"`
		Format   json.RawMessage `json:"format,omitempty"`
	}{
		Messages: requestMessages,
		Tools:    tools,
	}
	if rawFormat, ok := promptFormatForEstimate(format); ok {
		payload.Format = rawFormat
	}

	if b, err := json.Marshal(payload); err == nil {
		return estimateTokenCount(string(b))
	}

	var runes int
	for _, msg := range requestMessages {
		runes += estimateMessageRunes(msg)
	}
	runes += len([]rune(tools.String()))
	runes += len([]rune(strings.TrimSpace(format)))
	if runes == 0 {
		return 0
	}
	return max(1, (runes+3)/4)
}

func promptFormatForEstimate(format string) (json.RawMessage, bool) {
	format = strings.TrimSpace(format)
	if format == "" {
		return nil, false
	}
	if format == "json" {
		format = `"` + format + `"`
	}
	if !json.Valid([]byte(format)) {
		return nil, false
	}
	return json.RawMessage(format), true
}

func estimateMessageRunes(msg api.Message) int {
	var runes int
	runes += len([]rune(msg.Role))
	runes += len([]rune(msg.Content))
	runes += len([]rune(msg.Thinking))
	runes += len([]rune(msg.ToolName))
	runes += len([]rune(msg.ToolCallID))
	for _, image := range msg.Images {
		runes += len(image)
	}
	for _, call := range msg.ToolCalls {
		runes += len([]rune(call.ID))
		runes += len([]rune(call.Function.Name))
		runes += len([]rune(fmt.Sprint(call.Function.Arguments.ToMap())))
	}
	return runes
}

func estimateTokenCount(text string) int {
	text = strings.TrimSpace(text)
	if text == "" {
		return 0
	}
	return max(1, (len([]rune(text))+3)/4)
}

func formatTokenCount(count int) string {
	if count == 1 {
		return "1 token"
	}
	return fmt.Sprintf("%d tokens", count)
}

func (m chatModel) contextStatus() string {
	window := coreagent.ResolveContextWindowTokens(m.opts.Options, m.opts.ContextWindowTokens)
	if window <= 0 {
		return ""
	}
	used := clamp(m.contextTokens, 0, window)
	percent := 0
	if window > 0 {
		percent = (used*100 + window/2) / window
	}

	prefix := ""
	if m.contextEstimate {
		prefix = "~"
	}

	threshold := coreagent.ResolveCompactionThreshold(m.opts.CompactionThreshold)
	compactAt := int(float64(window)*threshold + 0.999999)
	if compactAt <= 0 || compactAt > window {
		compactAt = window
	}

	if used >= compactAt {
		return fmt.Sprintf("ctx %s%s/%s (%d%%) • compact due at %s", prefix, formatInteger(used), formatInteger(window), percent, formatInteger(compactAt))
	}

	noticeDistance := int(float64(window)*0.1 + 0.999999)
	if noticeDistance < 1 {
		noticeDistance = 1
	}
	if compactAt-used <= noticeDistance {
		return fmt.Sprintf("ctx %s%s/%s (%d%%) • compact at %s", prefix, formatInteger(used), formatInteger(window), percent, formatInteger(compactAt))
	}

	return fmt.Sprintf("ctx %s%s/%s (%d%%)", prefix, formatInteger(used), formatInteger(window), percent)
}

func formatInteger(value int) string {
	sign := ""
	if value < 0 {
		sign = "-"
		value = -value
	}
	s := strconv.Itoa(value)
	if len(s) <= 3 {
		return sign + s
	}
	var b strings.Builder
	b.WriteString(sign)
	first := len(s) % 3
	if first == 0 {
		first = 3
	}
	b.WriteString(s[:first])
	for i := first; i < len(s); i += 3 {
		b.WriteByte(',')
		b.WriteString(s[i : i+3])
	}
	return b.String()
}

func (m chatModel) spinnerFrame() string {
	if len(chatSpinnerFrames) == 0 {
		return ""
	}
	return chatSpinnerFrames[m.spinner%len(chatSpinnerFrames)]
}

func (m chatModel) scrollStatus() string {
	maxScroll := m.maxScroll()
	if maxScroll <= 0 {
		return ""
	}
	scroll := clamp(m.scroll, 0, maxScroll)
	if scroll == 0 {
		return "↑ more"
	}
	if scroll == maxScroll {
		return "↓ more"
	}
	return "↑/↓ more"
}

func (m chatModel) helpSummary() string {
	return strings.Join([]string{
		"**Commands**",
		"",
		"- `/tools`: show available tools",
		"- `/history`: show prompt message history",
		"- `/skills`: show or import skills",
		"- `/<skill>`: run the next message with a skill",
		"- `/new`: start a new chat",
		"- `/resume`: resume a saved chat",
		"- `/compact`: summarize older context",
		"- `/clear`: clear this chat",
		"- `/bye`: exit",
		"",
		"**Shortcuts**",
		"",
		"- `ctrl+o`: toggle tool output and details",
		"- `shift+tab`: toggle permission mode",
		"- `↑/↓`, `pgup/pgdn`, `home/end`: scroll transcript",
	}, "\n")
}

func (m chatModel) toolsSummary() string {
	if m.opts.Tools == nil || len(m.opts.Tools.Names()) == 0 {
		return "No tools are available for this model."
	}
	var b strings.Builder
	b.WriteString("Available tools:\n\n")
	for _, name := range m.opts.Tools.Names() {
		tool, _ := m.opts.Tools.Get(name)
		b.WriteString("- **")
		b.WriteString(name)
		b.WriteString("**")
		if tool != nil && tool.Description() != "" {
			b.WriteString(": ")
			b.WriteString(tool.Description())
		}
		b.WriteByte('\n')
	}
	return strings.TrimRight(b.String(), "\n")
}

func (m chatModel) historySummary() string {
	var b strings.Builder
	b.WriteString("**Message History**\n\n")

	count := 0
	if systemPrompt := strings.TrimSpace(m.systemPrompt("")); systemPrompt != "" {
		appendHistoryMessage(&b, api.Message{Role: "system", Content: systemPrompt})
		count++
	}
	for _, msg := range m.messages {
		appendHistoryMessage(&b, msg)
		count++
	}
	if count == 0 {
		b.WriteString("No messages yet.")
	}
	return strings.TrimRight(b.String(), "\n")
}

func appendHistoryMessage(b *strings.Builder, msg api.Message) {
	role := msg.Role
	if strings.TrimSpace(role) == "" {
		role = "message"
	}
	b.WriteString("**")
	b.WriteString(role)
	b.WriteString("**\n\n")

	if msg.ToolName != "" || msg.ToolCallID != "" {
		var parts []string
		if msg.ToolName != "" {
			parts = append(parts, "tool: `"+msg.ToolName+"`")
		}
		if msg.ToolCallID != "" {
			parts = append(parts, "tool call: `"+msg.ToolCallID+"`")
		}
		b.WriteString("  ")
		b.WriteString(strings.Join(parts, " · "))
		b.WriteString("\n\n")
	}

	if strings.TrimSpace(msg.Thinking) != "" {
		appendHistoryField(b, "thinking", msg.Thinking)
	}

	if len(msg.ToolCalls) > 0 {
		b.WriteString("  tool calls:\n")
		for _, call := range msg.ToolCalls {
			appendHistoryToolCall(b, call)
		}
		b.WriteString("\n")
	}

	if strings.TrimSpace(msg.Content) != "" {
		appendHistoryField(b, "content", msg.Content)
	}

	if strings.TrimSpace(msg.Thinking) == "" && len(msg.ToolCalls) == 0 && strings.TrimSpace(msg.Content) == "" {
		b.WriteString("  _empty_\n\n")
	}
}

func appendHistoryField(b *strings.Builder, label string, content string) {
	content = strings.TrimRight(content, "\n")
	if content == "" {
		return
	}
	if !strings.Contains(content, "\n") && !strings.Contains(content, "```") {
		b.WriteString("  ")
		b.WriteString(label)
		b.WriteString(": ")
		b.WriteString(content)
		b.WriteString("\n\n")
		return
	}
	b.WriteString("  ")
	b.WriteString(label)
	b.WriteString(":\n\n")
	appendHistoryCodeBlock(b, "text", content, "  ")
}

func appendHistoryToolCall(b *strings.Builder, call api.ToolCall) {
	name := call.Function.Name
	if name == "" {
		name = "tool"
	}
	if call.ID != "" {
		b.WriteString(fmt.Sprintf("    - `%s` %s\n", call.ID, toolDisplayName(name)))
	} else {
		b.WriteString(fmt.Sprintf("    - %s\n", toolDisplayName(name)))
	}

	args := call.Function.Arguments.ToMap()
	if len(args) == 0 {
		return
	}
	b.WriteString("      args:\n\n")
	data, err := json.MarshalIndent(args, "", "  ")
	if err != nil {
		appendHistoryCodeBlock(b, "text", fmt.Sprint(args), "      ")
		return
	}
	appendHistoryCodeBlock(b, "json", string(data), "      ")
}

func appendHistoryCodeBlock(b *strings.Builder, language string, content string, indent string) {
	content = strings.TrimRight(content, "\n")
	fence := "```"
	for strings.Contains(content, fence) {
		fence += "`"
	}
	b.WriteString(indent)
	b.WriteString(fence)
	if language != "" {
		b.WriteString(language)
	}
	b.WriteString("\n")
	for _, line := range strings.Split(content, "\n") {
		b.WriteString(indent)
		b.WriteString(line)
		b.WriteString("\n")
	}
	b.WriteString(indent)
	b.WriteString(fence)
	b.WriteString("\n\n")
}

func (m chatModel) skillsSummary() string {
	if m.opts.Skills == nil || m.opts.Skills.Empty() {
		return "No skills are installed.\n\nImport skills with `/skills import claude`, `/skills import codex`, `/skills import pi`, or `/skills import all`."
	}
	return m.opts.Skills.SummaryMarkdown() + "\n\nImport more with `/skills import claude`, `/skills import codex`, `/skills import pi`, or `/skills import all`."
}

func (m *chatModel) handleSkillsCommand(input string) string {
	fields := strings.Fields(input)
	if len(fields) == 1 {
		return m.skillsSummary()
	}
	if len(fields) >= 2 && fields[1] == "import" {
		return m.importSkills(fields[2:])
	}
	return "Usage:\n\n/skills\n/skills import claude|codex|pi|agents|all [--force]"
}

func (m *chatModel) importSkills(args []string) string {
	source := "all"
	force := false
	for _, arg := range args {
		switch arg {
		case "--force":
			force = true
		default:
			if strings.TrimSpace(arg) != "" {
				source = arg
			}
		}
	}

	results, err := skills.Import(source, force)
	if err != nil {
		return err.Error()
	}
	catalog, err := skills.LoadDefault()
	if err != nil {
		return fmt.Sprintf("imported skills but could not reload catalog: %v", err)
	}
	m.opts.Skills = catalog
	if m.opts.Tools != nil && !catalog.Empty() {
		m.opts.Tools.Register(agenttools.NewSkill(catalog))
	}
	m.opts.SystemPrompt = catalog.SystemPrompt(m.opts.Tools != nil && m.opts.Tools.Has("skill"))

	if len(results) == 0 {
		return fmt.Sprintf("No skills found for %s.", source)
	}
	var imported, skipped int
	var lines []string
	for _, result := range results {
		name := result.Skill.Name
		if name == "" {
			name = result.From
		}
		if result.Skipped {
			skipped++
			line := "skipped " + name
			if result.Error != "" {
				line += " (" + result.Error + ")"
			}
			lines = append(lines, line)
			continue
		}
		imported++
		lines = append(lines, "imported "+name)
	}
	lines = append(lines, "", fmt.Sprintf("%d imported, %d skipped", imported, skipped))
	return strings.Join(lines, "\n")
}

func (m chatModel) systemPrompt(extra string) string {
	var parts []string
	if strings.TrimSpace(m.opts.SystemPrompt) != "" {
		parts = append(parts, strings.TrimSpace(m.opts.SystemPrompt))
	}
	if strings.TrimSpace(extra) != "" {
		parts = append(parts, strings.TrimSpace(extra))
	}
	return strings.Join(parts, "\n\n")
}

func (m chatModel) skillTrigger(input string) (skills.Skill, string, bool) {
	if m.opts.Skills == nil || m.opts.Skills.Empty() {
		return skills.Skill{}, "", false
	}
	command, rest, _ := strings.Cut(strings.TrimSpace(input), " ")
	name := strings.TrimPrefix(command, "/")
	skill, ok := m.opts.Skills.Find(name)
	return skill, strings.TrimSpace(rest), ok
}

func (m chatModel) skillSlashCompletions(input string) []chatCompletion {
	if m.opts.Skills == nil || m.opts.Skills.Empty() {
		return nil
	}
	prefix := strings.TrimPrefix(strings.ToLower(strings.TrimSpace(input)), "/")
	var completions []chatCompletion
	for _, skill := range m.opts.Skills.Skills {
		if !strings.HasPrefix(skill.Name, prefix) {
			continue
		}
		completions = append(completions, chatCompletion{
			value:       "/" + skill.Name,
			label:       "/" + skill.Name,
			description: skill.Description,
		})
	}
	return completions
}

type chatEventSink struct {
	ctx context.Context
	ch  chan<- tea.Msg
}

func (s chatEventSink) Emit(event coreagent.Event) error {
	select {
	case s.ch <- chatAgentMsg{event: event}:
		return nil
	case <-s.ctx.Done():
		return s.ctx.Err()
	}
}

type chatApprovalPrompter struct {
	ch chan<- tea.Msg
}

type chatPermissionApprovalHandler struct {
	mode   *chatPermissionMode
	review coreagent.ApprovalHandler
}

func (h chatPermissionApprovalHandler) RequiresApproval(ctx context.Context, tool coreagent.Tool, req coreagent.ApprovalRequest) bool {
	if h.mode != nil && h.mode.AutoApprove() {
		return false
	}
	if h.review != nil {
		return h.review.RequiresApproval(ctx, tool, req)
	}
	return coreagent.ToolRequiresApproval(tool, req.Args)
}

func (h chatPermissionApprovalHandler) Approve(ctx context.Context, req coreagent.ApprovalRequest) (coreagent.ApprovalResult, error) {
	if h.mode != nil && h.mode.AutoApprove() {
		return coreagent.ApprovalResult{Decision: coreagent.ApprovalAllowOnce}, nil
	}
	if h.review != nil {
		return h.review.Approve(ctx, req)
	}
	return coreagent.ApprovalResult{Decision: coreagent.ApprovalDeny, Reason: "Tool execution requires approval."}, nil
}

func (p chatApprovalPrompter) PromptApproval(ctx context.Context, request coreagent.ApprovalRequest) (coreagent.ApprovalResult, error) {
	reply := make(chan coreagent.ApprovalResult, 1)
	select {
	case p.ch <- chatApprovalPromptMsg{request: request, reply: reply}:
	case <-ctx.Done():
		return coreagent.ApprovalResult{Decision: coreagent.ApprovalDeny, Reason: "Tool approval canceled."}, nil
	}

	select {
	case result := <-reply:
		return result, nil
	case <-ctx.Done():
		return coreagent.ApprovalResult{Decision: coreagent.ApprovalDeny, Reason: "Tool approval canceled."}, nil
	}
}

func waitForChatMsg(ch <-chan tea.Msg) tea.Cmd {
	if ch == nil {
		return nil
	}
	return func() tea.Msg {
		msg, ok := <-ch
		if !ok {
			return nil
		}
		return msg
	}
}

func chatTickCmd() tea.Cmd {
	return tea.Tick(120*time.Millisecond, func(time.Time) tea.Msg {
		return chatTickMsg{}
	})
}

func renderFullFrame(content string, width, height int) string {
	if width <= 0 {
		width = 80
	}
	if height <= 0 {
		height = 24
	}
	rendered := lipgloss.NewStyle().MaxWidth(width).Render(content)
	lines := strings.Split(strings.TrimRight(rendered, "\n"), "\n")
	if len(lines) > height {
		lines = lines[:height]
	}
	for len(lines) < height {
		lines = append(lines, "")
	}
	return strings.Join(lines, "\n")
}

func truncateRenderedLine(line string, width int) string {
	if width <= 0 || lipgloss.Width(line) <= width {
		return line
	}
	return lipgloss.NewStyle().MaxWidth(width).Render(line)
}

func clamp(value, minValue, maxValue int) int {
	if value < minValue {
		return minValue
	}
	if value > maxValue {
		return maxValue
	}
	return value
}

func newChatEntry(entry chatEntry) chatEntry {
	if entry.version <= 0 {
		entry.version = 1
	}
	return entry
}

func (m *chatModel) markEntryDirty(index int) {
	if index < 0 || index >= len(m.entries) {
		return
	}
	entry := &m.entries[index]
	entry.version++
	if entry.version <= 0 {
		entry.version = 1
	}
	entry.renderKey = chatEntryRenderKey{}
	entry.renderLines = nil
}

func entryRenderKey(entry chatEntry, body string, width int) chatEntryRenderKey {
	if entry.version > 0 {
		return chatEntryRenderKey{width: width, version: entry.version}
	}
	return chatEntryRenderKey{width: width, hash: fallbackEntryRenderHash(entry, body)}
}

func fallbackEntryRenderHash(entry chatEntry, body string) string {
	var b strings.Builder
	writeEntryRenderHash(&b, entry)
	b.WriteString("\x00body\x00")
	b.WriteString(body)
	return b.String()
}

func writeEntryRenderHash(b *strings.Builder, entry chatEntry) {
	for _, value := range []string{
		entry.role,
		entry.content,
		entry.label,
		entry.detail,
		entry.status,
		entry.err,
		entry.toolID,
		strconv.FormatBool(entry.expanded),
		strconv.FormatInt(entry.startedAt.UnixNano(), 10),
		strconv.FormatInt(entry.finishedAt.UnixNano(), 10),
	} {
		b.WriteString(value)
		b.WriteByte(0)
	}
	for _, tool := range entry.tools {
		writeEntryRenderHash(b, tool)
	}
}

func entriesFromMessages(messages []api.Message) []chatEntry {
	entries := make([]chatEntry, 0, len(messages))
	toolCalls := make(map[string]api.ToolCall)
	for _, msg := range messages {
		switch msg.Role {
		case "user", "system":
			if summary, ok := compactionSummaryContent(msg); ok {
				entries = append(entries, newChatEntry(chatEntry{
					role:    "compaction_summary",
					content: summary,
					status:  "done",
				}))
				continue
			}
			entries = append(entries, newChatEntry(chatEntry{role: msg.Role, content: msg.Content}))
		case "assistant":
			for _, call := range msg.ToolCalls {
				if call.ID != "" {
					toolCalls[call.ID] = call
				}
			}
			if strings.TrimSpace(msg.Content) != "" {
				entries = append(entries, newChatEntry(chatEntry{role: "assistant", content: msg.Content}))
			}
		case "tool":
			toolName := msg.ToolName
			var args map[string]any
			if call, ok := toolCalls[msg.ToolCallID]; ok {
				if toolName == "" {
					toolName = call.Function.Name
				}
				args = call.Function.Arguments.ToMap()
			}
			entries = append(entries, newChatEntry(chatEntry{
				role:    "tool",
				content: msg.Content,
				label:   toolInvocationLabel(toolName, args),
				detail:  toolName,
				status:  "done",
				toolID:  msg.ToolCallID,
				args:    args,
			}))
		}
	}
	return groupCompletedToolEntries(entries)
}

func compactionSummaryContent(msg api.Message) (string, bool) {
	if msg.Role != "user" && msg.Role != "system" {
		return "", false
	}
	if !strings.HasPrefix(msg.Content, chatCompactionSummaryPrefix) {
		return "", false
	}
	return strings.TrimSpace(strings.TrimPrefix(msg.Content, chatCompactionSummaryPrefix)), true
}

func groupCompletedToolEntries(entries []chatEntry) []chatEntry {
	grouped := make([]chatEntry, 0, len(entries))
	for i := 0; i < len(entries); {
		if !isCompletedToolHistoryEntry(entries[i]) {
			grouped = append(grouped, entries[i])
			i++
			continue
		}

		start := i
		for i < len(entries) && isCompletedToolHistoryEntry(entries[i]) {
			i++
		}

		tools := flattenToolHistory(entries[start:i])
		if len(tools) <= 1 {
			grouped = append(grouped, entries[start:i]...)
			continue
		}

		group := chatEntry{
			role:       "tool_group",
			label:      fmt.Sprintf("Tool calls (%d)", len(tools)),
			status:     aggregateToolStatus(tools),
			expanded:   anyToolExpanded(tools),
			startedAt:  firstToolStartedAt(tools),
			finishedAt: lastToolFinishedAt(tools),
			tools:      tools,
		}
		if group.status == "error" {
			group.err = "one or more tool calls failed"
		}
		grouped = append(grouped, newChatEntry(group))
	}
	return grouped
}

func anyToolExpanded(tools []chatEntry) bool {
	for _, tool := range tools {
		if tool.expanded {
			return true
		}
	}
	return false
}

func isCompletedToolHistoryEntry(entry chatEntry) bool {
	return (entry.role == "tool" && isToolResultStatus(entry.status)) ||
		(entry.role == "tool_group" && len(entry.tools) > 0)
}

func flattenToolHistory(entries []chatEntry) []chatEntry {
	var tools []chatEntry
	for _, entry := range entries {
		switch entry.role {
		case "tool":
			tools = append(tools, entry)
		case "tool_group":
			tools = append(tools, entry.tools...)
		}
	}
	return tools
}

func aggregateToolStatus(tools []chatEntry) string {
	for _, tool := range tools {
		if tool.err != "" || tool.status == "error" {
			return "error"
		}
	}
	return "done"
}

func firstToolStartedAt(tools []chatEntry) time.Time {
	for _, tool := range tools {
		if !tool.startedAt.IsZero() {
			return tool.startedAt
		}
	}
	return time.Time{}
}

func lastToolFinishedAt(tools []chatEntry) time.Time {
	for i := len(tools) - 1; i >= 0; i-- {
		if !tools[i].finishedAt.IsZero() {
			return tools[i].finishedAt
		}
	}
	return time.Time{}
}

func indentLines(lines []string, prefix string) []string {
	if len(lines) == 0 {
		return nil
	}
	out := make([]string, len(lines))
	for i, line := range lines {
		if line == "" {
			out[i] = prefix
		} else {
			out[i] = prefix + line
		}
	}
	return out
}

func wrapChatText(text string, width int) []string {
	if width < 20 {
		width = 20
	}
	var out []string
	for _, rawLine := range strings.Split(text, "\n") {
		line := strings.TrimRight(rawLine, "\r")
		for len([]rune(line)) > width {
			runes := []rune(line)
			cut := width
			for i := width; i > width/2; i-- {
				if runes[i-1] == ' ' || runes[i-1] == '\t' {
					cut = i
					break
				}
			}
			out = append(out, strings.TrimSpace(string(runes[:cut])))
			line = strings.TrimSpace(string(runes[cut:]))
		}
		out = append(out, line)
	}
	if len(out) == 0 {
		return []string{""}
	}
	return out
}
