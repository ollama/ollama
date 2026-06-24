package chat

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/ollama/ollama/api"
)

type cloudAuthKind string

const (
	cloudAuthSignIn   cloudAuthKind = "signin"
	cloudAuthUpgrade  cloudAuthKind = "upgrade"
	cloudAuthChecking cloudAuthKind = "checking"
)

// cloudAuthPrompt is an inline modal that handles sign-in and plan-upgrade
// flows when a user selects a cloud model from the picker.
type cloudAuthPrompt struct {
	modelName    string
	requiredPlan string
	signInURL    string
	kind         cloudAuthKind
	spinner      int
	openNow      bool
	polling      bool
}

type cloudAuthCheckMsg struct {
	err       error
	signInURL string
}

type cloudAuthTickMsg struct{}

type cloudAuthPollMsg struct {
	done bool
}

func checkCloudModelCmd(ctx context.Context, check func(context.Context, string, string) error, model, requiredPlan string) tea.Cmd {
	if check == nil {
		return nil
	}
	return func() tea.Msg {
		if ctx == nil {
			ctx = context.Background()
		}
		err := check(ctx, model, requiredPlan)
		var signInURL string
		if err != nil {
			var authErr api.AuthorizationError
			if errors.As(err, &authErr) && authErr.SigninURL != "" {
				signInURL = authErr.SigninURL
			}
		}
		return cloudAuthCheckMsg{err: err, signInURL: signInURL}
	}
}

func cloudAuthTickCmd() tea.Cmd {
	return tea.Tick(200*time.Millisecond, func(t time.Time) tea.Msg {
		return cloudAuthTickMsg{}
	})
}

func pollCloudAuthCmd(ctx context.Context, poll func(context.Context) (string, bool)) tea.Cmd {
	if poll == nil {
		return nil
	}
	return func() tea.Msg {
		if ctx == nil {
			ctx = context.Background()
		}
		pollCtx, cancel := context.WithTimeout(ctx, 3*time.Second)
		defer cancel()
		_, done := poll(pollCtx)
		return cloudAuthPollMsg{done: done}
	}
}

func (m *chatModel) startCloudAuthSignIn(modelName, requiredPlan, signInURL string) (tea.Model, tea.Cmd) {
	m.cloudAuthPrompt = &cloudAuthPrompt{
		modelName:    modelName,
		requiredPlan: requiredPlan,
		kind:         cloudAuthSignIn,
		signInURL:    signInURL,
		polling:      true,
	}
	m.status = "cloud-auth"
	m.modelPicker = nil
	if m.opts.OpenBrowser != nil && signInURL != "" {
		m.opts.OpenBrowser(signInURL)
	}
	if signInURL == "" {
		return m, checkCloudModelCmd(m.ctx, m.opts.CheckCloudModel, modelName, requiredPlan)
	}
	return m, tea.Batch(cloudAuthTickCmd(), pollCloudAuthCmd(m.ctx, m.opts.PollCloudAuth))
}

func (m *chatModel) startCloudAuthUpgrade(modelName, requiredPlan string) (tea.Model, tea.Cmd) {
	m.cloudAuthPrompt = &cloudAuthPrompt{
		modelName:    modelName,
		requiredPlan: requiredPlan,
		kind:         cloudAuthUpgrade,
		openNow:      true,
	}
	m.status = "cloud-auth"
	m.modelPicker = nil
	return m, nil
}

func (m chatModel) updateCloudAuthPrompt(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case cloudAuthCheckMsg:
		if msg.err == nil {
			// Auth passed — apply the pending model.
			return m.completeCloudAuth()
		}
		// Determine if sign-in or upgrade is needed.
		if msg.signInURL != "" {
			m.cloudAuthPrompt.kind = cloudAuthSignIn
			m.cloudAuthPrompt.signInURL = msg.signInURL
			m.cloudAuthPrompt.polling = true
			if m.opts.OpenBrowser != nil {
				m.opts.OpenBrowser(msg.signInURL)
			}
			return m, tea.Batch(cloudAuthTickCmd(), pollCloudAuthCmd(m.ctx, m.opts.PollCloudAuth))
		}
		// Could be a plan upgrade error or unknown error.
		m.cloudAuthPrompt = nil
		m.status = "ready"
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "error", content: fmt.Sprintf("Could not switch model: %v", msg.err), err: msg.err.Error()}))
		return m, nil

	case cloudAuthTickMsg:
		if m.cloudAuthPrompt == nil {
			return m, nil
		}
		m.cloudAuthPrompt.spinner++
		if m.cloudAuthPrompt.polling {
			return m, tea.Batch(cloudAuthTickCmd(), pollCloudAuthCmd(m.ctx, m.opts.PollCloudAuth))
		}
		return m, cloudAuthTickCmd()

	case cloudAuthPollMsg:
		if m.cloudAuthPrompt == nil {
			return m, nil
		}
		if msg.done {
			// Signed in — re-check auth to see if plan is satisfied.
			m.cloudAuthPrompt.polling = false
			return m, checkCloudModelCmd(m.ctx, m.opts.CheckCloudModel, m.cloudAuthPrompt.modelName, m.cloudAuthPrompt.requiredPlan)
		}
		return m, cloudAuthTickCmd()

	case tea.KeyMsg:
		if msg.Type == tea.KeyEsc || msg.Type == tea.KeyCtrlC {
			m.cloudAuthPrompt = nil
			m.pendingModel = ""
			m.status = "ready"
			return m, nil
		}
		if m.cloudAuthPrompt.kind == cloudAuthUpgrade && !m.cloudAuthPrompt.polling {
			switch msg.Type {
			case tea.KeyLeft, tea.KeyRight, tea.KeyTab:
				m.cloudAuthPrompt.openNow = !m.cloudAuthPrompt.openNow
			case tea.KeyEnter:
				if m.cloudAuthPrompt.openNow {
					m.cloudAuthPrompt.polling = true
					return m, tea.Batch(cloudAuthTickCmd(), pollCloudAuthCmd(m.ctx, m.opts.PollCloudAuth))
				}
				m.cloudAuthPrompt = nil
				m.pendingModel = ""
				m.status = "ready"
				return m, nil
			}
		}
	}

	return m, nil
}

func (m chatModel) completeCloudAuth() (tea.Model, tea.Cmd) {
	pending := m.cloudAuthPrompt.modelName
	m.cloudAuthPrompt = nil
	m.pendingModel = ""
	m.modelPicker = nil
	m.status = "ready"
	if err := m.applyModelSelection(pending, true); err != nil {
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "error", content: fmt.Sprintf("Could not switch model: %v", err), err: err.Error()}))
		m.status = "error"
		return m, nil
	}
	return m, m.startModelPreload(pending)
}

func (m chatModel) renderCloudAuthPrompt(width int) string {
	if m.cloudAuthPrompt == nil {
		return ""
	}
	if width <= 0 {
		width = 80
	}

	p := m.cloudAuthPrompt
	spinnerFrames := []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}
	frame := spinnerFrames[p.spinner%len(spinnerFrames)]

	var b strings.Builder

	switch p.kind {
	case cloudAuthChecking:
		fmt.Fprintf(&b, "%s Checking %s...\n\n", frame, chatResumeSelectedStyle.Render(p.modelName))
		b.WriteString(chatResumeMetaStyle.Render("esc cancel"))
	case cloudAuthSignIn:
		fmt.Fprintf(&b, "To use %s, please sign in.\n\n", chatResumeSelectedStyle.Render(p.modelName))
		b.WriteString("Navigate to:\n")
		urlWrap := chatResumeTextStyle
		if width > 4 {
			urlWrap = chatResumeTextStyle.Width(width - 4)
		}
		b.WriteString(urlWrap.Render(p.signInURL))
		b.WriteString("\n\n")
		b.WriteString(chatResumeMetaStyle.Render(frame + " Waiting for sign in to complete..."))
		b.WriteString("\n\n")
		b.WriteString(chatResumeMetaStyle.Render("esc cancel"))
	case cloudAuthUpgrade:
		fmt.Fprintf(&b, "To use %s, upgrade your Ollama plan.\n\n", chatResumeSelectedStyle.Render(p.modelName))
		if !p.polling {
			var yesBtn, noBtn string
			if p.openNow {
				yesBtn = chatResumeSelectedStyle.Render("› Yes  ")
				noBtn = chatResumeMetaStyle.Render("  No  ")
			} else {
				yesBtn = chatResumeMetaStyle.Render("  Yes  ")
				noBtn = chatResumeSelectedStyle.Render("› No  ")
			}
			b.WriteString("Open upgrade page now?\n")
			b.WriteString(yesBtn + "  " + noBtn)
			b.WriteString("\n\n")
			b.WriteString(chatResumeMetaStyle.Render("←/→ navigate • enter confirm • esc cancel"))
		} else {
			b.WriteString(chatResumeMetaStyle.Render(frame + " Waiting for upgrade to complete..."))
			b.WriteString("\n\n")
			b.WriteString(chatResumeMetaStyle.Render("esc cancel"))
		}
	}

	return b.String()
}
