package chat

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/cmd/launch"
	"github.com/ollama/ollama/internal/modelref"
)

type cloudAuthKind string

const (
	cloudAuthSignIn   cloudAuthKind = "signin"
	cloudAuthUpgrade  cloudAuthKind = "upgrade"
	cloudAuthChecking cloudAuthKind = "checking"
)

const cloudPlanVerificationUnavailable = "Could not verify Ollama plan. Try again in a moment or use a local model."

// cloudAuthPrompt is an inline modal that handles sign-in and plan-upgrade
// flows when a user selects a cloud model from the picker.
type cloudAuthPrompt struct {
	modelName    string
	requiredPlan string
	signInURL    string
	upgradeURL   string
	kind         cloudAuthKind
	spinner      int
	openNow      bool
	polling      bool
}

type cloudAuthCheckMsg struct {
	err       error
	signInURL string
}

type cloudModelPreflightMsg struct {
	model     string
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

func cloudModelPreflightCmd(ctx context.Context, opts Options, modelName, requiredPlan string) tea.Cmd {
	modelName = strings.TrimSpace(modelName)
	if opts.CheckCloudModel == nil || modelName == "" || !modelref.HasExplicitCloudSource(modelName) {
		return nil
	}
	return func() tea.Msg {
		if ctx == nil {
			ctx = context.Background()
		}
		plan := strings.TrimSpace(requiredPlan)
		if plan == "" && opts.ModelOptions != nil {
			models, err := opts.ModelOptions(ctx)
			if err == nil {
				for _, model := range models {
					if strings.EqualFold(strings.TrimSpace(model.Name), modelName) {
						plan = strings.TrimSpace(model.RequiredPlan)
						break
					}
				}
			}
		}
		err := opts.CheckCloudModel(ctx, modelName, plan)
		return cloudModelPreflightMsg{
			model:     modelName,
			err:       err,
			signInURL: cloudAuthSignInURL(err),
		}
	}
}

func cloudAuthSignInURL(err error) string {
	if err == nil {
		return ""
	}
	var authErr api.AuthorizationError
	if errors.As(err, &authErr) && (authErr.StatusCode == http.StatusUnauthorized || authErr.SigninURL != "") {
		return authErr.SigninURL
	}
	return ""
}

func cloudAuthTickCmd() tea.Cmd {
	return tea.Tick(200*time.Millisecond, func(t time.Time) tea.Msg {
		return cloudAuthTickMsg{}
	})
}

func (m chatModel) updateCloudModelPreflight(msg cloudModelPreflightMsg) (tea.Model, tea.Cmd) {
	if msg.model == "" || !strings.EqualFold(strings.TrimSpace(m.opts.Model), strings.TrimSpace(msg.model)) {
		return m, nil
	}
	if msg.err == nil {
		if m.status == cloudPlanVerificationUnavailable {
			m.status = "ready"
		}
		return m, nil
	}
	if msg.signInURL != "" {
		return m.startCloudAuthSignIn(msg.model, "", msg.signInURL)
	}
	m.status = cloudPlanVerificationUnavailable
	return m, nil
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
	// When no sign-in URL is available yet, show the "checking" state while
	// we verify the plan, rather than rendering a blank "Navigate to:" URL.
	kind := cloudAuthSignIn
	if signInURL == "" {
		kind = cloudAuthChecking
	}
	m.cloudAuthPrompt = &cloudAuthPrompt{
		modelName:    modelName,
		requiredPlan: requiredPlan,
		kind:         kind,
		signInURL:    signInURL,
		polling:      true,
	}
	m.status = "cloud-auth"
	m.modelPicker = nil
	m.modelPickerModels = nil
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
		upgradeURL:   launch.DefaultUpgradeURL,
		openNow:      true,
	}
	m.status = "cloud-auth"
	m.modelPicker = nil
	m.modelPickerModels = nil
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
		m.openModelOnInit = false
		m.status = "ready"
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "error", content: fmt.Sprintf("Could not switch model: %v", msg.err), err: msg.err.Error()}))
		return m, nil

	case cloudAuthTickMsg:
		if m.cloudAuthPrompt == nil {
			return m, nil
		}
		m.cloudAuthPrompt.spinner++
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
		return m, pollCloudAuthCmd(m.ctx, m.opts.PollCloudAuth)

	case tea.KeyMsg:
		if msg.Type == tea.KeyEsc || msg.Type == tea.KeyCtrlC {
			m.cloudAuthPrompt = nil
			m.pendingModel = ""
			m.openModelOnInit = false
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
					if m.opts.OpenBrowser != nil && m.cloudAuthPrompt.upgradeURL != "" {
						m.opts.OpenBrowser(m.cloudAuthPrompt.upgradeURL)
					}
					return m, tea.Batch(cloudAuthTickCmd(), pollCloudAuthCmd(m.ctx, m.opts.PollCloudAuth))
				}
				m.cloudAuthPrompt = nil
				m.pendingModel = ""
				m.openModelOnInit = false
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
	m.modelPickerModels = nil
	m.openModelOnInit = false
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
		fmt.Fprintf(&b, "%s Checking %s...\n\n", frame, chatPickerSelectedStyle.Render(p.modelName))
		b.WriteString(chatPickerMetaStyle.Render("esc cancel"))
	case cloudAuthSignIn:
		fmt.Fprintf(&b, "To use %s, please sign in.\n\n", chatPickerSelectedStyle.Render(p.modelName))
		b.WriteString("Navigate to:\n")
		urlWrap := chatPickerTextStyle
		if width > 4 {
			urlWrap = chatPickerTextStyle.Width(width - 4)
		}
		b.WriteString(urlWrap.Render(p.signInURL))
		b.WriteString("\n\n")
		b.WriteString(chatPickerMetaStyle.Render(frame + " Waiting for sign in to complete..."))
		b.WriteString("\n\n")
		b.WriteString(chatPickerMetaStyle.Render("esc cancel"))
	case cloudAuthUpgrade:
		fmt.Fprintf(&b, "To use %s, upgrade your Ollama plan.\n\n", chatPickerSelectedStyle.Render(p.modelName))
		if !p.polling {
			var yesBtn, noBtn string
			if p.openNow {
				yesBtn = chatPickerSelectedStyle.Render("› Yes  ")
				noBtn = chatPickerMetaStyle.Render("  No  ")
			} else {
				yesBtn = chatPickerMetaStyle.Render("  Yes  ")
				noBtn = chatPickerSelectedStyle.Render("› No  ")
			}
			b.WriteString("Open upgrade page now?\n")
			b.WriteString(yesBtn + "  " + noBtn)
			b.WriteString("\n\n")
			if !p.openNow {
				b.WriteString("Or navigate to:\n")
				urlWrap := chatPickerTextStyle
				if width > 4 {
					urlWrap = chatPickerTextStyle.Width(width - 4)
				}
				if u := p.upgradeURL; u != "" {
					b.WriteString(urlWrap.Render(u))
				} else {
					b.WriteString(urlWrap.Render(launch.DefaultUpgradeURL))
				}
				b.WriteString("\n\n")
			}
			b.WriteString(chatPickerMetaStyle.Render("←/→ navigate • enter confirm • esc cancel"))
		} else {
			b.WriteString(chatPickerMetaStyle.Render(frame + " Waiting for upgrade to complete..."))
			b.WriteString("\n\n")
			b.WriteString(chatPickerMetaStyle.Render("esc cancel"))
		}
	}

	return b.String()
}
