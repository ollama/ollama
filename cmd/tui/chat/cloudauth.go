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

// Sign-in/upgrade verification polling bounds. While the check is healthy but
// the user hasn't signed in yet, polling stays prompt so completion is detected
// quickly. When the check itself fails, polling backs off so a down server
// isn't hammered, and gives up after maxPollFailures consecutive errors (or
// pollHardCap elapsed) so the user isn't stuck on a spinner with no recourse
// beyond Esc.
const (
	maxPollFailures = 6
	pollBackoffBase = 3 * time.Second
	pollBackoffCap  = 30 * time.Second
	pollHardCap     = 2 * time.Minute
)

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
	// pollStarted tracks when sign-in/upgrade verification polling began, for
	// the hard-cap timeout. Lazily set on the first poll response.
	pollStarted time.Time
	// pollFailures counts consecutive verification-check errors; once it
	// reaches maxPollFailures the modal gives up and surfaces an error.
	pollFailures int
	// pollErr holds the last verification error, rendered while retrying.
	pollErr string
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
	err  error
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

func pollCloudAuthCmd(ctx context.Context, poll func(context.Context) (string, bool, error), delay time.Duration) tea.Cmd {
	if poll == nil {
		return nil
	}
	return func() tea.Msg {
		if ctx == nil {
			ctx = context.Background()
		}
		// Back off before the next check when the previous one failed. Honor
		// context cancellation so an abandoned modal doesn't block on the
		// full delay.
		if delay > 0 {
			timer := time.NewTimer(delay)
			defer timer.Stop()
			select {
			case <-ctx.Done():
			case <-timer.C:
			}
		}
		pollCtx, cancel := context.WithTimeout(ctx, 3*time.Second)
		defer cancel()
		_, done, err := poll(pollCtx)
		return cloudAuthPollMsg{done: done, err: err}
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
	return m, tea.Batch(cloudAuthTickCmd(), pollCloudAuthCmd(m.ctx, m.opts.PollCloudAuth, 0))
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
			return m, tea.Batch(cloudAuthTickCmd(), pollCloudAuthCmd(m.ctx, m.opts.PollCloudAuth, 0))
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
			m.cloudAuthPrompt.pollFailures = 0
			m.cloudAuthPrompt.pollErr = ""
			return m, checkCloudModelCmd(m.ctx, m.opts.CheckCloudModel, m.cloudAuthPrompt.modelName, m.cloudAuthPrompt.requiredPlan)
		}
		// Lazily mark the start of the polling window on the first response.
		if m.cloudAuthPrompt.pollStarted.IsZero() {
			m.cloudAuthPrompt.pollStarted = time.Now()
		}
		// Hard cap: give up if verification drags on too long for any reason.
		if time.Since(m.cloudAuthPrompt.pollStarted) > pollHardCap {
			return m.failCloudAuthPoll(errors.New("sign-in is taking longer than expected; check your connection and try again"))
		}
		if msg.err != nil {
			// The verification check itself failed (network down, server 5xx).
			// Back off and retry, but give up after a handful of consecutive
			// failures so the user isn't stuck on a spinner with no signal.
			m.cloudAuthPrompt.pollFailures++
			m.cloudAuthPrompt.pollErr = msg.err.Error()
			if m.cloudAuthPrompt.pollFailures >= maxPollFailures {
				return m.failCloudAuthPoll(fmt.Errorf("couldn't verify sign-in: %w", msg.err))
			}
			delay := pollBackoffCap
			if d := pollBackoffBase << (m.cloudAuthPrompt.pollFailures - 1); d < pollBackoffCap {
				delay = d
			}
			return m, pollCloudAuthCmd(m.ctx, m.opts.PollCloudAuth, delay)
		}
		// Healthy but not signed in yet — keep polling promptly so sign-in
		// completion is detected without added latency.
		m.cloudAuthPrompt.pollFailures = 0
		m.cloudAuthPrompt.pollErr = ""
		return m, pollCloudAuthCmd(m.ctx, m.opts.PollCloudAuth, 0)

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
					return m, tea.Batch(cloudAuthTickCmd(), pollCloudAuthCmd(m.ctx, m.opts.PollCloudAuth, 0))
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

// failCloudAuthPoll abandons the sign-in/upgrade verification modal, surfaces
// an error entry to the user, and returns to the ready state so they can
// re-pick a model and retry.
func (m chatModel) failCloudAuthPoll(err error) (tea.Model, tea.Cmd) {
	m.cloudAuthPrompt = nil
	m.pendingModel = ""
	m.openModelOnInit = false
	m.status = "ready"
	m.entries = append(m.entries, newChatEntry(chatEntry{role: "error", content: fmt.Sprintf("Could not switch model: %v", err), err: err.Error()}))
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
		if p.pollErr != "" {
			b.WriteString(chatPickerMetaStyle.Render(frame + " Couldn't verify sign-in: " + p.pollErr + " — retrying..."))
		} else {
			b.WriteString(chatPickerMetaStyle.Render(frame + " Waiting for sign in to complete..."))
		}
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
			if p.pollErr != "" {
				b.WriteString(chatPickerMetaStyle.Render(frame + " Couldn't verify upgrade: " + p.pollErr + " — retrying..."))
			} else {
				b.WriteString(chatPickerMetaStyle.Render(frame + " Waiting for upgrade to complete..."))
			}
			b.WriteString("\n\n")
			b.WriteString(chatPickerMetaStyle.Render("esc cancel"))
		}
	}

	return b.String()
}
