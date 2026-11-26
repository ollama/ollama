//go:build windows

package main

import (
	"context"
	_ "embed"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"text/template"
	"time"

	"github.com/ollama/ollama/app/assets"
	"github.com/ollama/ollama/app/winui"
)

// Onboarding matches the welcome flow earlier versions of the app showed in
// their main window, including its light appearance.
var (
	//go:embed onboarding_windows.xaml
	onboardingPageXAML string

	onboardingPageTemplate = template.Must(template.New("onboarding").Parse(onboardingPageXAML))
)

// firstModelCommand is what onboarding asks people to run to start a model.
const firstModelCommand = "ollama"

// While people sign in with their browser, onboarding checks for the new
// account this often, for up to signInTimeout.
const onboardingSignInPollInterval = time.Second

type onboardingStep int

const (
	onboardingWelcome onboardingStep = iota
	onboardingAccount
	onboardingRun
)

// openOnboarding is the onboarding window while it's open. Use it only on
// the UI thread.
var openOnboarding *onboardingWindow

func showOnboarding() {
	go func() {
		if err := winui.Start(settingsResourcesXAML); err != nil {
			slog.Error("failed to start the onboarding window", "error", err)
			showWindowUnavailable(err)
			return
		}
		winui.Post(func() {
			if openOnboarding == nil {
				w, err := newOnboardingWindow()
				if err != nil {
					slog.Error("failed to open onboarding", "error", err)
					return
				}
				openOnboarding = w
			}
			logUIError("activate", openOnboarding.window.Activate())
		})
	}()
}

// onboardingWindow welcomes people the first time they open Ollama: it
// offers an ollama.com account, then shows how to run a model. Its methods
// run on the UI thread.
type onboardingWindow struct {
	window   *winui.Window
	elements map[string]*winui.Element
	// ctx is canceled when the window closes, which drops pending results.
	ctx    context.Context //nolint:containedctx // Scoped to the window's lifetime.
	cancel context.CancelFunc

	step     onboardingStep
	signedIn bool
	// signInAttempt changes when people start or stop signing in, so
	// results from an earlier attempt are ignored.
	signInAttempt int
	signInStarted time.Time // zero unless waiting for people to sign in
	checking      bool
}

func newOnboardingWindow() (*onboardingWindow, error) {
	page, err := onboardingPage()
	if err != nil {
		return nil, err
	}
	window, err := winui.NewWindow("Welcome to Ollama", page)
	if err != nil {
		return nil, err
	}
	w := &onboardingWindow{window: window, elements: map[string]*winui.Element{}}
	w.ctx, w.cancel = context.WithCancel(context.Background())

	logUIError("extend into title bar", window.ExtendIntoTitleBar("TitleBar"))
	logUIError("use a light title bar", window.UseLightTitleBar())
	if icon := settingsIconPath(); icon != "" {
		logUIError("set icon", window.SetIcon(icon))
	}
	logUIError("resize", window.Resize(560, 600))
	logUIError("set minimum size", window.SetMinimumSize(480, 560))
	logUIError("center", window.Center())
	logUIError("observe closing", window.OnClosed(w.closed))
	logUIError("observe activation", window.OnActivated(w.activated))

	click := map[string]func(){
		"ContinueButton":    w.continueFromWelcome,
		"SignUpButton":      func() { w.signIn(true) },
		"TitleSignInButton": func() { w.signIn(false) },
		"LocalButton":       w.useLocally,
		"CopyButton":        w.copyCommand,
		"RetryButton":       w.saveCompletion,
	}
	for name, fn := range click {
		logUIError("handle "+name, w.el(name).OnClick(fn))
	}

	run(w.ctx, func(ctx context.Context) (accountState, error) {
		return settings.Account(ctx)
	}, func(account accountState, _ error) {
		w.signedIn = account.SignedIn
		w.updateTitleSignIn()
	})
	return w, nil
}

// el returns the element named name, looking it up once.
func (w *onboardingWindow) el(name string) *winui.Element {
	if e, ok := w.elements[name]; ok {
		return e
	}
	e, err := w.window.Element(name)
	if err != nil {
		slog.Error("onboarding window element missing", "name", name, "error", err)
	}
	w.elements[name] = e
	return e
}

func (w *onboardingWindow) closed() {
	w.cancel()
	for _, e := range w.elements {
		e.Release()
	}
	w.elements = nil
	if openOnboarding == w {
		openOnboarding = nil
	}
}

// activated checks for the account when people come back from signing in
// with their browser.
func (w *onboardingWindow) activated(active bool) {
	if active {
		w.checkSignIn()
	}
}

func (w *onboardingWindow) showStep(step onboardingStep) {
	w.step = step
	w.el("WelcomeStep").SetVisible(step == onboardingWelcome)
	w.el("AccountStep").SetVisible(step == onboardingAccount)
	w.el("RunStep").SetVisible(step == onboardingRun)
	w.showError("")
	w.updateTitleSignIn()
}

// updateTitleSignIn offers Sign in in the title bar on the account step,
// for people who already have an account.
func (w *onboardingWindow) updateTitleSignIn() {
	w.el("TitleSignInButton").SetVisible(w.step == onboardingAccount && !w.signedIn && w.signInStarted.IsZero())
}

func (w *onboardingWindow) continueFromWelcome() {
	if w.signedIn {
		w.finish()
		return
	}
	w.showStep(onboardingAccount)
}

func (w *onboardingWindow) signIn(signUp bool) {
	w.showError("")
	if w.signedIn {
		w.finish()
		return
	}
	w.signInAttempt++
	attempt := w.signInAttempt
	w.signInStarted = time.Now()
	w.showSignUp()
	run(w.ctx, func(context.Context) (string, error) {
		return onboardingSignInURL(signUp)
	}, func(signInURL string, err error) {
		if attempt != w.signInAttempt {
			return
		}
		if err != nil {
			slog.Warn("failed to start sign in", "error", err)
			w.stopSigningIn()
			w.showError("Unable to start sign in. Please try again.")
			return
		}
		openInBrowser(signInURL)
		w.pollSignIn(attempt)
	})
}

func (w *onboardingWindow) pollSignIn(attempt int) {
	after(w.ctx, onboardingSignInPollInterval, func() {
		if attempt != w.signInAttempt {
			return
		}
		w.checkSignIn()
		w.pollSignIn(attempt)
	})
}

// checkSignIn looks for the account people sign in to with their browser.
func (w *onboardingWindow) checkSignIn() {
	if w.signInStarted.IsZero() || w.checking {
		return
	}
	if time.Since(w.signInStarted) > signInTimeout {
		w.stopSigningIn()
		w.showError("Connection is taking longer than expected. Please try again.")
		return
	}
	attempt := w.signInAttempt
	w.checking = true
	run(w.ctx, func(ctx context.Context) (accountState, error) {
		return settings.Account(ctx)
	}, func(account accountState, _ error) {
		w.checking = false
		if attempt != w.signInAttempt || !account.SignedIn {
			return
		}
		w.signedIn = true
		w.stopSigningIn()
		logUIError("activate", w.window.Activate())
		w.finish()
	})
}

func (w *onboardingWindow) stopSigningIn() {
	w.signInAttempt++
	w.signInStarted = time.Time{}
	w.showSignUp()
}

func (w *onboardingWindow) showSignUp() {
	waiting := !w.signInStarted.IsZero()
	label := "Sign up"
	if waiting {
		label = "Finish in your browser…"
	}
	w.el("SignUpButton").SetContent(label)
	w.el("SignUpButton").SetEnabled(!waiting)
	w.updateTitleSignIn()
}

func (w *onboardingWindow) useLocally() {
	w.stopSigningIn()
	w.finish()
}

// finish shows how to run a model and saves that onboarding is done.
func (w *onboardingWindow) finish() {
	w.showStep(onboardingRun)
	w.saveCompletion()
}

func (w *onboardingWindow) saveCompletion() {
	w.showError("")
	run(w.ctx, func(context.Context) (struct{}, error) {
		return struct{}{}, completeOnboarding()
	}, func(_ struct{}, err error) {
		if err != nil {
			slog.Warn("failed to save onboarding", "error", err)
			w.showError("Unable to save setup. Please try again.")
		}
	})
}

// showError shows message under the current step, or hides the error when
// message is empty.
func (w *onboardingWindow) showError(message string) {
	name := "AccountError"
	if w.step == onboardingRun {
		name = "RunError"
		w.el("RetryButton").SetVisible(message != "")
	}
	w.el(name).SetText(message)
	w.el(name).SetVisible(message != "")
}

func (w *onboardingWindow) copyCommand() {
	if err := copyToClipboard(firstModelCommand); err != nil {
		slog.Warn("failed to copy the command", "error", err)
		return
	}
	w.el("CopyIcon").SetVisible(false)
	w.el("CopiedIcon").SetVisible(true)
	after(w.ctx, 2*time.Second, func() {
		w.el("CopyIcon").SetVisible(true)
		w.el("CopiedIcon").SetVisible(false)
	})
}

// onboardingPage returns the window's markup.
func onboardingPage() (string, error) {
	data := struct {
		Hello   string
		Command string
	}{Command: firstModelCommand}
	if hello := onboardingImagePath(); hello != "" {
		data.Hello = xamlFileURL(hello)
	}
	var page strings.Builder
	if err := onboardingPageTemplate.Execute(&page, data); err != nil {
		return "", err
	}
	return page.String(), nil
}

// onboardingImagePath writes the waving Ollama onboarding shows to a file,
// which XAML can load.
func onboardingImagePath() string {
	data, err := assets.GetImage("hello.png")
	if err != nil {
		return ""
	}
	path := filepath.Join(os.TempDir(), "ollama_hello.png")
	if err := os.WriteFile(path, data, 0o644); err != nil {
		return ""
	}
	return path
}
