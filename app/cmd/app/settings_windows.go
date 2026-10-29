//go:build windows

package main

import (
	"context"
	_ "embed"
	"errors"
	"fmt"
	"html"
	"log/slog"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"text/template"
	"time"

	"golang.org/x/sys/windows"

	"github.com/ollama/ollama/app/assets"
	"github.com/ollama/ollama/app/winui"
)

// The settings window is WinUI 3 markup that's parsed when the window
// opens, so the app builds with go build alone.
var (
	//go:embed settings_windows.xaml
	settingsPageXAML string
	//go:embed settings_resources_windows.xaml
	settingsResourcesXAML string

	settingsPageTemplate = template.Must(template.New("settings").Parse(settingsPageXAML))
)

const (
	// After people start signing in with their browser, Settings checks for
	// the new account this often, for up to signInTimeout.
	signInPollInterval = 3 * time.Second
	signInTimeout      = 5 * time.Minute

	// While the server starts, Settings checks for the context length it
	// picks this often, up to maxContextLengthChecks times.
	contextLengthCheckInterval = 2 * time.Second
	maxContextLengthChecks     = 10
)

// openSettings is the settings window while it's open. Use it only on the
// UI thread.
var openSettings *settingsWindow

// showSettings opens the settings window, or brings it to the front. Every
// pane is on the same page on Windows.
func showSettings(settingsPane) {
	go func() {
		if err := winui.Start(settingsResourcesXAML); err != nil {
			slog.Error("failed to start the settings window", "error", err)
			showSettingsUnavailable(err)
			return
		}
		winui.Post(func() {
			if openSettings == nil {
				w, err := newSettingsWindow()
				if err != nil {
					slog.Error("failed to open settings", "error", err)
					return
				}
				openSettings = w
			}
			openSettings.show()
		})
	}()
}

// settingsDidChange refreshes the settings window, if it's open, after a
// change made elsewhere, such as an update finishing its download.
func settingsDidChange() {
	winui.Post(func() {
		if openSettings != nil {
			openSettings.loadSettings()
		}
	})
}

// closeSettings shuts down the settings window's UI thread, if it started.
// The process can hang on exit while it runs.
func closeSettings() {
	winui.Stop()
}

// showSettingsUnavailable explains why the settings window can't open.
func showSettingsUnavailable(err error) {
	const title = "Ollama Settings"
	switch {
	case errors.Is(err, winui.ErrRuntimeMissing):
		if messageBox(title, "Ollama Settings needs the Windows App Runtime 1.8.\n\nDownload it now?",
			windows.MB_YESNO|windows.MB_ICONINFORMATION) == idYes {
			openInBrowser(winui.RuntimeDownloadURL)
		}
	case errors.Is(err, winui.ErrUnsupported):
		messageBox(title, "Ollama Settings needs Windows 10 version 1809 or later. Ollama keeps running in the notification area.",
			windows.MB_OK|windows.MB_ICONINFORMATION)
	default:
		messageBox(title, "Ollama couldn’t open Settings: "+err.Error(), windows.MB_OK|windows.MB_ICONERROR)
	}
}

// idYes is the MessageBox result for the Yes button.
const idYes = 6

func messageBox(title, text string, style uint32) int32 {
	titlePtr, _ := windows.UTF16PtrFromString(title)
	textPtr, _ := windows.UTF16PtrFromString(text)
	result, _ := windows.MessageBox(0, textPtr, titlePtr, style)
	return result
}

// settingsWindow shows and changes settings. Its methods run on the UI
// thread; work that can block runs on other goroutines and reports back
// with winui.Post.
type settingsWindow struct {
	window   *winui.Window
	elements map[string]*winui.Element
	// ctx is canceled when the window closes, which drops pending results.
	ctx    context.Context //nolint:containedctx // Scoped to the window's lifetime.
	cancel context.CancelFunc

	// updating is set while the window shows new values, so the controls'
	// change events don't apply them again.
	updating bool
	settings *settingsState
	account  *accountState

	accountNote       string    // why the account shown may be out of date
	shownPerson       string    // the name the account picture shows
	shownPicture      string    // the URL of the account picture shown
	signInStarted     time.Time // zero unless waiting for people to sign in
	refreshingAccount bool

	contextLengthChecks int
	inactive            bool // the window lost focus since it last loaded
}

func newSettingsWindow() (*settingsWindow, error) {
	icon := settingsIconPath()
	page, err := settingsPage(icon)
	if err != nil {
		return nil, err
	}
	window, err := winui.NewWindow("Ollama", page)
	if err != nil {
		return nil, err
	}
	w := &settingsWindow{window: window, elements: map[string]*winui.Element{}}
	w.ctx, w.cancel = context.WithCancel(context.Background())

	logUIError("extend into title bar", window.ExtendIntoTitleBar("TitleBar"))
	logUIError("use Mica", window.UseMica())
	if icon != "" {
		logUIError("set icon", window.SetIcon(icon))
	}
	logUIError("resize", window.Resize(860, 720))
	logUIError("set minimum size", window.SetMinimumSize(520, 400))
	logUIError("center", window.Center())
	logUIError("observe closing", window.OnClosed(w.closed))
	logUIError("observe activation", window.OnActivated(w.activated))
	w.bind()
	return w, nil
}

func logUIError(action string, err error) {
	if err != nil {
		slog.Warn("settings window: failed to "+action, "error", err)
	}
}

// el returns the element named name, looking it up once.
func (w *settingsWindow) el(name string) *winui.Element {
	if e, ok := w.elements[name]; ok {
		return e
	}
	e, err := w.window.Element(name)
	if err != nil {
		slog.Error("settings window element missing", "name", name, "error", err)
	}
	w.elements[name] = e
	return e
}

func (w *settingsWindow) bind() {
	click := map[string]func(){
		"SignInButton":          w.signIn,
		"SignOutConfirmButton":  w.signOut,
		"ManageCard":            func() { openInBrowser(accountSettingsURL()) },
		"UpgradeCard":           func() { openInBrowser(accountUpgradeURL()) },
		"StartupAppsButton":     func() { openInBrowser("ms-settings:startupapps") },
		"RestartToUpdateButton": w.restartToUpdate,
		"ModelsPathButton":      w.openModelsFolder,
		"ChangeModelsButton":    w.changeModelsPath,
		"ResetModelsButton":     func() { w.setModelsPath("") },
		"ExportButton":          w.exportChats,
	}
	for name, fn := range click {
		logUIError("handle "+name, w.el(name).OnClick(fn))
	}

	toggled := map[string]func(){
		"StartupToggle": w.toggleStartup,
		"AutoUpdateToggle": func() {
			w.toggle("AutoUpdateToggle", settings.SetAutoUpdate, func(bool) string {
				return "Ollama couldn’t change automatic updates"
			})
		},
		"CloudToggle": func() {
			w.toggle("CloudToggle", settings.SetCloudEnabled, func(on bool) string {
				if on {
					return "Ollama couldn’t include cloud models"
				}
				return "Ollama couldn’t remove cloud models"
			})
		},
		"ExposeToggle": func() {
			w.toggle("ExposeToggle", settings.SetExpose, func(bool) string {
				return "Ollama couldn’t change network access"
			})
		},
	}
	for name, fn := range toggled {
		logUIError("handle "+name, w.el(name).OnToggled(fn))
	}
	logUIError("handle ContextLengthBox", w.el("ContextLengthBox").OnSelectionChanged(w.chooseContextLength))
}

func (w *settingsWindow) show() {
	logUIError("activate", w.window.Activate())
	w.reload()
}

// reload shows the current settings and checks ollama.com for the account.
func (w *settingsWindow) reload() {
	w.contextLengthChecks = 0
	w.loadSettings()
	w.loadLoginItem()
	if w.account == nil {
		// Show the last known account right away while checking ollama.com.
		run(w, func(context.Context) (accountState, error) {
			return settings.CachedAccount(), nil
		}, func(account accountState, _ error) {
			if w.account == nil {
				w.showAccount(&account)
			}
		})
	}
	w.refreshAccount()
}

func (w *settingsWindow) activated(active bool) {
	if !active {
		w.inactive = true
		return
	}
	if w.inactive {
		w.inactive = false
		w.reload()
	}
}

func (w *settingsWindow) closed() {
	w.cancel()
	for _, e := range w.elements {
		e.Release()
	}
	w.elements = nil
	if openSettings == w {
		openSettings = nil
	}
}

// run calls work on another goroutine and then done on the UI thread,
// unless the window has closed by then.
func run[T any](w *settingsWindow, work func(context.Context) (T, error), done func(T, error)) {
	ctx := w.ctx
	go func() {
		workCtx, cancel := context.WithTimeout(ctx, settingsTimeout)
		v, err := work(workCtx)
		cancel()
		winui.Post(func() {
			if ctx.Err() == nil {
				done(v, err)
			}
		})
	}()
}

// after calls fn on the UI thread after d, unless the window has closed.
func (w *settingsWindow) after(d time.Duration, fn func()) {
	ctx := w.ctx
	time.AfterFunc(d, func() {
		winui.Post(func() {
			if ctx.Err() == nil {
				fn()
			}
		})
	})
}

func (w *settingsWindow) showError(title string, err error) {
	slog.Warn(title, "error", err)
	logUIError("show error", w.el("ErrorBar").ShowInfoBar(winui.InfoBarError, title, err.Error()))
}

// Settings

func (w *settingsWindow) loadSettings() {
	run(w, func(ctx context.Context) (settingsState, error) {
		return settings.State(ctx)
	}, func(state settingsState, err error) {
		if err != nil {
			w.showError("Ollama couldn’t read its settings", err)
			return
		}
		w.showSettings(&state)
	})
}

func (w *settingsWindow) showSettings(s *settingsState) {
	w.updating = true
	defer func() { w.updating = false }()
	w.settings = s

	w.setToggle("AutoUpdateToggle", s.AutoUpdate, true)
	w.el("RestartToUpdateButton").SetVisible(s.UpdateReady)
	if s.UpdateReady {
		w.el("UpdateDescription").SetText("A new version of Ollama is ready to install.")
	} else {
		w.el("UpdateDescription").SetText(fmt.Sprintf("You’re using Ollama %s.", s.Version))
	}
	w.el("VersionText").SetText(s.Version)

	w.el("ModelsPathText").SetText(modelsPath(s))
	w.el("ModelsPathButton").SetToolTip("Open in File Explorer")
	w.el("ChangeModelsButton").SetEnabled(true)
	// Reset goes back to the default folder, so it only applies to others.
	w.el("ResetModelsButton").SetEnabled(!s.ModelsPathIsDefault)
	w.el("ResetModelsButton").SetToolTip("Use " + s.DefaultModelsPath)

	automatic := "Automatic"
	if s.DefaultContextLength > 0 {
		automatic = fmt.Sprintf("Automatic (%s)", contextLengthTitle(s.DefaultContextLength))
	}
	w.el("ContextLengthAutomatic").SetContent(automatic)
	contextLength := w.el("ContextLengthBox")
	contextLength.SetEnabled(true)
	index := slices.Index(contextLengths, s.ContextLength)
	contextLength.SetSelectedIndex(index)
	if index < 0 {
		// Keep a length chosen outside of Settings visible.
		contextLength.SetPlaceholderText(contextLengthTitle(s.ContextLength))
	}
	if s.DefaultContextLength == 0 {
		w.checkContextLengthLater()
	}

	w.setToggle("CloudToggle", s.CloudEnabled, !s.CloudDisabledByEnv)
	if s.CloudDisabledByEnv {
		w.el("CloudDescription").SetText("Turned off by the OLLAMA_NO_CLOUD environment variable.")
	} else {
		w.el("CloudDescription").SetText("Cloud models run on ollama.com. Turning this off also turns off web search, so Ollama only uses this PC.")
	}

	w.setToggle("ExposeToggle", s.Expose, true)

	// Earlier versions of the app saved chats. Offer them as files.
	w.el("ChatsSection").SetVisible(s.ChatCount > 0)
	if s.ChatCount == 1 {
		w.el("ExportDescription").SetText("Save your chat from earlier versions of Ollama as a Markdown file.")
	} else {
		w.el("ExportDescription").SetText(fmt.Sprintf("Save your %d chats from earlier versions of Ollama as Markdown files.", s.ChatCount))
	}
}

func (w *settingsWindow) setToggle(name string, on, enabled bool) {
	toggle := w.el(name)
	toggle.SetOn(on)
	toggle.SetEnabled(enabled)
}

// checkContextLengthLater reloads settings until the server, which picks the
// automatic context length when it starts, reports its choice.
func (w *settingsWindow) checkContextLengthLater() {
	if w.contextLengthChecks >= maxContextLengthChecks {
		return
	}
	w.contextLengthChecks++
	w.after(contextLengthCheckInterval, w.loadSettings)
}

// apply makes a change on another goroutine with control disabled, then
// shows the settings that result.
func (w *settingsWindow) apply(control, failure string, change func() error) {
	w.el(control).SetEnabled(false)
	// Changes that restart the server pick a new automatic context length.
	w.contextLengthChecks = 0
	run(w, func(ctx context.Context) (*settingsState, error) {
		err := change()
		state, stateErr := settings.State(ctx)
		if stateErr != nil {
			return nil, errors.Join(err, stateErr)
		}
		return &state, err
	}, func(state *settingsState, err error) {
		w.el(control).SetEnabled(true)
		if state != nil {
			w.showSettings(state)
		}
		if err != nil {
			w.showError(failure, err)
		}
	})
}

// toggle applies the state of the ToggleSwitch named name with set.
func (w *settingsWindow) toggle(name string, set func(bool) error, failure func(on bool) string) {
	if w.updating {
		return
	}
	on := w.el(name).IsOn()
	w.apply(name, failure(on), func() error { return set(on) })
}

func (w *settingsWindow) chooseContextLength() {
	if w.updating || w.settings == nil {
		return
	}
	index := w.el("ContextLengthBox").SelectedIndex()
	if index < 0 || index >= len(contextLengths) || contextLengths[index] == w.settings.ContextLength {
		return
	}
	length := contextLengths[index]
	w.apply("ContextLengthBox", "Ollama couldn’t change the context length", func() error {
		return settings.SetContextLength(length)
	})
}

func (w *settingsWindow) restartToUpdate() {
	w.el("RestartToUpdateButton").SetEnabled(false)
	go app.DoUpdate()
}

// Startup

func (w *settingsWindow) loadLoginItem() {
	run(w, func(context.Context) (loginItemStatus, error) {
		return getLoginItemStatus(), nil
	}, func(status loginItemStatus, _ error) {
		w.showLoginItem(status)
	})
}

func (w *settingsWindow) showLoginItem(status loginItemStatus) {
	w.updating = true
	defer func() { w.updating = false }()
	w.setToggle("StartupToggle", status == loginItemOn, status != loginItemUnavailable)
	w.el("StartupAppsButton").SetVisible(status == loginItemOffInSettings)
	switch status {
	case loginItemOffInSettings:
		w.el("StartupDescription").SetText("Turned off in Startup apps settings.")
	case loginItemUnavailable:
		w.el("StartupDescription").SetText("Available when Ollama is installed.")
	default:
		w.el("StartupDescription").SetText("Ollama runs in the background and appears in the notification area.")
	}
}

func (w *settingsWindow) toggleStartup() {
	if w.updating {
		return
	}
	toggle := w.el("StartupToggle")
	on := toggle.IsOn()
	toggle.SetEnabled(false)
	run(w, func(context.Context) (loginItemStatus, error) {
		err := setLoginItemEnabled(on)
		return getLoginItemStatus(), err
	}, func(status loginItemStatus, err error) {
		w.showLoginItem(status)
		if err != nil {
			w.showError("Ollama couldn’t change whether it starts when you sign in", err)
		}
	})
}

// Models

func modelsPath(s *settingsState) string {
	if s.ModelsPath != "" {
		return s.ModelsPath
	}
	return s.DefaultModelsPath
}

func (w *settingsWindow) openModelsFolder() {
	if w.settings == nil {
		return
	}
	path := modelsPath(w.settings)
	if _, err := os.Stat(path); errors.Is(err, os.ErrNotExist) {
		// Ollama creates the default folder when it first downloads a model.
		logUIError("show note", w.el("ErrorBar").ShowInfoBar(winui.InfoBarInformational,
			"The model folder doesn’t exist yet", "Ollama creates it when you download your first model."))
		return
	}
	openInExplorer(path)
}

func (w *settingsWindow) changeModelsPath() {
	start := ""
	if w.settings != nil {
		start = modelsPath(w.settings)
	}
	dir, err := w.window.PickFolder("Choose where Ollama stores models", "", start)
	if errors.Is(err, winui.ErrCanceled) {
		return
	}
	if err != nil {
		w.showError("Ollama couldn’t show folders", err)
		return
	}
	w.setModelsPath(dir)
}

func (w *settingsWindow) setModelsPath(path string) {
	w.apply("ChangeModelsButton", "Ollama couldn’t use that folder for models", func() error {
		return settings.SetModelsPath(path)
	})
}

// contextLengthTitle formats a context length in tokens, such as "32K".
func contextLengthTitle(length int) string {
	return fmt.Sprintf("%dK", length/1024)
}

// Chats

type chatExport struct {
	count int
	dir   string
}

func (w *settingsWindow) exportChats() {
	documents, _ := windows.KnownFolderPath(windows.FOLDERID_Documents, 0)
	dir, err := w.window.PickFolder("Choose where to save your chats", "", documents)
	if errors.Is(err, winui.ErrCanceled) {
		return
	}
	if err != nil {
		w.showError("Ollama couldn’t show folders", err)
		return
	}
	w.el("ExportButton").SetEnabled(false)
	w.el("ExportProgress").SetActive(true)
	run(w, func(context.Context) (chatExport, error) {
		// Each export goes in a new folder of Markdown files.
		target := availablePath(filepath.Join(dir, "Ollama Chats"))
		count, err := settings.ExportChats(target)
		return chatExport{count: count, dir: target}, err
	}, func(export chatExport, err error) {
		w.el("ExportButton").SetEnabled(true)
		w.el("ExportProgress").SetActive(false)
		if export.count > 0 {
			openInExplorer(export.dir)
		}
		if err != nil {
			title := "Ollama couldn’t export your chats"
			if export.count > 0 {
				title = "Some chats couldn’t be exported"
			}
			w.showError(title, err)
		}
	})
}

// availablePath returns path, or the first of "path (2)", "path (3)", and so
// on that doesn't exist yet, as File Explorer names copies.
func availablePath(path string) string {
	candidate := path
	for n := 2; ; n++ {
		if _, err := os.Stat(candidate); errors.Is(err, os.ErrNotExist) {
			return candidate
		}
		candidate = fmt.Sprintf("%s (%d)", path, n)
	}
}

// Account

func (w *settingsWindow) refreshAccount() {
	if w.refreshingAccount {
		return
	}
	w.refreshingAccount = true
	run(w, func(ctx context.Context) (accountState, error) {
		return settings.Account(ctx)
	}, func(account accountState, err error) {
		w.refreshingAccount = false
		w.accountNote = ""
		if err != nil && account.Cached {
			w.accountNote = "Couldn’t reach ollama.com. Showing your account from the last check."
		}
		w.showAccount(&account)
		if account.SignedIn {
			w.stopWaitingForSignIn()
		}
	})
}

func (w *settingsWindow) showAccount(account *accountState) {
	w.account = account
	waiting := !w.signInStarted.IsZero()
	signedIn := account != nil && account.SignedIn

	var name, detail string
	switch {
	case signedIn:
		name = account.Name
		var details []string
		if account.Email != "" {
			details = append(details, account.Email)
		}
		if account.Plan != "" {
			details = append(details, upperFirst(account.Plan)+" plan")
		}
		detail = strings.Join(details, " · ")
	case account != nil:
		name, detail = "Not signed in", "Sign in to use cloud models and web search."
		if waiting {
			detail = "Finish signing in with your browser."
		}
	}
	if w.accountNote != "" {
		detail = strings.TrimSpace(detail + "\n" + w.accountNote)
	}
	w.el("AccountName").SetText(name)
	w.el("AccountDetail").SetText(detail)

	person, picture := "", ""
	if signedIn {
		person, picture = account.Name, account.AvatarURL
		if picture == "" {
			// Accounts from the last check have no picture; keep the one shown.
			picture = w.shownPicture
		}
	}
	if person != w.shownPerson || picture != w.shownPicture {
		w.shownPerson, w.shownPicture = person, picture
		logUIError("show account picture", w.el("AccountPicture").SetPerson(person, picture))
	}

	w.el("AccountProgress").SetActive(waiting && !signedIn)
	w.el("SignInButton").SetVisible(account != nil && !signedIn)
	w.el("SignInButton").SetEnabled(!waiting)
	w.el("SignOutButton").SetVisible(signedIn)
	w.el("ManageCard").SetVisible(signedIn)
	w.el("UpgradeCard").SetVisible(signedIn && strings.EqualFold(account.Plan, "free"))
}

func (w *settingsWindow) signIn() {
	run(w, func(context.Context) (string, error) {
		return settings.SignInURL()
	}, func(signInURL string, err error) {
		if err != nil {
			w.showError("Ollama couldn’t start signing in", err)
			return
		}
		openInBrowser(signInURL)
		w.signInStarted = time.Now()
		w.showAccount(w.account)
		w.pollSignIn(w.signInStarted)
	})
}

// pollSignIn checks for the account people sign in to with their browser.
func (w *settingsWindow) pollSignIn(started time.Time) {
	w.after(signInPollInterval, func() {
		if w.signInStarted != started {
			return
		}
		if time.Since(started) > signInTimeout {
			w.stopWaitingForSignIn()
			return
		}
		w.refreshAccount()
		w.pollSignIn(started)
	})
}

func (w *settingsWindow) stopWaitingForSignIn() {
	if w.signInStarted.IsZero() {
		return
	}
	w.signInStarted = time.Time{}
	w.showAccount(w.account)
}

func (w *settingsWindow) signOut() {
	w.el("SignOutFlyout").HideFlyout()
	w.el("SignOutButton").SetEnabled(false)
	run(w, func(ctx context.Context) (accountState, error) {
		if err := settings.SignOut(ctx); err != nil {
			return settings.CachedAccount(), err
		}
		return settings.Account(ctx)
	}, func(account accountState, err error) {
		w.el("SignOutButton").SetEnabled(true)
		w.showAccount(&account)
		if err != nil {
			w.showError("Ollama couldn’t sign out", err)
		}
	})
}

// Page and files

// settingsPage returns the window's markup, with the app icon at icon.
func settingsPage(icon string) (string, error) {
	type contextLengthItem struct {
		Title     string
		Automatic bool
	}
	data := struct {
		Icon           string
		ContextLengths []contextLengthItem
	}{}
	if icon != "" {
		data.Icon = html.EscapeString((&url.URL{Scheme: "file", Path: "/" + filepath.ToSlash(icon)}).String())
	}
	for _, length := range contextLengths {
		if length == 0 {
			data.ContextLengths = append(data.ContextLengths, contextLengthItem{Title: "Automatic", Automatic: true})
		} else {
			data.ContextLengths = append(data.ContextLengths, contextLengthItem{Title: contextLengthTitle(length)})
		}
	}
	var page strings.Builder
	if err := settingsPageTemplate.Execute(&page, data); err != nil {
		return "", err
	}
	return page.String(), nil
}

// settingsIconPath returns the app icon file, which the installer puts next
// to the app. Development builds write the embedded copy.
func settingsIconPath() string {
	installed := filepath.Join(appPath, "app.ico")
	if _, err := os.Stat(installed); err == nil {
		return installed
	}
	data, err := assets.GetIcon("app.ico")
	if err != nil {
		return ""
	}
	path := filepath.Join(os.TempDir(), "ollama_app.ico")
	if err := os.WriteFile(path, data, 0o644); err != nil {
		return ""
	}
	return path
}

func openInExplorer(path string) {
	if err := exec.Command("explorer.exe", path).Start(); err != nil {
		slog.Error("failed to open folder", "path", path, "error", err)
	}
}
