package tui

import (
	"os"
	"sync"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/ollama/ollama/cmd/launch"
	"golang.org/x/term"
)

// spinnerStyle dims the spinner so it reads as ancillary status text, matching
// the sign-in/upgrade spinners in signin.go.
var spinnerStyle = lipgloss.NewStyle().
	Foreground(lipgloss.AdaptiveColor{Light: "242", Dark: "246"})

type spinnerTickMsg struct{}

// spinnerQuitMsg is sent by Stop to ask the program to quit cleanly.
type spinnerQuitMsg struct{}

type spinnerModel struct {
	message   string
	frame     int
	quitting  bool
	cancelled chan struct{}
	once      sync.Once
}

func (m *spinnerModel) Init() tea.Cmd {
	return tea.Tick(100*time.Millisecond, func(time.Time) tea.Msg { return spinnerTickMsg{} })
}

func (m *spinnerModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case spinnerTickMsg:
		if m.quitting {
			return m, nil
		}
		m.frame++
		return m, tea.Tick(100*time.Millisecond, func(time.Time) tea.Msg { return spinnerTickMsg{} })
	case tea.KeyMsg:
		// bubbletea runs the terminal in raw mode, so Ctrl+C is delivered here
		// as a key rather than as a SIGINT. Treat it as a user cancellation:
		// close the cancelled channel (so the caller's wait loop can abort) and
		// quit the program so bubbletea restores the terminal before control
		// returns to the caller.
		if msg.String() == "ctrl+c" {
			m.once.Do(func() { close(m.cancelled) })
			m.quitting = true
			return m, tea.Quit
		}
	case spinnerQuitMsg:
		m.quitting = true
		// Returning "" from View on quit clears the spinner line, mirroring how
		// confirm.go blanks its view when it quits.
		return m, tea.Quit
	}
	return m, nil
}

func (m *spinnerModel) View() string {
	if m.quitting {
		return ""
	}
	frame := launch.SpinnerFrames[m.frame%len(launch.SpinnerFrames)]
	return spinnerStyle.Render(frame + " " + m.message)
}

// RunSpinner runs a bubbletea spinner displaying message until the returned
// Spinner's Stop is called. Stop signals the program to quit and blocks until
// it has exited and cleared its line. If the user presses Ctrl+C while the
// spinner is running, Spinner.Cancelled() is closed so the caller can abort
// its wait; the program quits and the terminal is restored before Stop
// returns. RunSpinner returns nil when there is no interactive terminal, so
// launch.StartSpinner can fall back to its ANSI spinner for headless/--yes
// runs.
func RunSpinner(message string) *launch.Spinner {
	if !term.IsTerminal(int(os.Stdin.Fd())) || !term.IsTerminal(int(os.Stderr.Fd())) {
		return nil
	}

	cancelled := make(chan struct{})
	m := &spinnerModel{message: message, cancelled: cancelled}
	p := tea.NewProgram(m, tea.WithOutput(os.Stderr))
	done := make(chan struct{})
	go func() {
		_, _ = p.Run()
		close(done)
	}()

	var once sync.Once
	stop := func() {
		once.Do(func() {
			select {
			case <-done:
				// Program already finished (e.g. the user cancelled), so don't
				// send to it; just ensure it has exited.
				return
			default:
			}
			p.Send(spinnerQuitMsg{})
			<-done
		})
	}

	return launch.NewSpinner(stop, cancelled)
}
