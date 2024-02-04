package lifecycle

import (
	"context"
	"fmt"
	"log"
	"log/slog"
	"runtime"

	"github.com/jmorganca/ollama/app/store"
	"github.com/jmorganca/ollama/app/tray"
)

func GetOllamaName() string {
	if runtime.GOOS == "windows" {
		return "ollama.exe"
	} else {
		return "ollama"
	}
}

func Run() {
	InitLogging("app.log")

	ctx, cancel := context.WithCancel(context.Background())

	t, err := tray.NewTray(DoUpgrade)
	if err != nil {
		log.Fatalf("Failed to start: %s", err)
	}
	callbacks := t.GetCallbacks()

	go func() {
		slog.Debug("XXX starting callback handler")
		for {
			select {
			case <-callbacks.Quit:
				slog.Debug("QUIT called")
				t.Quit()
			case <-callbacks.Update:
				slog.Debug("XXX about to call DoUpgrade")
				err := DoUpgrade()
				if err != nil {
					slog.Debug(fmt.Sprintf("DoUpgrade FAILED: %s", err))
				}
			case <-callbacks.ShowLogs:
				ShowLogs()
			case <-callbacks.DoFirstUse:
				slog.Debug("Spawning getting started shell terminal")
				err := GetStarted()
				if err != nil {
					slog.Warn(fmt.Sprintf("Failed to launch getting started shell: %s", err))
				}
			}
		}
	}()

	// Are we first use?
	if !store.GetFirstTimeRun() {
		slog.Debug("First time run")
		err = t.DisplayFirstUseNotification()
		if err != nil {
			slog.Debug(fmt.Sprintf("XXX failed to display first use notification %v", err))
		}
		store.SetFirstTimeRun(true)
	} else {
		slog.Debug("Not first time, skipping first run notification")
	}

	var done chan int
	if IsServerRunning(ctx) {
		slog.Debug("XXX detected server already running")
		// TODO - should we fail fast, try to kill it, or just ignore?
	} else {
		slog.Debug("XXX spawning server")
		done, err = SpawnServer(ctx, CLIName)
		if err != nil {
			// TODO - should we retry in a backoff loop?
			// TODO - should we pop up a warning and maybe add a menu item to view application logs?
			slog.Error(fmt.Sprintf("Failed to spawn ollama server %s", err))
			done = make(chan int, 1)
			done <- 1
		}
	}

	StartBackgroundUpdaterChecker(ctx, t.UpdateAvailable)

	t.Run()
	cancel()
	slog.Info("Waiting for ollama server to shutdown...")
	if done != nil {
		<-done
	}
	slog.Info("Ollama app exiting")
}
