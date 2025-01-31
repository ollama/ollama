package lifecycle

import (
	"context"
	"fmt"
	"log"
	"log/slog"
	"os"
	"os/signal"
	"syscall"

	"github.com/ollama/ollama/app/tray"
	"github.com/ollama/ollama/envconfig"
)

func Run() {
	InitLogging()
	slog.Info("app config", "env", envconfig.Values())

	_, cancel := context.WithCancel(context.Background())
	var done chan int

	t, err := tray.NewTray()
	if err != nil {
		log.Fatalf("Failed to start: %s", err)
	}
	callbacks := t.GetCallbacks()

	signals := make(chan os.Signal, 1)
	signal.Notify(signals, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		slog.Debug("starting callback loop")
		for {
			select {
			case <-callbacks.Quit:
				slog.Debug("quit called")
				t.Quit()
			case <-signals:
				slog.Debug("shutting down due to signal")
				t.Quit()
			case <-callbacks.Update:
				err := DoUpgrade(cancel, done)
				if err != nil {
					slog.Warn(fmt.Sprintf("upgrade attempt failed: %s", err))
				}
			case <-callbacks.ShowLogs:
				ShowLogs()
			case <-callbacks.DoFirstUse:
				err := GetStarted()
				if err != nil {
					slog.Warn(fmt.Sprintf("Failed to launch getting started shell: %s", err))
				}
			}
		}
	}()

	// Are we first use?

	t.Run()
	cancel()
	slog.Info("Waiting for ollama server to shutdown...")
	if done != nil {
		<-done
	}
	slog.Info("Ollama app exiting")
}
