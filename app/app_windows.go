package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"log/slog"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"

	"github.com/jmorganca/ollama/app/lifecycle"
	"github.com/jmorganca/ollama/app/store"
	"github.com/jmorganca/ollama/app/tray"
	"github.com/jmorganca/ollama/app/updater"
)

func init() {
	AppName += ".exe"
	CLIName += ".exe"
	// Logs, configs, downloads go to LOCALAPPDATA
	localAppData := os.Getenv("LOCALAPPDATA")
	AppDataDir = filepath.Join(localAppData, "Ollama")
	AppLogFile = filepath.Join(AppDataDir, "app.log")
	ServerLogFile = filepath.Join(AppDataDir, "server.log")

	// Executables are stored in APPDATA
	AppDir = filepath.Join(localAppData, "Programs", "Ollama")

	// Make sure we have PATH set correctly for any spawned children
	paths := strings.Split(os.Getenv("PATH"), ";")
	// Start with whatever we find in the PATH/LD_LIBRARY_PATH
	found := false
	for _, path := range paths {
		d, err := filepath.Abs(path)
		if err != nil {
			continue
		}
		if strings.EqualFold(AppDir, d) {
			found = true
		}
	}
	if !found {
		paths = append(paths, AppDir)

		pathVal := strings.Join(paths, ";")
		slog.Debug("setting PATH=" + pathVal)
		err := os.Setenv("PATH", pathVal)
		if err != nil {
			slog.Error(fmt.Sprintf("failed to update PATH: %s", err))
		}
	}

	// Make sure our logging dir exists
	_, err := os.Stat(AppDataDir)
	if errors.Is(err, os.ErrNotExist) {
		if err := os.MkdirAll(AppDataDir, 0o755); err != nil {
			slog.Error(fmt.Sprintf("create ollama dir %s: %v", AppDataDir, err))
		}
	}
}

func ShowLogs() {
	cmd_path := "c:\\Windows\\system32\\cmd.exe"
	slog.Debug(fmt.Sprintf("viewing logs with start %s", AppDataDir))
	cmd := exec.Command(cmd_path, "/c", "start", AppDataDir)
	cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: false, CreationFlags: 0x08000000}
	err := cmd.Start()
	if err != nil {
		slog.Error(fmt.Sprintf("Failed to open log dir: %s", err))
	}
}

func Start() {
	cmd_path := "c:\\Windows\\system32\\cmd.exe"
	slog.Debug(fmt.Sprintf("viewing logs with start %s", AppDataDir))
	cmd := exec.Command(cmd_path, "/c", "start", AppDataDir)
	cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: false, CreationFlags: 0x08000000}
	err := cmd.Start()
	if err != nil {
		slog.Error(fmt.Sprintf("Failed to open log dir: %s", err))
	}
}

func run() {
	initLogging()

	slog.Info("ollama windows app started")

	ctx, cancel := context.WithCancel(context.Background())
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
				err := updater.DoUpgrade(cancel, done)
				if err != nil {
					slog.Warn(fmt.Sprintf("upgrade attempt failed: %s", err))
				}
			case <-callbacks.ShowLogs:
				ShowLogs()
			case <-callbacks.DoFirstUse:
				err := lifecycle.GetStarted()
				if err != nil {
					slog.Warn(fmt.Sprintf("Failed to launch getting started shell: %s", err))
				}
			}
		}
	}()

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

	if isServerRunning(ctx) {
		slog.Info("Detected another instance of ollama running, exiting")
		os.Exit(1)
	}

	done, err = SpawnServer(ctx, CLIName)
	if err != nil {
		// TODO - should we retry in a backoff loop?
		// TODO - should we pop up a warning and maybe add a menu item to view application logs?
		slog.Error(fmt.Sprintf("Failed to spawn ollama server %s", err))
		done = make(chan int, 1)
		done <- 1
	}

	updater.StartBackgroundUpdaterChecker(ctx, t.UpdateAvailable)

	t.Run()
	cancel()
	slog.Info("Waiting for ollama server to shutdown...")
	if done != nil {
		<-done
	}
	slog.Info("Ollama app exiting")
}
