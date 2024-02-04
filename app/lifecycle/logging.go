package lifecycle

import (
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
)

func InitLogging(filename string) {
	// level := slog.LevelInfo

	// TODO once things are solid, wire this up to quiet logs down by default
	// if debug := os.Getenv("OLLAMA_DEBUG"); debug != "" {
	level := slog.LevelDebug
	// }

	var logFile *os.File
	// Detect if we're a GUI app on windows, and if not, send logs to console
	if os.Stderr.Fd() != 0 {
		// Console app detected
		logFile = os.Stderr
		// TODO - write one-line to the app.log file saying we're running in console mode to help avoid confusion
	} else {
		_, err := os.Stat(AppDataDir)
		if errors.Is(err, os.ErrNotExist) {
			if err := os.MkdirAll(AppDataDir, 0o755); err != nil {
				slog.Error(fmt.Sprintf("create ollama dir %s: %v", AppDataDir, err))
				return
			}
		}
		logFilename := filepath.Join(AppDataDir, filename)
		logFile, err = os.OpenFile(logFilename, os.O_APPEND|os.O_WRONLY|os.O_CREATE, 0755)
		if err != nil {
			slog.Error(fmt.Sprintf("failed to create server log %v", err))
			return
		}
	}
	handler := slog.NewTextHandler(logFile, &slog.HandlerOptions{
		Level:     level,
		AddSource: true,
		ReplaceAttr: func(_ []string, attr slog.Attr) slog.Attr {
			if attr.Key == slog.SourceKey {
				source := attr.Value.Any().(*slog.Source)
				source.File = filepath.Base(source.File)
			}
			return attr
		},
	})

	slog.SetDefault(slog.New(handler))

	slog.Info("ollama app started")
}
