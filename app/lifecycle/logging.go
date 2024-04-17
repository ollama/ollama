package lifecycle

import (
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
)

func InitLogging() {
	level := slog.LevelInfo

	if debug := os.Getenv("OLLAMA_DEBUG"); debug != "" {
		level = slog.LevelDebug
	}

	var logFile *os.File
	var err error
	// Detect if we're a GUI app on windows, and if not, send logs to console
	if os.Stderr.Fd() != 0 {
		// Console app detected
		logFile = os.Stderr
		// TODO - write one-line to the app.log file saying we're running in console mode to help avoid confusion
	} else {
		logFile, err = os.OpenFile(AppLogFile, os.O_APPEND|os.O_WRONLY|os.O_CREATE, 0755)
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
