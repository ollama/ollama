package logging

import (
	"context"
	"log/slog"
	"os"
)

const LevelTrace slog.Level = slog.LevelDebug - 4

type Logger struct {
	logger *slog.Logger
}

func NewLogger() *Logger {
	handler := slog.NewTextHandler(os.Stdout, nil)
	return &Logger{
		logger: slog.New(handler),
	}
}

func (l *Logger) Trace(msg string, args ...any) {
	l.logger.Log(context.Background(), LevelTrace, msg, args...)
}

func (l *Logger) Debug(msg string, args ...any) {
	l.logger.Debug(msg, args...)
}

func (l *Logger) Info(msg string, args ...any) {
	l.logger.Info(msg, args...)
}

func (l *Logger) Warn(msg string, args ...any) {
	l.logger.Warn(msg, args...)
}

func (l *Logger) Error(msg string, args ...any) {
	l.logger.Error(msg, args...)
}
