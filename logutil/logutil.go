package logutil

import (
	"context"
	"io"
	"log/slog"
	"path/filepath"
	"runtime"
	"time"
)

const LevelTrace slog.Level = -8

func NewLogger(w io.Writer, level slog.Level) *slog.Logger {
	return slog.New(slog.NewTextHandler(w, &slog.HandlerOptions{
		Level:     level,
		AddSource: true,
		ReplaceAttr: func(_ []string, attr slog.Attr) slog.Attr {
			switch attr.Key {
			case slog.LevelKey:
				switch attr.Value.Any().(slog.Level) {
				case LevelTrace:
					attr.Value = slog.StringValue("TRACE")
				}
			case slog.SourceKey:
				source := attr.Value.Any().(*slog.Source)
				source.File = filepath.Base(source.File)
			}
			return attr
		},
	}))
}

type key string

func Trace(msg string, args ...any) {
	TraceContext(context.WithValue(context.TODO(), key("skip"), 1), msg, args...)
}

func TraceContext(ctx context.Context, msg string, args ...any) {
	if logger := slog.Default(); logger.Enabled(ctx, LevelTrace) {
		skip, _ := ctx.Value(key("skip")).(int)
		pc, _, _, _ := runtime.Caller(1 + skip)
		record := slog.NewRecord(time.Now(), LevelTrace, msg, pc)
		record.Add(args...)
		logger.Handler().Handle(ctx, record)
	}
}
