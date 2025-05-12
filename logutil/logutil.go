package logutil

import (
	"io"
	"log/slog"
	"path/filepath"
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
