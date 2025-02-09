package lifecycle

import (
    "fmt"
    "log/slog"
    "os"
    "path/filepath"
    "strconv"
    "strings"

    "github.com/ollama/ollama/envconfig"
)

var (
    envconfigDebug = envconfig.Debug
    osOpenFile     = os.OpenFile
    osStderrFd     = func() uintptr { return os.Stderr.Fd() }
    osRemove       = os.Remove
    osRename       = os.Rename
)

func InitLogging() {
    level := slog.LevelInfo

    if envconfigDebug() {
        level = slog.LevelDebug
    }

    var logFile *os.File
    var err error
    // Detect if we're a GUI app on windows, and if not, send logs to console
    if osStderrFd() != 0 {
        // Console app detected
        logFile = os.Stderr
        // TODO - write one-line to the app.log file saying we're running in console mode to help avoid confusion
    } else {
        rotateLogs(AppLogFile)
        logFile, err = osOpenFile(AppLogFile, os.O_APPEND|os.O_WRONLY|os.O_CREATE, 0o755)
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

func rotateLogs(logFile string) {
    if _, err := os.Stat(logFile); os.IsNotExist(err) {
        return
    }
    index := strings.LastIndex(logFile, ".")
    pre := logFile[:index]
    post := "." + logFile[index+1:]
    for i := LogRotationCount; i > 0; i-- {
        older := pre + "-" + strconv.Itoa(i) + post
        newer := pre + "-" + strconv.Itoa(i-1) + post
        if i == 1 {
            newer = pre + post
        }
        if _, err := os.Stat(newer); err == nil {
            if _, err := os.Stat(older); err == nil {
                err := osRemove(older)
                if err != nil {
                    slog.Warn("Failed to remove older log", "older", older, "error", err)
                    continue
                }
            }
            err := osRename(newer, older)
            if err != nil {
                slog.Warn("Failed to rotate log", "older", older, "newer", newer, "error", err)
            }
        }
    }
}
