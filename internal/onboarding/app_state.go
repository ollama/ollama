package onboarding

import (
	"context"
	"database/sql"
	"log/slog"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	_ "github.com/mattn/go-sqlite3"
)

// CompletedInApp checks the existing desktop database without starting the app
// or creating, migrating, or updating its database. An unavailable database is
// not proof of completion and must not prevent the CLI from offering welcome.
func CompletedInApp() bool {
	return readAppCompletion(AppDatabasePath())
}

// AppDatabasePath is the production desktop database location shared by the
// app store and the CLI's read-only lookup. Empty means no desktop database.
func AppDatabasePath() string {
	home, _ := os.UserHomeDir()
	localAppData := os.Getenv("LOCALAPPDATA")
	switch {
	case runtime.GOOS == "darwin" && home != "":
		return filepath.Join(home, "Library", "Application Support", "Ollama", "db.sqlite")
	case runtime.GOOS == "windows" && localAppData != "":
		return filepath.Join(localAppData, "Ollama", "db.sqlite")
	default:
		return ""
	}
}

func readAppCompletion(path string) bool {
	if path == "" {
		return false
	}
	info, err := os.Stat(path)
	if err != nil || !info.Mode().IsRegular() {
		return false
	}

	uriPath := filepath.ToSlash(path)
	if !strings.HasPrefix(uriPath, "/") {
		uriPath = "/" + uriPath // Windows drive-letter paths need file:///C:/...
	}
	u := url.URL{Scheme: "file", Path: uriPath, RawQuery: "mode=ro&_query_only=1&_busy_timeout=200"}
	db, err := sql.Open("sqlite3", u.String())
	if err != nil {
		return false
	}
	defer db.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 250*time.Millisecond)
	defer cancel()
	var version int
	if err := db.QueryRowContext(ctx, "SELECT onboarding_version FROM settings WHERE id = 1").Scan(&version); err != nil {
		// Migration 16 -> 17 marks existing users completed at onboarding version 1.
		var schema int
		if err := db.QueryRowContext(ctx, "SELECT schema_version FROM settings WHERE id = 1").Scan(&schema); err == nil && schema >= 1 && schema <= 16 {
			return CurrentVersion <= 1
		}
		slog.Debug("could not read app onboarding completion", "error", err)
		return false
	}
	return version >= CurrentVersion
}
