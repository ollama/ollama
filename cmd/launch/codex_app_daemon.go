package launch

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

func defaultCodexAppStopManagedDaemon() error {
	// This phase starts after the confirmed GUI exit. Codex's own graceful
	// shutdown can take longer than the GUI quit deadline (up to five minutes).
	ctx, cancel := context.WithTimeout(context.Background(), 6*time.Minute)
	defer cancel()
	return stopCodexAppManagedDaemon(ctx)
}

func stopCodexAppManagedDaemon(ctx context.Context) error {
	executable, err := codexAppCodexExecutable()
	if errors.Is(err, exec.ErrNotFound) {
		// Older desktop clients can run an embedded server without this CLI.
		return nil
	}
	if err != nil {
		return fmt.Errorf("find Codex background server command: %w", err)
	}
	configPath, err := codexConfigPath()
	if err != nil {
		return err
	}
	cmd := exec.CommandContext(ctx, executable, "app-server", "daemon", "stop")
	// Stop the daemon for the profile being edited, even when Ollama inherited
	// a different CODEX_HOME. Keep the rest of the user's environment intact.
	for _, value := range os.Environ() {
		name, _, _ := strings.Cut(value, "=")
		if name != "CODEX_HOME" {
			cmd.Env = append(cmd.Env, value)
		}
	}
	cmd.Env = append(cmd.Env, "CODEX_HOME="+filepath.Dir(configPath))
	cmd.Dir = os.TempDir()
	output, err := cmd.Output()
	if err != nil {
		if ctx.Err() != nil {
			return fmt.Errorf("stop Codex background server: %w", ctx.Err())
		}
		var exitErr *exec.ExitError
		if errors.As(err, &exitErr) && exitErr.ExitCode() == 2 {
			stderr := string(exitErr.Stderr)
			if strings.Contains(stderr, "unrecognized subcommand 'daemon'") ||
				strings.Contains(stderr, "unexpected argument 'daemon'") {
				// These clients predate managed daemons; quitting their GUI is enough.
				return nil
			}
		}
		return fmt.Errorf("stop Codex background server: %w", err)
	}
	var result struct {
		Status string `json:"status"`
	}
	if err := json.Unmarshal(output, &result); err != nil {
		return fmt.Errorf("read Codex background server stop result: %w", err)
	}
	if result.Status != "stopped" && result.Status != "notRunning" {
		return errors.New("Codex did not confirm that its background server stopped")
	}
	return nil
}
