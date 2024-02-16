package lifecycle

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"time"

	"github.com/jmorganca/ollama/api"
)

func getCLIFullPath(command string) string {
	cmdPath := ""
	appExe, err := os.Executable()
	if err == nil {
		cmdPath = filepath.Join(filepath.Dir(appExe), command)
		_, err := os.Stat(cmdPath)
		if err == nil {
			return cmdPath
		}
	}
	cmdPath, err = exec.LookPath(command)
	if err == nil {
		_, err := os.Stat(cmdPath)
		if err == nil {
			return cmdPath
		}
	}
	cmdPath = filepath.Join(".", command)
	_, err = os.Stat(cmdPath)
	if err == nil {
		return cmdPath
	}
	return command
}

func SpawnServer(ctx context.Context, command string) (chan int, error) {
	done := make(chan int)

	logDir := filepath.Dir(ServerLogFile)
	_, err := os.Stat(logDir)
	if errors.Is(err, os.ErrNotExist) {
		if err := os.MkdirAll(logDir, 0o755); err != nil {
			return done, fmt.Errorf("create ollama server log dir %s: %v", logDir, err)
		}
	}

	cmd := getCmd(ctx, getCLIFullPath(command))
	// send stdout and stderr to a file
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return done, fmt.Errorf("failed to spawn server stdout pipe %s", err)
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return done, fmt.Errorf("failed to spawn server stderr pipe %s", err)
	}
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return done, fmt.Errorf("failed to spawn server stdin pipe %s", err)
	}

	// TODO - rotation
	logFile, err := os.OpenFile(ServerLogFile, os.O_APPEND|os.O_WRONLY|os.O_CREATE, 0755)
	if err != nil {
		return done, fmt.Errorf("failed to create server log %w", err)
	}
	go func() {
		defer logFile.Close()
		io.Copy(logFile, stdout) //nolint:errcheck
	}()
	go func() {
		defer logFile.Close()
		io.Copy(logFile, stderr) //nolint:errcheck
	}()

	// run the command and wait for it to finish
	if err := cmd.Start(); err != nil {
		return done, fmt.Errorf("failed to start server %w", err)
	}
	if cmd.Process != nil {
		slog.Info(fmt.Sprintf("started ollama server with pid %d", cmd.Process.Pid))
	}
	slog.Info(fmt.Sprintf("ollama server logs %s", ServerLogFile))

	go func() {
		// Keep the server running unless we're shuttind down the app
		crashCount := 0
		for {
			cmd.Wait() //nolint:errcheck
			stdin.Close()
			var code int
			if cmd.ProcessState != nil {
				code = cmd.ProcessState.ExitCode()
			}

			select {
			case <-ctx.Done():
				slog.Debug(fmt.Sprintf("server shutdown with exit code %d", code))
				done <- code
				return
			default:
				crashCount++
				slog.Warn(fmt.Sprintf("server crash %d - exit code %d - respawning", crashCount, code))
				time.Sleep(500 * time.Millisecond)
				if err := cmd.Start(); err != nil {
					slog.Error(fmt.Sprintf("failed to restart server %s", err))
					// Keep trying, but back off if we keep failing
					time.Sleep(time.Duration(crashCount) * time.Second)
				}
			}
		}
	}()
	return done, nil
}

func IsServerRunning(ctx context.Context) bool {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		slog.Info("unable to connect to server")
		return false
	}
	err = client.Heartbeat(ctx)
	if err != nil {
		slog.Debug(fmt.Sprintf("heartbeat from server: %s", err))
		slog.Info("unable to connect to server")
		return false
	}
	return true
}
