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

	"github.com/ollama/ollama/api"
)

func getCLIFullPath(command string) string {
	var cmdPath string
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
	pwd, err := os.Getwd()
	if err == nil {
		cmdPath = filepath.Join(pwd, command)
		_, err = os.Stat(cmdPath)
		if err == nil {
			return cmdPath
		}
	}

	return command
}

func start(ctx context.Context, command string) (*exec.Cmd, error) {
	cmd := getCmd(ctx, getCLIFullPath(command))
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("failed to spawn server stdout pipe: %w", err)
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return nil, fmt.Errorf("failed to spawn server stderr pipe: %w", err)
	}

	rotateLogs(ServerLogFile)
	logFile, err := os.OpenFile(ServerLogFile, os.O_APPEND|os.O_WRONLY|os.O_CREATE, 0o755)
	if err != nil {
		return nil, fmt.Errorf("failed to create server log: %w", err)
	}

	logDir := filepath.Dir(ServerLogFile)
	_, err = os.Stat(logDir)
	if err != nil {
		if !errors.Is(err, os.ErrNotExist) {
			return nil, fmt.Errorf("stat ollama server log dir %s: %v", logDir, err)
		}

		if err := os.MkdirAll(logDir, 0o755); err != nil {
			return nil, fmt.Errorf("create ollama server log dir %s: %v", logDir, err)
		}
	}

	go func() {
		defer logFile.Close()
		io.Copy(logFile, stdout) //nolint:errcheck
	}()
	go func() {
		defer logFile.Close()
		io.Copy(logFile, stderr) //nolint:errcheck
	}()

	// Re-wire context done behavior to attempt a graceful shutdown of the server
	cmd.Cancel = func() error {
		if cmd.Process != nil {
			err := terminate(cmd)
			if err != nil {
				slog.Warn("error trying to gracefully terminate server", "err", err)
				return cmd.Process.Kill()
			}

			tick := time.NewTicker(10 * time.Millisecond)
			defer tick.Stop()

			for {
				select {
				case <-tick.C:
					exited, err := isProcessExited(cmd.Process.Pid)
					if err != nil {
						return err
					}

					if exited {
						return nil
					}
				case <-time.After(5 * time.Second):
					slog.Warn("graceful server shutdown timeout, killing", "pid", cmd.Process.Pid)
					return cmd.Process.Kill()
				}
			}
		}
		return nil
	}

	// run the command and wait for it to finish
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("failed to start server %w", err)
	}
	if cmd.Process != nil {
		slog.Info(fmt.Sprintf("started ollama server with pid %d", cmd.Process.Pid))
	}
	slog.Info(fmt.Sprintf("ollama server logs %s", ServerLogFile))

	return cmd, nil
}

func SpawnServer(ctx context.Context, command string) (chan int, error) {
	done := make(chan int)

	go func() {
		// Keep the server running unless we're shuttind down the app
		crashCount := 0
		for {
			slog.Info("starting server...")
			cmd, err := start(ctx, command)
			if err != nil {
				crashCount++
				slog.Error(fmt.Sprintf("failed to start server %s", err))
				time.Sleep(500 * time.Millisecond * time.Duration(crashCount))
				continue
			}

			cmd.Wait() //nolint:errcheck
			var code int
			if cmd.ProcessState != nil {
				code = cmd.ProcessState.ExitCode()
			}

			select {
			case <-ctx.Done():
				slog.Info(fmt.Sprintf("server shutdown with exit code %d", code))
				done <- code
				return
			default:
				crashCount++
				slog.Warn(fmt.Sprintf("server crash %d - exit code %d - respawning", crashCount, code))
				time.Sleep(500 * time.Millisecond * time.Duration(crashCount))
				break
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
