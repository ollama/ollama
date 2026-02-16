//go:build windows || darwin

package server

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/ollama/ollama/app/logrotate"
	"github.com/ollama/ollama/app/store"
)

const restartDelay = time.Second

// Server is a managed ollama server process
type Server struct {
	store *store.Store
	bin   string // resolved path to `ollama`
	log   io.WriteCloser
	dev   bool // true if running with the dev flag
}

type InferenceCompute struct {
	Library string
	Variant string
	Compute string
	Driver  string
	Name    string
	VRAM    string
}

func New(s *store.Store, devMode bool) *Server {
	p := resolvePath("ollama")
	return &Server{store: s, bin: p, dev: devMode}
}

func resolvePath(name string) string {
	// look in the app bundle first
	if exe, _ := os.Executable(); exe != "" {
		var dir string
		if runtime.GOOS == "windows" {
			dir = filepath.Dir(exe)
		} else {
			dir = filepath.Join(filepath.Dir(exe), "..", "Resources")
		}
		if _, err := os.Stat(filepath.Join(dir, name)); err == nil {
			return filepath.Join(dir, name)
		}
	}

	// check the development dist path
	for _, path := range []string{
		filepath.Join("dist", runtime.GOOS, name),
		filepath.Join("dist", runtime.GOOS+"-"+runtime.GOARCH, name),
	} {
		if _, err := os.Stat(path); err == nil {
			return path
		}
	}

	// fallback to system path
	if p, _ := exec.LookPath(name); p != "" {
		return p
	}

	return name
}

// cleanup checks the pid file for a running ollama process
// and shuts it down gracefully if it is running
func cleanup() error {
	data, err := os.ReadFile(pidFile)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	defer os.Remove(pidFile)

	pid, err := strconv.Atoi(strings.TrimSpace(string(data)))
	if err != nil {
		return err
	}

	proc, err := os.FindProcess(pid)
	if err != nil {
		return nil
	}

	ok, err := terminated(pid)
	if err != nil {
		slog.Debug("cleanup: error checking if terminated", "pid", pid, "err", err)
	}
	if ok {
		return nil
	}

	slog.Info("detected previous ollama process, cleaning up", "pid", pid)
	return stop(proc)
}

// stop waits for a process with the provided pid to exit by polling
// `terminated(pid)`. If the process has not exited within 5 seconds, it logs a
// warning and kills the process.
func stop(proc *os.Process) error {
	if proc == nil {
		return nil
	}

	if err := terminate(proc); err != nil {
		slog.Warn("graceful terminate failed, killing", "err", err)
		return proc.Kill()
	}

	deadline := time.NewTimer(5 * time.Second)
	defer deadline.Stop()

	for {
		select {
		case <-deadline.C:
			slog.Warn("timeout waiting for graceful shutdown; killing", "pid", proc.Pid)
			return proc.Kill()
		default:
			ok, err := terminated(proc.Pid)
			if err != nil {
				slog.Error("error checking if ollama process is terminated", "err", err)
				return err
			}
			if ok {
				return nil
			}
			time.Sleep(10 * time.Millisecond)
		}
	}
}

func (s *Server) Run(ctx context.Context) error {
	l, err := openRotatingLog()
	if err != nil {
		return err
	}
	s.log = l
	defer s.log.Close()

	if err := cleanup(); err != nil {
		slog.Warn("failed to cleanup previous ollama process", "err", err)
	}

	reaped := false
	for ctx.Err() == nil {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(restartDelay):
		}

		cmd, err := s.cmd(ctx)
		if err != nil {
			return err
		}

		if err := cmd.Start(); err != nil {
			return err
		}

		err = os.WriteFile(pidFile, []byte(strconv.Itoa(cmd.Process.Pid)), 0o644)
		if err != nil {
			slog.Warn("failed to write pid file", "file", pidFile, "err", err)
		}

		if err = cmd.Wait(); err != nil && !errors.Is(err, context.Canceled) {
			var exitErr *exec.ExitError
			if errors.As(err, &exitErr) && exitErr.ExitCode() == 1 && !s.dev && !reaped {
				reaped = true
				// This could be a port conflict, try to kill any existing ollama processes
				if err := reapServers(); err != nil {
					slog.Warn("failed to stop existing ollama server", "err", err)
				} else {
					slog.Debug("conflicting server stopped, waiting for port to be released")
					continue
				}
			}
			slog.Error("ollama exited", "err", err)
		}
	}
	return ctx.Err()
}

func (s *Server) cmd(ctx context.Context) (*exec.Cmd, error) {
	settings, err := s.store.Settings()
	if err != nil {
		return nil, err
	}

	cmd := commandContext(ctx, s.bin, "serve")
	cmd.Stdout, cmd.Stderr = s.log, s.log

	// Copy and mutate the environment to merge in settings the user has specified without dups
	env := map[string]string{}
	for _, kv := range os.Environ() {
		s := strings.SplitN(kv, "=", 2)
		env[s[0]] = s[1]
	}
	if settings.Expose {
		env["OLLAMA_HOST"] = "0.0.0.0"
	}
	if settings.Browser {
		env["OLLAMA_ORIGINS"] = "*"
	}
	if settings.Models != "" {
		if _, err := os.Stat(settings.Models); err == nil {
			env["OLLAMA_MODELS"] = settings.Models
		} else {
			slog.Warn("models path not accessible, using default", "path", settings.Models, "err", err)
		}
	}
	if settings.ContextLength > 0 {
		env["OLLAMA_CONTEXT_LENGTH"] = strconv.Itoa(settings.ContextLength)
	}
	cmd.Env = []string{}
	for k, v := range env {
		cmd.Env = append(cmd.Env, k+"="+v)
	}

	cmd.Cancel = func() error {
		if cmd.Process == nil {
			return nil
		}
		return stop(cmd.Process)
	}

	return cmd, nil
}

func openRotatingLog() (io.WriteCloser, error) {
	// TODO consider rotation based on size or time, not just every server invocation
	dir := filepath.Dir(serverLogPath)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("create log directory: %w", err)
	}

	logrotate.Rotate(serverLogPath)
	f, err := os.OpenFile(serverLogPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return nil, fmt.Errorf("open log file: %w", err)
	}
	return f, nil
}

// Attempt to retrieve inference compute information from the server
// log.  Set ctx to timeout to control how long to wait for the logs to appear
func GetInferenceComputer(ctx context.Context) ([]InferenceCompute, error) {
	inference := []InferenceCompute{}
	marker := regexp.MustCompile(`inference compute.*library=`)
	q := `inference compute.*%s=["]([^"]*)["]`
	nq := `inference compute.*%s=(\S+)\s`
	type regex struct {
		q  *regexp.Regexp
		nq *regexp.Regexp
	}
	regexes := map[string]regex{
		"library": {
			q:  regexp.MustCompile(fmt.Sprintf(q, "library")),
			nq: regexp.MustCompile(fmt.Sprintf(nq, "library")),
		},
		"variant": {
			q:  regexp.MustCompile(fmt.Sprintf(q, "variant")),
			nq: regexp.MustCompile(fmt.Sprintf(nq, "variant")),
		},
		"compute": {
			q:  regexp.MustCompile(fmt.Sprintf(q, "compute")),
			nq: regexp.MustCompile(fmt.Sprintf(nq, "compute")),
		},
		"driver": {
			q:  regexp.MustCompile(fmt.Sprintf(q, "driver")),
			nq: regexp.MustCompile(fmt.Sprintf(nq, "driver")),
		},
		"name": {
			q:  regexp.MustCompile(fmt.Sprintf(q, "name")),
			nq: regexp.MustCompile(fmt.Sprintf(nq, "name")),
		},
		"total": {
			q:  regexp.MustCompile(fmt.Sprintf(q, "total")),
			nq: regexp.MustCompile(fmt.Sprintf(nq, "total")),
		},
	}
	get := func(field, line string) string {
		regex, ok := regexes[field]
		if !ok {
			slog.Warn("missing field", "field", field)
			return ""
		}
		match := regex.q.FindStringSubmatch(line)

		if len(match) > 1 {
			return match[1]
		}
		match = regex.nq.FindStringSubmatch(line)
		if len(match) > 1 {
			return match[1]
		}
		return ""
	}
	for {
		select {
		case <-ctx.Done():
			return nil, fmt.Errorf("timeout scanning server log for inference compute details")
		default:
		}
		file, err := os.Open(serverLogPath)
		if err != nil {
			slog.Debug("failed to open server log", "log", serverLogPath, "error", err)
			time.Sleep(time.Second)
			continue
		}
		defer file.Close()
		scanner := bufio.NewScanner(file)
		for scanner.Scan() {
			line := scanner.Text()
			match := marker.FindStringSubmatch(line)
			if len(match) > 0 {
				ic := InferenceCompute{
					Library: get("library", line),
					Variant: get("variant", line),
					Compute: get("compute", line),
					Driver:  get("driver", line),
					Name:    get("name", line),
					VRAM:    get("total", line),
				}

				slog.Info("Matched", "inference compute", ic)
				inference = append(inference, ic)
			} else {
				// Break out on first non matching line after we start matching
				if len(inference) > 0 {
					return inference, nil
				}
			}
		}
		time.Sleep(100 * time.Millisecond)
	}
}
