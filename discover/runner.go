package discover

// Runner based GPU discovery

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"math/rand"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/ml"
)

func GPUDevices() []ml.DeviceInfo {
	devices := []ml.DeviceInfo{}

	// Overall timeout to try to run 1 or more runners for GPU discovery
	ctx, cancel := context.WithTimeout(context.TODO(), 5*time.Second)
	defer cancel()
	directories := map[string]struct{}{}

	exe, err := os.Executable()
	if err != nil {
		slog.Error("unable to lookup executable path", "error", err)
		return nil
	}

	if eval, err := filepath.EvalSymlinks(exe); err == nil {
		exe = eval
	}

	files, err := filepath.Glob(filepath.Join(LibOllamaPath, "*", "*ggml-*"))
	if err != nil {
		slog.Debug("unable to lookup runner library directories", "error", err)
	}
	for _, file := range files {
		directories[filepath.Dir(file)] = struct{}{}
	}

	// Our current packaging model places ggml-hip in the main directory
	// but keeps rocm in an isolated directory.  We have to add it to
	// the [LD_LIBRARY_]PATH so ggml-hip will load properly
	rocmDir := filepath.Join(LibOllamaPath, "rocm")
	if _, err := os.Stat(rocmDir); err != nil {
		rocmDir = ""
	}

	directories[""] = struct{}{}
	slog.Info("discovering available GPUs")

	// TODO if we find timeouts hitting under stress scenarios refactor this to
	// run the runners in parallel go routines to reduce the overall time
	for dir := range directories {
		// TODO DRY out with llm/server.go
		slog.Debug("spawing runner for", "lib_dir", dir)
		port := 0
		if a, err := net.ResolveTCPAddr("tcp", "localhost:0"); err == nil {
			var l *net.TCPListener
			if l, err = net.ListenTCP("tcp", a); err == nil {
				port = l.Addr().(*net.TCPAddr).Port
				l.Close()
			}
		}
		if port == 0 {
			slog.Debug("ResolveTCPAddr failed, using random port")
			port = rand.Intn(65535-49152) + 49152 // get a random port in the ephemeral range
		}
		params := []string{"runner", "--ollama-engine", "--port", strconv.Itoa(port)}
		var pathEnv string
		switch runtime.GOOS {
		case "windows":
			pathEnv = "PATH"
		case "darwin":
			pathEnv = "DYLD_LIBRARY_PATH"
		default:
			pathEnv = "LD_LIBRARY_PATH"
		}
		libraryPaths := []string{LibOllamaPath, dir}
		if rocmDir != "" {
			libraryPaths = append(libraryPaths, rocmDir)
		}
		// Note: we always put our dependency paths first
		// since these are the exact version we compiled/linked against
		if libraryPath, ok := os.LookupEnv(pathEnv); ok {
			libraryPaths = append(libraryPaths, filepath.SplitList(libraryPath)...)
		}

		ggmlPaths := []string{LibOllamaPath, dir}
		cmd := exec.Command(exe, params...)
		cmd.Env = os.Environ()
		cmd.Stdout = os.Stdout
		errBuf := &bytes.Buffer{}
		if envconfig.LogLevel() == slog.Level(-8) {
			cmd.Stderr = os.Stderr
		} else {
			cmd.Stderr = errBuf
		}
		// cmd.SysProcAttr = llm.LlamaServerSysProcAttr // circular dependency - bring back once refactored
		cmd.Env = append(cmd.Env, "OLLAMA_LIBRARY_PATH="+strings.Join(ggmlPaths, string(filepath.ListSeparator)))
		pathEnvVal := strings.Join(libraryPaths, string(filepath.ListSeparator))
		pathNeeded := true
		for i := range cmd.Env {
			cmp := strings.SplitN(cmd.Env[i], "=", 2)
			if strings.EqualFold(cmp[0], pathEnv) {
				cmd.Env[i] = pathEnv + "=" + pathEnvVal
				pathNeeded = false
			}
		}
		if pathNeeded {
			cmd.Env = append(cmd.Env, pathEnv+"="+pathEnvVal)
		}
		slog.Log(context.TODO(), logutil.LevelTrace, "startung runner for device discovery", "env", cmd.Env, "cmd", cmd)
		if err = cmd.Start(); err != nil {
			slog.Warn("unable to start discovery subprocess", "cmd", cmd, "error", err)
			continue
		}

		defer cmd.Process.Kill()

		// Individual timeout for a runner
		timeout := time.After(2 * time.Second)
		tick := time.Tick(500 * time.Millisecond)
	retry:
		for {
			select {
			case <-ctx.Done():
				slog.Warn("failed to finish discovery before timeout")
				os.Stderr.Write(errBuf.Bytes())
				return devices
			case <-timeout:
				slog.Warn("timeout waiting for server to respond with GPU list", "lib_dir", dir)
				os.Stderr.Write(errBuf.Bytes())
				break retry
			case <-tick:
				r, err := http.NewRequestWithContext(ctx, http.MethodGet, fmt.Sprintf("http://127.0.0.1:%d/info", port), nil)
				if err != nil {
					slog.Warn("failed to create request", "error", err)
					continue
				}
				r.Header.Set("Content-Type", "application/json")

				resp, err := http.DefaultClient.Do(r)
				if err != nil {
					slog.Warn("failed to send request", "error", err)
					continue
				}
				defer resp.Body.Close()

				body, err := io.ReadAll(resp.Body)
				if err != nil {
					slog.Warn("failed to read response", "error", err)
					continue
				}
				var moreDevices []ml.DeviceInfo
				if err := json.Unmarshal(body, &moreDevices); err != nil {
					slog.Warn("unmarshal encode response", "error", err)
					continue
				}
				devices = append(devices, moreDevices...)
				break retry
			}
		}
	}

	return devices
}
