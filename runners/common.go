package runners

import (
	"compress/gzip"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"log/slog"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"syscall"

	"golang.org/x/sync/errgroup"

	"github.com/ollama/ollama/discover"
	"github.com/ollama/ollama/envconfig"
)

const (
	binGlob = "*/*/*/*"
)

var (
	lock       sync.Mutex
	runnersDir = ""
)

// Return the location where runners are stored
// If runners are payloads, this will either extract them
// or refresh them if any have disappeared due to tmp cleaners
func Refresh(payloadFS fs.FS) (string, error) {
	lock.Lock()
	defer lock.Unlock()
	var err error

	// Wire up extra logging on our first load
	if runnersDir == "" {
		defer func() {
			var runners []string
			for v := range GetAvailableServers(runnersDir) {
				runners = append(runners, v)
			}
			slog.Info("Dynamic LLM libraries", "runners", runners)
			slog.Debug("Override detection logic by setting OLLAMA_LLM_LIBRARY")
		}()
	}

	if hasPayloads(payloadFS) {
		if runnersDir == "" {
			runnersDir, err = extractRunners(payloadFS)
		} else {
			err = refreshRunners(payloadFS, runnersDir)
		}
	} else if runnersDir == "" {
		runnersDir, err = locateRunners()
	}

	return runnersDir, err
}

func Cleanup(payloadFS fs.FS) {
	lock.Lock()
	defer lock.Unlock()
	if hasPayloads(payloadFS) && runnersDir != "" {
		// We want to fully clean up the tmpdir parent of the payloads dir
		tmpDir := filepath.Clean(filepath.Join(runnersDir, ".."))
		slog.Debug("cleaning up", "dir", tmpDir)
		err := os.RemoveAll(tmpDir)
		if err != nil {
			slog.Warn("failed to clean up", "dir", tmpDir, "err", err)
		}
	}
}

func locateRunners() (string, error) {
	exe, err := os.Executable()
	if err != nil {
		return "", err
	}

	cwd, err := os.Getwd()
	if err != nil {
		return "", err
	}

	var paths []string
	for _, root := range []string{filepath.Dir(exe), filepath.Join(filepath.Dir(exe), envconfig.LibRelativeToExe()), cwd} {
		paths = append(paths,
			root,
			filepath.Join(root, runtime.GOOS+"-"+runtime.GOARCH),
			filepath.Join(root, "dist", runtime.GOOS+"-"+runtime.GOARCH),
		)
	}

	// Try a few variations to improve developer experience when building from source in the local tree
	for _, path := range paths {
		candidate := filepath.Join(path, "lib", "ollama", "runners")
		if _, err := os.Stat(candidate); err == nil {
			return candidate, nil
		}
	}
	return "", fmt.Errorf("unable to locate runners in any search path %v", paths)
}

// Return true if we're carying nested payloads for the runners
func hasPayloads(payloadFS fs.FS) bool {
	files, err := fs.Glob(payloadFS, binGlob)
	if err != nil || len(files) == 0 || (len(files) == 1 && strings.Contains(files[0], "placeholder")) {
		return false
	}
	return true
}

func extractRunners(payloadFS fs.FS) (string, error) {
	cleanupTmpDirs()
	tmpDir, err := os.MkdirTemp(envconfig.TmpDir(), "ollama")
	if err != nil {
		return "", fmt.Errorf("failed to generate tmp dir: %w", err)
	}
	// Track our pid so we can clean up orphaned tmpdirs
	n := filepath.Join(tmpDir, "ollama.pid")
	if err := os.WriteFile(n, []byte(strconv.Itoa(os.Getpid())), 0o644); err != nil {
		slog.Warn("failed to write pid file", "file", n, "error", err)
	}
	// We create a distinct subdirectory for payloads within the tmpdir
	// This will typically look like /tmp/ollama3208993108/runners on linux
	rDir := filepath.Join(tmpDir, "runners")

	slog.Info("extracting embedded files", "dir", rDir)
	return rDir, refreshRunners(payloadFS, rDir)
}

func refreshRunners(payloadFS fs.FS, rDir string) error {
	// extract or refresh server libraries
	err := extractFiles(payloadFS, rDir, binGlob)
	if err != nil {
		return fmt.Errorf("extract binaries: %v", err)
	}
	return nil
}

// extract extracts the embedded files to the target directory
func extractFiles(payloadFS fs.FS, targetDir string, glob string) error {
	files, err := fs.Glob(payloadFS, glob)
	if err != nil || len(files) == 0 {
		// Should not happen
		return fmt.Errorf("extractFiles called without payload present")
	}

	if err := os.MkdirAll(targetDir, 0o755); err != nil {
		return fmt.Errorf("extractFiles could not mkdir %s: %v", targetDir, err)
	}

	g := new(errgroup.Group)

	// $OS/$GOARCH/$RUNNER/$FILE
	for _, file := range files {
		filename := file

		runner := filepath.Base(filepath.Dir(filename))

		slog.Debug("extracting", "runner", runner, "payload", filename)

		g.Go(func() error {
			srcf, err := payloadFS.Open(filename)
			if err != nil {
				return err
			}
			defer srcf.Close()

			src := io.Reader(srcf)
			if strings.HasSuffix(filename, ".gz") {
				src, err = gzip.NewReader(src)
				if err != nil {
					return fmt.Errorf("decompress payload %s: %v", filename, err)
				}
				filename = strings.TrimSuffix(filename, ".gz")
			}

			runnerDir := filepath.Join(targetDir, runner)
			if err := os.MkdirAll(runnerDir, 0o755); err != nil {
				return fmt.Errorf("extractFiles could not mkdir %s: %v", runnerDir, err)
			}

			base := filepath.Base(filename)
			destFilename := filepath.Join(runnerDir, base)

			_, err = os.Stat(destFilename)
			switch {
			case errors.Is(err, os.ErrNotExist):
				destFile, err := os.OpenFile(destFilename, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o755)
				if err != nil {
					return fmt.Errorf("write payload %s: %v", filename, err)
				}
				defer destFile.Close()
				if _, err := io.Copy(destFile, src); err != nil {
					return fmt.Errorf("copy payload %s: %v", filename, err)
				}
			case err != nil:
				return fmt.Errorf("stat payload %s: %v", filename, err)
			}
			return nil
		})
	}

	err = g.Wait()
	if err != nil {
		slog.Error("failed to extract files", "error", err)
		// If we fail to extract, the payload dir is most likely unusable, so cleanup whatever we extracted
		err := os.RemoveAll(targetDir)
		if err != nil {
			slog.Warn("failed to cleanup incomplete payload dir", "dir", targetDir, "error", err)
		}
		return err
	}
	return nil
}

// Best effort to clean up prior tmpdirs
func cleanupTmpDirs() {
	tmpDir := envconfig.TmpDir()
	if tmpDir == "" {
		tmpDir = os.TempDir()
	}
	matches, err := filepath.Glob(filepath.Join(tmpDir, "ollama*", "ollama.pid"))
	if err != nil {
		return
	}

	for _, match := range matches {
		raw, err := os.ReadFile(match)
		if errors.Is(err, os.ErrNotExist) {
			slog.Debug("not a ollama runtime directory, skipping", "path", match)
			continue
		} else if err != nil {
			slog.Warn("could not read ollama.pid, skipping", "path", match, "error", err)
			continue
		}

		pid, err := strconv.Atoi(string(raw))
		if err != nil {
			slog.Warn("invalid pid, skipping", "path", match, "error", err)
			continue
		}

		p, err := os.FindProcess(pid)
		if err == nil && !errors.Is(p.Signal(syscall.Signal(0)), os.ErrProcessDone) {
			slog.Warn("process still running, skipping", "pid", pid, "path", match)
			continue
		}

		if err := os.Remove(match); err != nil {
			slog.Warn("could not cleanup stale pidfile", "path", match, "error", err)
		}

		runners := filepath.Join(filepath.Dir(match), "runners")
		if err := os.RemoveAll(runners); err != nil {
			slog.Warn("could not cleanup stale runners", "path", runners, "error", err)
		}

		if err := os.Remove(filepath.Dir(match)); err != nil {
			slog.Warn("could not cleanup stale tmpdir", "path", filepath.Dir(match), "error", err)
		}
	}
}

// directory names are the name of the runner and may contain an optional
// variant prefixed with '_' as the separator. For example, "cuda_v11" and
// "cuda_v12" or "cpu" and "cpu_avx2". Any library without a variant is the
// lowest common denominator
func GetAvailableServers(payloadsDir string) map[string]string {
	if payloadsDir == "" {
		slog.Error("empty runner dir")
		return nil
	}

	// glob payloadsDir for files that start with ollama_
	pattern := filepath.Join(payloadsDir, "*", "ollama_*")

	files, err := filepath.Glob(pattern)
	if err != nil {
		slog.Debug("could not glob", "pattern", pattern, "error", err)
		return nil
	}

	servers := make(map[string]string)
	for _, file := range files {
		slog.Debug("availableServers : found", "file", file)
		servers[filepath.Base(filepath.Dir(file))] = filepath.Dir(file)
	}

	return servers
}

// serversForGpu returns a list of compatible servers give the provided GPU
// info, ordered by performance. assumes Init() has been called
// TODO - switch to metadata based mapping
func ServersForGpu(info discover.GpuInfo) []string {
	// glob workDir for files that start with ollama_
	availableServers := GetAvailableServers(runnersDir)
	requested := info.Library
	if info.Variant != discover.CPUCapabilityNone.String() {
		requested += "_" + info.Variant
	}

	servers := []string{}

	// exact match first
	for a := range availableServers {
		if a == requested {
			servers = []string{a}

			if a == "metal" {
				return servers
			}

			break
		}
	}

	alt := []string{}

	// Then for GPUs load alternates and sort the list for consistent load ordering
	if info.Library != "cpu" {
		for a := range availableServers {
			if info.Library == strings.Split(a, "_")[0] && a != requested {
				alt = append(alt, a)
			}
		}

		slices.Sort(alt)
		servers = append(servers, alt...)
	}

	if !(runtime.GOOS == "darwin" && runtime.GOARCH == "arm64") {
		// Load up the best CPU variant if not primary requested
		if info.Library != "cpu" {
			variant := discover.GetCPUCapability()
			// If no variant, then we fall back to default
			// If we have a variant, try that if we find an exact match
			// Attempting to run the wrong CPU instructions will panic the
			// process
			if variant != discover.CPUCapabilityNone {
				for cmp := range availableServers {
					if cmp == "cpu_"+variant.String() {
						servers = append(servers, cmp)
						break
					}
				}
			} else {
				servers = append(servers, "cpu")
			}
		}

		if len(servers) == 0 {
			servers = []string{"cpu"}
		}
	}

	return servers
}

// Return the optimal server for this CPU architecture
func ServerForCpu() string {
	if runtime.GOOS == "darwin" && runtime.GOARCH == "arm64" {
		return "metal"
	}
	variant := discover.GetCPUCapability()
	availableServers := GetAvailableServers(runnersDir)
	if variant != discover.CPUCapabilityNone {
		for cmp := range availableServers {
			if cmp == "cpu_"+variant.String() {
				return cmp
			}
		}
	}
	return "cpu"
}
