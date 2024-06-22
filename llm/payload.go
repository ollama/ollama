package llm

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
	"strings"

	"golang.org/x/sync/errgroup"

	"github.com/ollama/ollama/gpu"
)

var errPayloadMissing = errors.New("expected payloads not included in this build of ollama")

func Init() error {
	payloadsDir, err := gpu.PayloadsDir()
	if err != nil {
		return err
	}

	if runtime.GOOS != "windows" {
		slog.Info("extracting embedded files", "dir", payloadsDir)
		binGlob := "build/*/*/*/bin/*"

		// extract server libraries
		err = extractFiles(payloadsDir, binGlob)
		if err != nil {
			return fmt.Errorf("extract binaries: %v", err)
		}
	}

	var variants []string
	for v := range availableServers() {
		variants = append(variants, v)
	}
	slog.Info(fmt.Sprintf("Dynamic LLM libraries %v", variants))
	slog.Debug("Override detection logic by setting OLLAMA_LLM_LIBRARY")

	return nil
}

// binary names may contain an optional variant separated by '_'
// For example, "ollama_rocm_v6" and "ollama_rocm_v5" or "ollama_cpu" and "ollama_cpu_avx2"
// Any library without a variant is the lowest common denominator
func availableServers() map[string]string {
	payloadsDir, err := gpu.PayloadsDir()
	if err != nil {
		slog.Error("payload lookup error", "error", err)
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
func serversForGpu(info gpu.GpuInfo) []string {
	// glob workDir for files that start with ollama_
	availableServers := availableServers()
	requested := info.Library
	if info.Variant != gpu.CPUCapabilityNone {
		requested += "_" + info.Variant.String()
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

	// Load up the best CPU variant if not primary requested
	if info.Library != "cpu" {
		variant := gpu.GetCPUCapability()
		// If no variant, then we fall back to default
		// If we have a variant, try that if we find an exact match
		// Attempting to run the wrong CPU instructions will panic the
		// process
		if variant != gpu.CPUCapabilityNone {
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

	return servers
}

// Return the optimal server for this CPU architecture
func serverForCpu() string {
	if runtime.GOOS == "darwin" && runtime.GOARCH == "arm64" {
		return "metal"
	}
	variant := gpu.GetCPUCapability()
	availableServers := availableServers()
	if variant != gpu.CPUCapabilityNone {
		for cmp := range availableServers {
			if cmp == "cpu_"+variant.String() {
				return cmp
			}
		}
	}
	return "cpu"
}

// extract extracts the embedded files to the target directory
func extractFiles(targetDir string, glob string) error {
	files, err := fs.Glob(libEmbed, glob)
	if err != nil || len(files) == 0 {
		return errPayloadMissing
	}

	if err := os.MkdirAll(targetDir, 0o755); err != nil {
		return fmt.Errorf("extractFiles could not mkdir %s: %v", targetDir, err)
	}

	g := new(errgroup.Group)

	// build/$OS/$GOARCH/$VARIANT/{bin,lib}/$FILE
	for _, file := range files {
		filename := file

		variant := filepath.Base(filepath.Dir(filepath.Dir(filename)))

		slog.Debug("extracting", "variant", variant, "file", filename)

		g.Go(func() error {
			srcf, err := libEmbed.Open(filename)
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

			variantDir := filepath.Join(targetDir, variant)
			if err := os.MkdirAll(variantDir, 0o755); err != nil {
				return fmt.Errorf("extractFiles could not mkdir %s: %v", variantDir, err)
			}

			base := filepath.Base(filename)
			destFilename := filepath.Join(variantDir, base)

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
		// If we fail to extract, the payload dir is unusable, so cleanup whatever we extracted
		gpu.Cleanup()
		return err
	}
	return nil
}
