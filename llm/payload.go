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
	"strings"

	"golang.org/x/exp/slices"
	"golang.org/x/sync/errgroup"

	"github.com/jmorganca/ollama/gpu"
)

var errPayloadMissing = fmt.Errorf("expected payloads not included in this build of ollama")

// TODO: error check this or initialize it in init()
// TODO: this should be a determinate location to avoid required files disappearing
var workDir, _ = os.MkdirTemp("", "ollama")

func Init() error {
	slog.Info("extracting embedded files", "dir", workDir)

	// also extract the .metal shader on darwin/arm64
	if runtime.GOOS == "darwin" && runtime.GOARCH == "arm64" {
		err := extractFiles(workDir, "build/*/*/*/bin/ggml-metal.metal.gz")
		if err != nil {
			return fmt.Errorf("extract .metal shader: %v", err)
		}

		os.Setenv("GGML_METAL_PATH_RESOURCES", filepath.Join(workDir, "metal"))
	}

	var binGlob string
	if runtime.GOOS == "windows" {
		binGlob = "build/windows/*/*/bin/*/*"
	} else {
		binGlob = "build/*/*/*/bin/*"
	}

	// extract server libraries
	err := extractFiles(workDir, binGlob)
	if err != nil {
		return fmt.Errorf("extract binaries: %v", err)
	}

	// TODO: print available variants
	// slog.Info(fmt.Sprintf("Dynamic LLM libraries %v", variants))
	// slog.Debug("Override detection logic by setting OLLAMA_LLM_LIBRARY")
	var variants []string
	for v := range available() {
		variants = append(variants, v)
	}
	slog.Info(fmt.Sprintf("Dynamic LLM libraries %v", variants))
	slog.Debug("Override detection logic by setting OLLAMA_LLM_LIBRARY")

	return nil
}

// binary names may contain an optional variant separated by '_'
// For example, "ollama_rocm_v6" and "ollama_rocm_v5" or "ollama_cpu" and "ollama_cpu_avx2"
// Any library without a variant is the lowest common denominator
func available() map[string]string {
	// glob workDir for files that start with ollama_
	pattern := filepath.Join(workDir, "*")

	files, err := filepath.Glob(pattern)
	if err != nil {
		slog.Debug("could not glob", "pattern", pattern, "error", err)
		return nil
	}

	servers := make(map[string]string)
	for _, file := range files {
		slog.Debug("available: found", "file", file)
		servers[filepath.Base(file)] = file
	}

	return servers
}

// serversForGpu returns a list of compatible servers give the provided GPU
// info, ordered by performance. assumes Init() has been called
// TODO: complete this given above
func serversForGpu(info gpu.GpuInfo) []string {
	// glob workDir for files that start with ollama_
	available := available()
	requested := info.Library
	if info.Variant != "" {
		requested += "_" + info.Variant
	}

	servers := []string{}

	// exact match first
	for a := range available {
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
		for a := range available {
			if info.Library == strings.Split(a, "_")[0] && a != requested {
				alt = append(alt, a)
			}
		}

		slices.Sort(alt)
		servers = append(servers, alt...)
	}

	// Load up the best CPU variant if not primary requested
	if info.Library != "cpu" {
		variant := gpu.GetCPUVariant()
		// If no variant, then we fall back to default
		// If we have a variant, try that if we find an exact match
		// Attempting to run the wrong CPU instructions will panic the
		// process
		if variant != "" {
			for cmp := range available {
				if cmp == "cpu_"+variant {
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

func Cleanup() error {
	return os.RemoveAll(workDir)
}

// extract extracts the embedded files to the target directory
// TODO: return a list of files extracted
func extractFiles(targetDir string, glob string) error {
	files, err := fs.Glob(libEmbed, glob)
	if err != nil || len(files) == 0 {
		return errPayloadMissing
	}

	if err := os.MkdirAll(targetDir, 0o755); err != nil {
		return fmt.Errorf("extractFiles could not mkdir %s: %v", workDir, err)
	}

	g := new(errgroup.Group)

	// build/$OS/$GOARCH/$VARIANT/{bin,lib}/$FILE
	for _, file := range files {
		filename := file
		var variant string

		if runtime.GOOS == "windows" {
			variant = filepath.Base(filepath.Dir(filepath.Dir(filepath.Dir(filename))))
		} else {
			variant = filepath.Base(filepath.Dir(filepath.Dir(filename)))
		}

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
				return fmt.Errorf("extractFiles could not mkdir %s: %v", workDir, err)
			}

			// rename "server" to a more descriptive binary name
			base := filepath.Base(filename)
			destFilename := filepath.Join(variantDir, base)
			destFilename = strings.Replace(destFilename, "server", "ollama_llama_server", 1)

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

	return g.Wait()
}
