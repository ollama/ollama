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

// Libraries names may contain an optional variant separated by '_'
// For example, "rocm_v6" and "rocm_v5" or "cpu" and "cpu_avx2"
// Any library without a variant is the lowest common denominator
var availableDynLibs = map[string]string{}

const pathComponentCount = 7

// getDynLibs returns an ordered list of LLM libraries to try, starting with the best
func getDynLibs(gpuInfo gpu.GpuInfo) []string {
	// Short circuit if we know we're using the default built-in (darwin only)
	if gpuInfo.Library == "default" {
		return []string{"default"}
	}
	// TODO - temporary until we have multiple CPU variations for Darwin
	// Short circuit on darwin with metal only
	if len(availableDynLibs) == 1 {
		if _, onlyMetal := availableDynLibs["metal"]; onlyMetal {
			return []string{availableDynLibs["metal"]}
		}
	}

	exactMatch := ""
	dynLibs := []string{}
	altDynLibs := []string{}
	requested := gpuInfo.Library
	if gpuInfo.Variant != "" {
		requested += "_" + gpuInfo.Variant
	}
	// Try to find an exact match
	for cmp := range availableDynLibs {
		if requested == cmp {
			exactMatch = cmp
			dynLibs = []string{availableDynLibs[cmp]}
			break
		}
	}
	// Then for GPUs load alternates and sort the list for consistent load ordering
	if gpuInfo.Library != "cpu" {
		for cmp := range availableDynLibs {
			if gpuInfo.Library == strings.Split(cmp, "_")[0] && cmp != exactMatch {
				altDynLibs = append(altDynLibs, cmp)
			}
		}
		slices.Sort(altDynLibs)
		for _, altDynLib := range altDynLibs {
			dynLibs = append(dynLibs, availableDynLibs[altDynLib])
		}
	}

	// Load up the best CPU variant if not primary requested
	if gpuInfo.Library != "cpu" {
		variant := gpu.GetCPUVariant()
		// If no variant, then we fall back to default
		// If we have a variant, try that if we find an exact match
		// Attempting to run the wrong CPU instructions will panic the
		// process
		if variant != "" {
			for cmp := range availableDynLibs {
				if cmp == "cpu_"+variant {
					dynLibs = append(dynLibs, availableDynLibs[cmp])
					break
				}
			}
		} else {
			dynLibs = append(dynLibs, availableDynLibs["cpu"])
		}
	}

	// Finally, if we didn't find any matches, LCD CPU FTW
	if len(dynLibs) == 0 {
		dynLibs = []string{availableDynLibs["cpu"]}
	}
	slog.Debug(fmt.Sprintf("ordered list of LLM libraries to try %v", dynLibs))
	return dynLibs
}

func rocmDynLibPresent() bool {
	for dynLibName := range availableDynLibs {
		if strings.HasPrefix(dynLibName, "rocm") {
			return true
		}
	}
	return false
}

func nativeInit() error {
	slog.Info("Extracting dynamic libraries...")
	libDir, err := libDir()
	if err != nil {
		return err
	}
	if runtime.GOOS == "darwin" {
		err := extractPayloadFiles(libDir, "llama.cpp/ggml-metal.metal")
		if err != nil {
			if err == payloadMissing {
				// TODO perhaps consider this a hard failure on arm macs?
				slog.Info("ggml-meta.metal payload missing")
				return nil
			}
			return err
		}
		os.Setenv("GGML_METAL_PATH_RESOURCES", libDir)
	}

	libs, err := extractDynamicLibs(libDir, "llama.cpp/build/*/*/*/lib/*")
	if err != nil {
		if err == payloadMissing {
			slog.Info(fmt.Sprintf("%s", payloadMissing))
			return nil
		}
		return err
	}
	for _, lib := range libs {
		// The last dir component is the variant name
		variant := filepath.Base(filepath.Dir(lib))
		availableDynLibs[variant] = lib
	}

	if err := verifyDriverAccess(); err != nil {
		return err
	}

	// Report which dynamic libraries we have loaded to assist troubleshooting
	variants := make([]string, len(availableDynLibs))
	i := 0
	for variant := range availableDynLibs {
		variants[i] = variant
		i++
	}
	slog.Info(fmt.Sprintf("Dynamic LLM libraries %v", variants))
	slog.Debug("Override detection logic by setting OLLAMA_LLM_LIBRARY")

	return nil
}

func extractDynamicLibs(libDir, glob string) ([]string, error) {
	files, err := fs.Glob(libEmbed, glob)
	if err != nil || len(files) == 0 {
		return nil, payloadMissing
	}
	libs := []string{}

	g := new(errgroup.Group)
	for _, file := range files {
		pathComps := strings.Split(file, "/")
		if len(pathComps) != pathComponentCount {
			slog.Error(fmt.Sprintf("unexpected payload components: %v", pathComps))
			continue
		}

		file := file
		g.Go(func() error {
			// llama.cpp/build/$OS/$GOARCH/$VARIANT/lib/$LIBRARY
			// Include the variant in the path to avoid conflicts between multiple server libs
			targetDir := filepath.Join(libDir, pathComps[pathComponentCount-3])
			srcFile, err := libEmbed.Open(file)
			if err != nil {
				return fmt.Errorf("read payload %s: %v", file, err)
			}
			defer srcFile.Close()
			if err := os.MkdirAll(targetDir, 0o755); err != nil {
				return fmt.Errorf("create payload lib dir %s: %v", libDir, err)
			}
			src := io.Reader(srcFile)
			filename := file
			if strings.HasSuffix(file, ".gz") {
				src, err = gzip.NewReader(src)
				if err != nil {
					return fmt.Errorf("decompress payload %s: %v", file, err)
				}
				filename = strings.TrimSuffix(filename, ".gz")
			}

			destFile := filepath.Join(targetDir, filepath.Base(filename))
			if strings.Contains(destFile, "server") {
				libs = append(libs, destFile)
			}

			_, err = os.Stat(destFile)
			switch {
			case errors.Is(err, os.ErrNotExist):
				destFp, err := os.OpenFile(destFile, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o755)
				if err != nil {
					return fmt.Errorf("write payload %s: %v", file, err)
				}
				defer destFp.Close()
				if _, err := io.Copy(destFp, src); err != nil {
					return fmt.Errorf("copy payload %s: %v", file, err)
				}
			case err != nil:
				return fmt.Errorf("stat payload %s: %v", file, err)
			case err == nil:
				slog.Debug("payload already exists: " + destFile)
			}
			return nil
		})
	}
	return libs, g.Wait()
}

func extractPayloadFiles(libDir, glob string) error {
	files, err := fs.Glob(libEmbed, glob)
	if err != nil || len(files) == 0 {
		return payloadMissing
	}

	for _, file := range files {
		srcFile, err := libEmbed.Open(file)
		if err != nil {
			return fmt.Errorf("read payload %s: %v", file, err)
		}
		defer srcFile.Close()
		if err := os.MkdirAll(libDir, 0o755); err != nil {
			return fmt.Errorf("create payload lib dir %s: %v", libDir, err)
		}
		src := io.Reader(srcFile)
		filename := file
		if strings.HasSuffix(file, ".gz") {
			src, err = gzip.NewReader(src)
			if err != nil {
				return fmt.Errorf("decompress payload %s: %v", file, err)
			}
			filename = strings.TrimSuffix(filename, ".gz")
		}

		destFile := filepath.Join(libDir, filepath.Base(filename))
		_, err = os.Stat(destFile)
		switch {
		case errors.Is(err, os.ErrNotExist):
			destFp, err := os.OpenFile(destFile, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o755)
			if err != nil {
				return fmt.Errorf("write payload %s: %v", file, err)
			}
			defer destFp.Close()
			if _, err := io.Copy(destFp, src); err != nil {
				return fmt.Errorf("copy payload %s: %v", file, err)
			}
		case err != nil:
			return fmt.Errorf("stat payload %s: %v", file, err)
		case err == nil:
			slog.Debug("payload already exists: " + destFile)
		}
	}
	return nil
}

func verifyDriverAccess() error {
	if runtime.GOOS != "linux" {
		return nil
	}
	// Only check ROCm access if we have the dynamic lib loaded
	if rocmDynLibPresent() {
		// Verify we have permissions - either running as root, or we have group access to the driver
		fd, err := os.OpenFile("/dev/kfd", os.O_RDWR, 0666)
		if err != nil {
			if errors.Is(err, fs.ErrPermission) {
				return fmt.Errorf("Radeon card detected, but permissions not set up properly.  Either run ollama as root, or add you user account to the render group.")
			} else if errors.Is(err, fs.ErrNotExist) {
				// expected behavior without a radeon card
				return nil
			}

			return fmt.Errorf("failed to check permission on /dev/kfd: %w", err)
		}
		fd.Close()
	}
	return nil
}
