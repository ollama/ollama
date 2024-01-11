package llm

import (
	"errors"
	"fmt"
	"golang.org/x/exp/slices"
	"io"
	"io/fs"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/jmorganca/ollama/gpu"
)

// Libraries names may contain an optional variant separated by '_'
// For example, "rocm_v6" and "rocm_v5" or "cpu" and "cpu_avx2"
// Any library without a variant is the lowest common denominator
var availableDynLibs = map[string]string{}

const pathComponentCount = 6

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

func nativeInit(workdir string) error {
	if runtime.GOOS == "darwin" {
		err := extractPayloadFiles(workdir, "llama.cpp/ggml-metal.metal")
		if err != nil {
			if err == payloadMissing {
				// TODO perhaps consider this a hard failure on arm macs?
				log.Printf("ggml-meta.metal payload missing")
				return nil
			}
			return err
		}
		os.Setenv("GGML_METAL_PATH_RESOURCES", workdir)
	}

	libs, err := extractDynamicLibs(workdir, "llama.cpp/build/*/*/lib/*")
	if err != nil {
		if err == payloadMissing {
			log.Printf("%s", payloadMissing)
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
	log.Printf("Dynamic LLM libraries %v", variants)
	log.Printf("Override detection logic by setting OLLAMA_LLM_LIBRARY")

	return nil
}

func extractDynamicLibs(workDir, glob string) ([]string, error) {
	files, err := fs.Glob(libEmbed, glob)
	if err != nil || len(files) == 0 {
		return nil, payloadMissing
	}
	libs := []string{}

	for _, file := range files {
		pathComps := strings.Split(file, "/")
		if len(pathComps) != pathComponentCount {
			log.Printf("unexpected payload components: %v", pathComps)
			continue
		}
		// llama.cpp/build/$OS/$VARIANT/lib/$LIBRARY
		// Include the variant in the path to avoid conflicts between multiple server libs
		targetDir := filepath.Join(workDir, pathComps[pathComponentCount-3])
		srcFile, err := libEmbed.Open(file)
		if err != nil {
			return nil, fmt.Errorf("read payload %s: %v", file, err)
		}
		defer srcFile.Close()
		if err := os.MkdirAll(targetDir, 0o755); err != nil {
			return nil, fmt.Errorf("create payload temp dir %s: %v", workDir, err)
		}

		destFile := filepath.Join(targetDir, filepath.Base(file))
		if strings.Contains(destFile, "server") {
			libs = append(libs, destFile)
		}

		_, err = os.Stat(destFile)
		switch {
		case errors.Is(err, os.ErrNotExist):
			destFile, err := os.OpenFile(destFile, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o755)
			if err != nil {
				return nil, fmt.Errorf("write payload %s: %v", file, err)
			}
			defer destFile.Close()
			if _, err := io.Copy(destFile, srcFile); err != nil {
				return nil, fmt.Errorf("copy payload %s: %v", file, err)
			}
		case err != nil:
			return nil, fmt.Errorf("stat payload %s: %v", file, err)
		}
	}
	return libs, nil
}

func extractPayloadFiles(workDir, glob string) error {
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
		if err := os.MkdirAll(workDir, 0o755); err != nil {
			return fmt.Errorf("create payload temp dir %s: %v", workDir, err)
		}

		destFile := filepath.Join(workDir, filepath.Base(file))
		_, err = os.Stat(destFile)
		switch {
		case errors.Is(err, os.ErrNotExist):
			destFile, err := os.OpenFile(destFile, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o755)
			if err != nil {
				return fmt.Errorf("write payload %s: %v", file, err)
			}
			defer destFile.Close()
			if _, err := io.Copy(destFile, srcFile); err != nil {
				return fmt.Errorf("copy payload %s: %v", file, err)
			}
		case err != nil:
			return fmt.Errorf("stat payload %s: %v", file, err)
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
