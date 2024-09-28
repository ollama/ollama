package runners

import (
	"encoding/json"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"

	"golang.org/x/sys/cpu"

	"github.com/ollama/ollama/gpu"
)

type RunnerRequirements struct {
	SystemInfo  string   `json:"system_info"`
	Version     string   `json:"version"`
	CPUFeatures []string `json:"cpu_features"`
}

func GatherRequirements(exe string) (RunnerRequirements, error) {
	var requirements RunnerRequirements
	// TODO - DRY this out with server.go
	pathEnv := "LD_LIBRARY_PATH"
	if runtime.GOOS == "windows" {
		pathEnv = "PATH"
	}
	// Start with the server directory for the LD_LIBRARY_PATH/PATH
	libraryPaths := []string{filepath.Dir(exe), gpu.LibraryDir()}
	if libraryPath, ok := os.LookupEnv(pathEnv); ok {
		// favor our bundled library dependencies over system libraries
		libraryPaths = append(libraryPaths, filepath.SplitList(libraryPath)...)
	}

	cmd := exec.Command(exe, "--requirements")
	cmd.Env = append(os.Environ(), pathEnv+"="+strings.Join(libraryPaths, string(filepath.ListSeparator)))
	buf, err := cmd.CombinedOutput()
	if err != nil {
		return requirements, err
	}
	err = json.Unmarshal(buf, &requirements)
	return requirements, err
}

func IsCompatible(feature string) bool {
	switch feature {
	case "avx":
		return cpu.X86.HasAVX
	case "avx2":
		return cpu.X86.HasAVX2
	case "avx512f":
		return cpu.X86.HasAVX512F
	case "avx512bw":
		return cpu.X86.HasAVX512BW
	case "avx512vbmi":
		return cpu.X86.HasAVX512VBMI
	case "avx512vnni":
		return cpu.X86.HasAVX512VNNI
	case "avx512bf16":
		return cpu.X86.HasAVX512BF16
	case "sve":
		return cpu.ARM64.HasSVE
	default:
		slog.Warn("unrecognized cpu feature", "flag", feature)
		return false
	}
}
