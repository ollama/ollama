package envconfig

import (
	"bufio"
	"os"
	"path/filepath"
	"strings"
)

type Override struct {
	ModelName    string
	NumGPULayers int   // -1 means unset
	TensorSplit  []int // nil means unset
}

// LoadOverride loads overrides for the given model section name (e.g. "llama3.2-vision:90b").
// The INI format is:
//   [model-name:params]
//   tensor-split=<int[,int,...]>
// Note: n-gpu-layers is not read from the file; it is always derived as the sum of tensor-split.
// Returns nil if no file or no matching section.
func LoadOverride(model string) *Override {
	// Resolve config path
	path := OverrideConfigPath()
	if path == "" {
		home, _ := os.UserHomeDir()
		if home == "" {
			return nil
		}
		path = filepath.Join(home, ".ollama.ini")
	}
	f, err := os.Open(path)
	if err != nil {
		return nil
	}
	defer f.Close()

	sectionHdr := "[" + model + "]"
	var inSection bool
	ovr := &Override{ModelName: model, NumGPULayers: -1}

	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") || strings.HasPrefix(line, ";") {
			continue
		}
		if strings.HasPrefix(line, "[") && strings.HasSuffix(line, "]") {
			inSection = (line == sectionHdr)
			continue
		}
		if !inSection {
			continue
		}
		kv := strings.SplitN(line, "=", 2)
		if len(kv) != 2 {
			continue
		}
		k := strings.TrimSpace(strings.ToLower(kv[0]))
		v := strings.TrimSpace(kv[1])

		switch k {
		case "tensor-split":
			if arr := parseUintList(v); len(arr) > 0 {
				ovr.TensorSplit = arr
			}
		}
	}

	// If a tensor-split is provided, NumGPULayers is always the sum of entries.
	if len(ovr.TensorSplit) > 0 {
		total := 0
		for _, n := range ovr.TensorSplit {
			total += n
		}
		ovr.NumGPULayers = total
	}

	// If nothing set, return nil
	if ovr.NumGPULayers < 0 && len(ovr.TensorSplit) == 0 {
		return nil
	}
	return ovr
}

func parseUint(s string) int {
	s = strings.TrimSpace(s)
	if s == "" {
		return -1
	}
	var n int
	for _, r := range s {
		if r < '0' || r > '9' {
			return -1
		}
		n = n*10 + int(r-'0')
	}
	return n
}

func parseUintList(s string) []int {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil
	}
	parts := strings.Split(s, ",")
	out := make([]int, 0, len(parts))
	for _, p := range parts {
		n := parseUint(strings.TrimSpace(p))
		if n < 0 {
			return nil
		}
		out = append(out, n)
	}
	return out
}
