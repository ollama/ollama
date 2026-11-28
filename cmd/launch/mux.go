package launch

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/ollama/ollama/cmd/internal/fileutil"
	"github.com/ollama/ollama/envconfig"
)

const muxEndpointName = "ollama-launch"

// Mux implements Runner and Editor for the Mux coding agent.
type Mux struct{}

func (m *Mux) String() string { return "Mux" }

func muxEndpointsPath() (string, error) {
	if dir := os.Getenv("MUX_CONFIG_DIR"); dir != "" {
		return filepath.Join(dir, "endpoints.json"), nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".mux", "endpoints.json"), nil
}

func muxLaunchArgs(model string, extra []string) []string {
	args := []string{"--endpoint", muxEndpointName, "--adapter-type", "ollama", "--base-url", envconfig.ConnectableHost().String(), "--model", model}
	return append(args, extra...)
}

func (m *Mux) Run(model string, _ []LaunchModel, args []string) error {
	bin, err := exec.LookPath("mux")
	if err != nil {
		return fmt.Errorf("mux is not installed, install from https://github.com/jchristn/Mux#installation")
	}
	cmd := exec.Command(bin, muxLaunchArgs(model, args)...)
	cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, os.Stdout, os.Stderr
	return cmd.Run()
}

func (m *Mux) Paths() []string {
	path, err := muxEndpointsPath()
	if err == nil {
		if _, err := os.Stat(path); err == nil {
			return []string{path}
		}
	}
	return nil
}

func readMuxEndpoints(path string) (map[string]json.RawMessage, []map[string]json.RawMessage, error) {
	root := map[string]json.RawMessage{}
	data, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return root, nil, nil
	}
	if err != nil {
		return nil, nil, err
	}
	if err := json.Unmarshal(data, &root); err != nil {
		return nil, nil, fmt.Errorf("failed to parse Mux endpoints at %s: %w", path, err)
	}
	if root == nil {
		return nil, nil, fmt.Errorf("Mux endpoints at %s must be an object", path)
	}
	var endpoints []map[string]json.RawMessage
	if raw, ok := root["endpoints"]; ok {
		if err := json.Unmarshal(raw, &endpoints); err != nil {
			return nil, nil, fmt.Errorf("invalid Mux endpoints at %s: %w", path, err)
		}
	}
	return root, endpoints, nil
}

func (m *Mux) Edit(models []LaunchModel) error {
	if len(models) == 0 {
		return nil
	}
	path, err := muxEndpointsPath()
	if err != nil {
		return err
	}
	root, endpoints, err := readMuxEndpoints(path)
	if err != nil {
		return err
	}
	// Replace only the launcher-owned endpoint, including any stale credentials.
	kept := make([]map[string]json.RawMessage, 0, len(endpoints)+1)
	for _, endpoint := range endpoints {
		var name string
		json.Unmarshal(endpoint["name"], &name)
		if !strings.EqualFold(name, muxEndpointName) {
			kept = append(kept, endpoint)
		}
	}
	endpoint := map[string]any{"name": muxEndpointName, "adapterType": "ollama", "baseUrl": envconfig.ConnectableHost().String(), "model": models[0].Name, "authPlacement": "bearer", "isDefault": false}
	if models[0].ContextLength > 0 {
		endpoint["contextWindow"] = models[0].ContextLength
	}
	data, err := json.Marshal(endpoint)
	if err != nil {
		return err
	}
	var entry map[string]json.RawMessage
	if err := json.Unmarshal(data, &entry); err != nil {
		return err
	}
	kept = append(kept, entry)
	root["endpoints"], err = json.Marshal(kept)
	if err != nil {
		return err
	}
	data, err = json.MarshalIndent(root, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	return fileutil.WriteWithBackup(path, data, "mux")
}

func (m *Mux) Models() []string {
	path, err := muxEndpointsPath()
	if err != nil {
		return nil
	}
	_, endpoints, err := readMuxEndpoints(path)
	if err != nil {
		return nil
	}
	for _, endpoint := range endpoints {
		var name, model string
		json.Unmarshal(endpoint["name"], &name)
		json.Unmarshal(endpoint["model"], &model)
		if strings.EqualFold(name, muxEndpointName) && model != "" {
			return []string{model}
		}
	}
	return nil
}
