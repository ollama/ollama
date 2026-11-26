package launch

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/ollama/ollama/cmd/config"
	"github.com/ollama/ollama/internal/modelref"
	"github.com/ollama/ollama/types/model"
	"golang.org/x/mod/semver"
)

const openCodePluginHeader = "// Managed by ollama launch opencode."
const openCodePluginID = "ollama.integration"
const openCodePluginSource = "git+https://github.com/ollama/opencode-ollama.git#main"
const openCodeStartupEnv = "OLLAMA_OPENCODE_STARTUP"

type openCodeStartup struct {
	Model            string `json:"model"`
	ModelWarning     string `json:"modelWarning,omitempty"`
	CheckModelAccess bool   `json:"checkModelAccess,omitempty"`
}

func canLaunchOpenCodePlugin(req IntegrationLaunchRequest) bool {
	return !req.ForceConfigure && !req.ConfigureOnly && !req.Restore &&
		(len(req.ExtraArgs) == 0 || isOpenCodeResume(req.ExtraArgs)) && os.Getenv("OLLAMA_HOST") == ""
}

func isOpenCodeResume(args []string) bool {
	switch len(args) {
	case 1:
		id, hasID := strings.CutPrefix(args[0], "--session=")
		return args[0] == "--continue" || args[0] == "-c" || (hasID && id != "")
	case 2:
		return (args[0] == "--session" || args[0] == "-s") && args[1] != "" && !strings.HasPrefix(args[1], "-")
	}
	return false
}

func launchOpenCodeWithPlugin(ctx context.Context, req IntegrationLaunchRequest, modelReady bool) (bool, error) {
	path, err := ensureOpenCodeInstalled()
	if err != nil {
		return true, err
	}

	version := openCodeVersion(path)
	if semver.Major(version) != "v2" || semver.Compare(version, "v2.4.1") < 0 {
		return false, nil
	}
	if req.ModelOverride != "" && len(req.ExtraArgs) > 0 {
		return true, errors.New("omit --model to resume an OpenCode chat with its saved model")
	}
	if err := ensureOpenCodePlugin(ctx, path); err != nil {
		return true, fmt.Errorf("prepare Ollama OpenCode plugin: %w", err)
	}
	if req.ModelOverride == "" {
		return true, runOpenCode(path, req.ExtraArgs, "")
	}
	if !modelReady {
		startup, err := openCodeStartupPayload(req.ModelOverride)
		if err != nil {
			return true, err
		}
		return true, runOpenCode(path, nil, "", openCodeStartupEnv+"="+string(startup))
	}

	session, err := createOpenCodeStartupSession(ctx, path, req.ModelOverride)
	if err != nil {
		return true, err
	}
	if session == "" {
		// Keep passing an inline catalog for models OpenCode cannot discover yet.
		return false, nil
	}
	return true, runOpenCode(path, []string{"--session", session}, "")
}

func openCodeStartupPayload(requested string) ([]byte, error) {
	pullName, cloud, err := modelref.NormalizePullName(requested)
	if err != nil {
		return nil, err
	}
	name := model.ParseName(pullName)
	if !name.IsValid() {
		return nil, fmt.Errorf("invalid Ollama model name %q", requested)
	}
	startup := openCodeStartup{
		Model:        name.DisplayShortest(),
		ModelWarning: deprecatedLaunchModelPrompt(requested, "OpenCode", "", "", ""),
	}
	if cloud {
		saved, _ := config.LoadIntegration("opencode")
		startup.CheckModelAccess = !slices.Contains(editorPreCheckedModels(saved, ""), requested)
	}
	return json.Marshal(startup)
}

func createOpenCodeStartupSession(ctx context.Context, path, requestedModel string) (string, error) {
	ctx, cancel := context.WithTimeout(ctx, 20*time.Second)
	defer cancel()
	model, err := waitForOpenCodeModel(ctx, path, requestedModel)
	if err != nil || model == "" {
		return "", err
	}
	directory, err := os.Getwd()
	if err != nil {
		return "", err
	}
	payload, err := json.Marshal(map[string]map[string]string{
		"model":    {"providerID": "ollama", "id": model},
		"location": {"directory": directory},
	})
	if err != nil {
		return "", err
	}
	// Use OpenCode's CLI to authenticate to the same service the TUI will use.
	output, err := openCodeCommandOutput(ctx, path, "api", "POST", "/api/session", "--data", string(payload))
	if err != nil {
		return "", fmt.Errorf("create OpenCode chat for %s: %w", model, err)
	}
	var response struct {
		Data struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.Unmarshal(output, &response); err != nil || response.Data.ID == "" {
		return "", errors.New("OpenCode did not return a session ID for the new chat")
	}
	return response.Data.ID, nil
}

func waitForOpenCodeModel(ctx context.Context, path, requested string) (string, error) {
	// Native discovery may still be running after the first catalog read.
	until := time.Now().Add(3 * time.Second)
	for {
		output, err := openCodeCommandOutput(ctx, path, "models")
		if err != nil {
			return "", fmt.Errorf("list OpenCode models: %w", err)
		}
		if model := openCodeStartupModel(string(output), requested); model != "" {
			return model, nil
		}
		if time.Now().After(until) {
			return "", nil
		}
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		case <-time.After(250 * time.Millisecond):
		}
	}
}

func openCodeStartupModel(catalog, requested string) string {
	var fallback string
	for _, entry := range strings.Fields(catalog) {
		model, ok := strings.CutPrefix(entry, "ollama/")
		if !ok {
			continue
		}
		if model == requested {
			return model
		}
		if launchModelMatches(model, requested) {
			fallback = model
		}
	}
	return fallback
}

func openCodeCommandOutput(ctx context.Context, path string, args ...string) ([]byte, error) {
	cmd := exec.CommandContext(ctx, path, args...)
	// An npm launcher can leave a child holding the output pipe when canceled.
	cmd.WaitDelay = time.Second
	data, err := cmd.Output()
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}
	return data, err
}

func ensureOpenCodePlugin(ctx context.Context, path string) (err error) {
	output, err := openCodeCommandOutput(ctx, path, "plugin", "list")
	if err != nil {
		var failure *exec.ExitError
		if errors.As(err, &failure) && len(failure.Stderr) > 0 {
			return fmt.Errorf("list OpenCode plugins: %w: %s", err, strings.TrimSpace(string(failure.Stderr)))
		}
		return fmt.Errorf("list OpenCode plugins: %w", err)
	}

	legacy, err := managedOpenCodeLegacyPlugin()
	if err != nil {
		return err
	}
	if openCodePluginInstalled(string(output), legacy) {
		return nil
	}
	if legacy != "" {
		backup, archiveErr := archiveOpenCodeLegacyPlugin(legacy)
		if archiveErr != nil {
			return archiveErr
		}
		defer func() {
			if err != nil {
				err = errors.Join(err, os.Rename(backup, legacy))
			}
		}()
	}

	cmd := exec.CommandContext(ctx, path, "plugin", "add", openCodePluginSource)
	cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, os.Stdout, os.Stderr
	cmd.WaitDelay = time.Second
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("install %s: %w", openCodePluginSource, err)
	}
	return nil
}

func openCodePluginInstalled(output, legacy string) bool {
	for line := range strings.SplitSeq(output, "\n") {
		id, rest, _ := strings.Cut(line, " ")
		if id != openCodePluginID {
			continue
		}
		version, source, ok := strings.Cut(strings.TrimSpace(rest), " ")
		if !ok {
			continue
		}
		if version != "local" || legacy == "" {
			return true
		}
		// The original server-only prototype may have used this same ID.
		listedPath, err := filepath.EvalSymlinks(strings.TrimSpace(source))
		if err != nil || listedPath != legacy {
			return true
		}
	}
	return false
}

func managedOpenCodeLegacyPlugin() (string, error) {
	configDir := os.Getenv("OPENCODE_CONFIG_DIR")
	if configDir == "" {
		root := os.Getenv("XDG_CONFIG_HOME")
		if root == "" {
			home, err := os.UserHomeDir()
			if err != nil {
				return "", err
			}
			root = filepath.Join(home, ".config")
		}
		configDir = filepath.Join(root, "opencode")
	}
	path := filepath.Join(configDir, "plugins", "ollama-thinking.js")
	info, err := os.Lstat(path)
	if errors.Is(err, os.ErrNotExist) {
		return "", nil
	}
	if err != nil {
		return "", err
	}
	if !info.Mode().IsRegular() {
		return "", nil
	}
	current, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	if !bytes.HasPrefix(current, []byte(openCodePluginHeader+"\n")) {
		return "", nil
	}
	path, err = filepath.Abs(path)
	if err != nil {
		return "", err
	}
	return filepath.EvalSymlinks(path)
}

func archiveOpenCodeLegacyPlugin(path string) (string, error) {
	backup, err := os.CreateTemp(filepath.Dir(path), ".ollama-thinking.*.bak")
	if err != nil {
		return "", err
	}
	name := backup.Name()
	if err := backup.Close(); err != nil {
		_ = os.Remove(name)
		return "", err
	}
	if err := os.Rename(path, name); err != nil {
		_ = os.Remove(name)
		return "", err
	}
	return name, nil
}
