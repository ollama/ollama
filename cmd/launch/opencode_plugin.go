package launch

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"slices"
	"strings"
	"time"

	"github.com/ollama/ollama/cmd/config"
	"github.com/ollama/ollama/internal/modelref"
	"github.com/ollama/ollama/types/model"
	"golang.org/x/mod/semver"
)

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
	var env []string
	if req.ModelOverride != "" {
		startup, err := openCodeStartupPayload(req.ModelOverride, modelReady)
		if err != nil {
			return true, err
		}
		env = []string{openCodeStartupEnv + "=" + string(startup)}
	}
	if err := ensureOpenCodePlugin(ctx, path); err != nil {
		return true, fmt.Errorf("prepare Ollama OpenCode plugin: %w", err)
	}
	return true, runOpenCode(path, req.ExtraArgs, "", env...)
}

func openCodeStartupPayload(requested string, modelReady bool) ([]byte, error) {
	pullName, cloud, err := modelref.NormalizePullName(requested)
	if err != nil {
		return nil, err
	}
	name := model.ParseName(pullName)
	if !name.IsValid() {
		return nil, fmt.Errorf("invalid Ollama model name %q", requested)
	}
	startup := openCodeStartup{Model: name.DisplayShortest()}
	if modelReady {
		// The launcher already handled confirmations under --yes.
		return json.Marshal(startup)
	}
	startup.ModelWarning = deprecatedLaunchModelPrompt(requested, "OpenCode", "", "", "")
	if cloud {
		saved, _ := config.LoadIntegration("opencode")
		startup.CheckModelAccess = !slices.Contains(editorPreCheckedModels(saved, ""), requested)
	}
	return json.Marshal(startup)
}

func ensureOpenCodePlugin(ctx context.Context, path string) error {
	list := exec.CommandContext(ctx, path, "plugin", "list")
	// An npm launcher can leave a child holding the output pipe when canceled.
	list.WaitDelay = time.Second
	output, err := list.Output()
	if ctx.Err() != nil {
		return ctx.Err()
	}
	if err != nil {
		var failure *exec.ExitError
		if errors.As(err, &failure) && len(failure.Stderr) > 0 {
			return fmt.Errorf("list OpenCode plugins: %w: %s", err, strings.TrimSpace(string(failure.Stderr)))
		}
		return fmt.Errorf("list OpenCode plugins: %w", err)
	}

	for line := range strings.SplitSeq(string(output), "\n") {
		fields := strings.Fields(line)
		if len(fields) < 3 {
			continue
		}
		// OpenCode can list a resolved package before its plugin ID is registered.
		if fields[0] == openCodePluginID || (fields[0] == "-" && fields[1] != "-" && fields[2] == openCodePluginSource) {
			return nil
		}
	}

	cmd := exec.CommandContext(ctx, path, "plugin", "add", openCodePluginSource)
	cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, os.Stdout, os.Stderr
	cmd.WaitDelay = time.Second
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("install %s: %w", openCodePluginSource, err)
	}
	return nil
}
