package launch

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/ollama/ollama/model/renderers"
	"github.com/ollama/ollama/openai"
)

func warnCodexThinking(model LaunchModel, extra []string) {
	if !model.Thinking.Valid() {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	// This is a diagnostic: an unavailable config API must not prevent launch.
	if effort, err := codexConfiguredThinking(ctx, extra); err == nil {
		fmt.Fprint(os.Stderr, codexThinkingWarning(model, effort))
	}
}

func codexThinkingWarning(model LaunchModel, effort string) string {
	if effort == "" || !model.Thinking.Valid() {
		return ""
	}
	converted, err := openai.ThinkingFromReasoningEffort(effort, model.Thinking)
	if err != nil || converted == nil {
		return ""
	}
	// Boolean thinking aliases already have defined on/off semantics. Only warn
	// when a named level falls back to a different model-defined value.
	level, named := converted.Value.(string)
	if !named || model.Thinking.Supports(level) {
		return ""
	}
	entry := buildCodexModelEntry(model)
	var levels []string
	for _, level := range entry["supported_reasoning_levels"].([]any) {
		levels = append(levels, level.(map[string]any)["effort"].(string))
	}
	if len(levels) == 0 {
		return ""
	}
	// Codex normalizes UI aliases before sending an ordinary inference request.
	// The generated catalog has no multi_agent_reasoning_effort override, so
	// ultra selects max, or the last non-ultra catalog level when max is absent.
	wireEffort := effort
	switch effort {
	case "ultra":
		wireEffort = "medium"
		for _, level := range levels {
			if level != "ultra" {
				wireEffort = level
			}
			if level == "max" {
				break
			}
		}
	case "persistent":
		wireEffort = "disabled"
	}
	converted, err = openai.ThinkingFromReasoningEffort(wireEffort, model.Thinking)
	if err != nil || converted == nil {
		return ""
	}
	resolved := renderers.ResolveThinking(converted, model.Thinking)
	used, ok := codexAppThinkingLevelForOllamaValue(resolved.Value)
	if !ok {
		return ""
	}
	return fmt.Sprintf("Warning: thinking level %q isn't supported by %s.\nOllama will use %q. Supported levels: %s.\n", effort, model.Name, used, strings.Join(levels, ", "))
}

// codexConfigReadOptions keeps config overrides in their original order and
// resolves the working directory used by Codex's project config loader.
func codexConfigReadOptions(extra []string) (args []string, cwd string, ok bool) {
	commandPosition, execCommand := true, false
	for i := 0; i < len(extra); i++ {
		arg := extra[i]
		switch {
		case arg == "--":
			return args, cwd, true
		case commandPosition && (arg == "exec" || arg == "e") && !execCommand:
			execCommand = true
		case commandPosition && (arg == "help" || !execCommand && codexNonSessionCommand(arg)):
			return nil, "", false
		case commandPosition && (arg == "resume" || arg == "fork" || arg == "review"), arg == "--worktree",
			arg == "--ignore-user-config", arg == "--remote", strings.HasPrefix(arg, "--remote="),
			arg == "--oss", arg == "--local-provider", strings.HasPrefix(arg, "--local-provider="),
			arg == "-h", arg == "--help", arg == "-V", arg == "--version":
			// These can use saved, remote, or otherwise different configuration.
			return nil, "", false
		case arg == "-c", arg == "--config", arg == "--enable", arg == "--disable":
			if i+1 == len(extra) {
				return nil, "", false
			}
			i++
			args = append(args, arg, extra[i])
		case strings.HasPrefix(arg, "--config="), strings.HasPrefix(arg, "-c"),
			strings.HasPrefix(arg, "--enable="), strings.HasPrefix(arg, "--disable="):
			args = append(args, arg)
		case arg == "-m", arg == "--model":
			if i+1 == len(extra) {
				return nil, "", false
			}
			i++
			args = append(args, "-c", fmt.Sprintf("model=%q", extra[i]))
		case arg == "--profile", arg == "-p":
			// app-server cannot select profiles. The generated profile has no
			// thinking setting; its managed config overrides are passed below.
			if i+1 == len(extra) || extra[i+1] != codexProfileName {
				return nil, "", false
			}
			i++
		case arg == "-C", arg == "--cd":
			if i+1 == len(extra) {
				return nil, "", false
			}
			i++
			cwd = extra[i]
		case strings.HasPrefix(arg, "--cd="):
			cwd = strings.TrimPrefix(arg, "--cd=")
		case strings.HasPrefix(arg, "-C"):
			cwd = strings.TrimPrefix(arg, "-C")
		case arg == "-i", arg == "--image":
			if i+1 == len(extra) || strings.HasPrefix(extra[i+1], "-") {
				return nil, "", false
			}
			fallthrough
		case strings.HasPrefix(arg, "--image="), strings.HasPrefix(arg, "-i"):
			// Codex accepts multiple image paths up to the next option.
			for i+1 < len(extra) && !strings.HasPrefix(extra[i+1], "-") {
				i++
			}
		case arg == "--add-dir", arg == "--output-schema", arg == "-o", arg == "--output-last-message",
			arg == "--thread-source", arg == "--cyber-access-program", arg == "--color",
			arg == "-s", arg == "--sandbox", arg == "-a", arg == "--ask-for-approval",
			arg == "--remote-auth-token-env":
			// These values cannot affect thinking, but may look like commands
			// (for example, --output-last-message review).
			if i+1 == len(extra) {
				return nil, "", false
			}
			i++
		case !strings.HasPrefix(arg, "-"):
			commandPosition = false
		}
	}
	return args, cwd, true
}

func codexNonSessionCommand(command string) bool {
	switch command {
	case "agents", "login", "logout", "mcp", "plugin", "app-server", "remote-control", "app",
		"completion", "update", "doctor", "sandbox", "debug", "apply", "a", "queue", "archive",
		"delete", "migrate-rollouts", "unarchive", "cloud", "cloud-tasks", "exec-server", "features",
		"tcp-tunnel", "execpolicy", "responses-api-proxy", "stdio-to-uds":
		return true
	default:
		return false
	}
}

// Ask Codex to resolve its own config layers, including project trust and CLI
// overrides, including the provider/catalog overrides from the real invocation.
// The generated Ollama profile contains no thinking setting, so it does not
// change this value. No thread or model request is created here.
func codexConfiguredThinking(ctx context.Context, extra []string) (string, error) {
	args, cwd, ok := codexConfigReadOptions(extra)
	if !ok {
		return "", nil
	}
	cwd, err := filepath.Abs(cwd)
	if err != nil {
		return "", err
	}
	cmd := exec.CommandContext(ctx, "codex", append([]string{"app-server"}, args...)...)
	cmd.Dir = cwd
	return codexReadThinking(cmd, cwd)
}

func codexReadThinking(cmd *exec.Cmd, cwd string) (string, error) {
	cmd.Stderr = io.Discard
	cmd.WaitDelay = time.Second
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return "", err
	}
	defer stdin.Close()
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return "", err
	}
	if err := cmd.Start(); err != nil {
		return "", err
	}
	defer func() {
		stdin.Close()
		cmd.Wait()
	}()

	encoder := json.NewEncoder(stdin)
	decoder := json.NewDecoder(io.LimitReader(stdout, 16<<20))
	read := func(id int) (string, error) {
		for {
			var response struct {
				ID     *int            `json:"id"`
				Error  json.RawMessage `json:"error"`
				Result struct {
					Config struct {
						Effort string `json:"model_reasoning_effort"`
					} `json:"config"`
				} `json:"result"`
			}
			if err := decoder.Decode(&response); err != nil {
				return "", err
			}
			if response.ID == nil || *response.ID != id {
				continue
			}
			if len(response.Error) > 0 && string(response.Error) != "null" {
				return "", errors.New("could not read Codex configuration")
			}
			return response.Result.Config.Effort, nil
		}
	}
	if err := encoder.Encode(map[string]any{
		"id": 0, "method": "initialize",
		"params": map[string]any{"clientInfo": map[string]string{"name": "ollama", "version": "1"}},
	}); err != nil {
		return "", err
	}
	if _, err := read(0); err != nil {
		return "", err
	}
	if err := encoder.Encode(map[string]any{"method": "initialized"}); err != nil {
		return "", err
	}
	if err := encoder.Encode(map[string]any{
		"id": 1, "method": "config/read", "params": map[string]any{"cwd": cwd},
	}); err != nil {
		return "", err
	}
	return read(1)
}
