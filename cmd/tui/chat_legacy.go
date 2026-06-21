package tui

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/ollama/ollama/api"
)

type chatShowClient interface {
	Show(context.Context, *api.ShowRequest) (*api.ShowResponse, error)
}

func (m *chatModel) handleLegacySetCommand(input string) (tea.Model, tea.Cmd) {
	args := strings.Fields(input)
	if len(args) < 2 {
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "system", content: legacySetUsage()}))
		return *m, nil
	}

	switch args[1] {
	case "think":
		return m.handleLegacySetThinkCommand(input)
	case "nothink":
		return m.applyThinkValue("off")
	case "verbose":
		return m.handleVerboseCommand("/verbose on")
	case "quiet":
		return m.handleVerboseCommand("/verbose off")
	case "format":
		if len(args) != 3 || args[2] != "json" {
			return m.legacyUsageError("Invalid or missing format. For JSON mode use `/set format json`.")
		}
		m.opts.Format = "json"
		m.refreshContextEstimate()
		m.status = "format json"
		return *m, nil
	case "noformat":
		m.opts.Format = ""
		m.refreshContextEstimate()
		m.status = "format off"
		return *m, nil
	case "parameter":
		if len(args) < 4 {
			m.entries = append(m.entries, newChatEntry(chatEntry{role: "system", content: legacyParameterUsage()}))
			return *m, nil
		}
		params, err := api.FormatParams(map[string][]string{args[2]: args[3:]})
		if err != nil {
			return m.legacyUsageError(fmt.Sprintf("Couldn't set parameter: %v", err))
		}
		if m.opts.Options == nil {
			m.opts.Options = make(map[string]any)
		}
		for key, value := range params {
			m.opts.Options[key] = value
		}
		m.refreshContextEstimate()
		m.status = "parameter " + args[2]
		return *m, nil
	default:
		return m.legacyUsageError(fmt.Sprintf("Unknown command `/set %s`.", args[1]))
	}
}

func (m *chatModel) handleLegacyLoadCommand(input string) (tea.Model, tea.Cmd) {
	args := strings.Fields(input)
	if len(args) != 2 {
		return m.legacyUsageError("Usage: `/load <model>`")
	}
	previous := m.opts.Model
	if err := m.applyModelSelection(args[1], true); err != nil {
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "error", content: fmt.Sprintf("Could not switch model: %v", err), err: err.Error()}))
		m.status = "error"
		return *m, nil
	}
	if args[1] == previous {
		m.status = "model unchanged"
	} else {
		m.status = "model " + args[1]
	}
	return *m, nil
}

func (m *chatModel) handleLegacyShowCommand(input string) (tea.Model, tea.Cmd) {
	args := strings.Fields(input)
	if len(args) != 2 {
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "system", content: legacyShowUsage()}))
		return *m, nil
	}

	resp, err := m.legacyShowResponse()
	if err != nil {
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "error", content: fmt.Sprintf("Could not show model info: %v", err), err: err.Error()}))
		m.status = "error"
		return *m, nil
	}

	content, ok := m.legacyShowContent(args[1], resp)
	if !ok {
		return m.legacyUsageError(fmt.Sprintf("Unknown command `/show %s`.", args[1]))
	}
	m.entries = append(m.entries, newChatEntry(chatEntry{role: "system", content: content}))
	m.status = "show " + args[1]
	return *m, nil
}

func (m *chatModel) handleLegacyHelpCommand(input string) (tea.Model, tea.Cmd) {
	args := strings.Fields(input)
	if len(args) != 2 {
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "system", content: m.helpSummary()}))
		return *m, nil
	}
	switch strings.TrimPrefix(args[1], "/") {
	case "set":
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "system", content: legacySetUsage()}))
	case "show":
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "system", content: legacyShowUsage()}))
	case "shortcut", "shortcuts":
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "system", content: legacyShortcutUsage()}))
	default:
		m.entries = append(m.entries, newChatEntry(chatEntry{role: "system", content: m.helpSummary()}))
	}
	return *m, nil
}

func (m *chatModel) legacyShowResponse() (*api.ShowResponse, error) {
	client, ok := m.opts.Client.(chatShowClient)
	if !ok || client == nil {
		return nil, fmt.Errorf("model show is unavailable")
	}
	ctx := m.ctx
	if ctx == nil {
		ctx = context.Background()
	}
	return client.Show(ctx, &api.ShowRequest{
		Model:   m.opts.Model,
		Name:    m.opts.Model,
		Options: m.opts.Options,
	})
}

func (m chatModel) legacyShowContent(kind string, resp *api.ShowResponse) (string, bool) {
	switch kind {
	case "info":
		return legacyShowInfo(m.opts.Model, resp), true
	case "license":
		if strings.TrimSpace(resp.License) == "" {
			return "No license was specified for this model.", true
		}
		return resp.License, true
	case "modelfile":
		return resp.Modelfile, true
	case "parameters":
		return legacyShowParameters(resp.Parameters, m.opts.Options), true
	case "system":
		if strings.TrimSpace(resp.System) == "" {
			return "No system message was specified for this model.", true
		}
		return resp.System, true
	case "template":
		if strings.TrimSpace(resp.Template) == "" {
			return "No prompt template was specified for this model.", true
		}
		return resp.Template, true
	default:
		return "", false
	}
}

func legacyShowInfo(modelName string, resp *api.ShowResponse) string {
	var b strings.Builder
	b.WriteString("**Model info**\n\n")
	if strings.TrimSpace(modelName) != "" {
		fmt.Fprintf(&b, "- model: `%s`\n", modelName)
	}
	if !resp.ModifiedAt.IsZero() {
		fmt.Fprintf(&b, "- modified: `%s`\n", resp.ModifiedAt.Format(time.RFC3339))
	}
	if resp.Details.Family != "" {
		fmt.Fprintf(&b, "- family: `%s`\n", resp.Details.Family)
	}
	if resp.Details.ParameterSize != "" {
		fmt.Fprintf(&b, "- parameters: `%s`\n", resp.Details.ParameterSize)
	}
	if resp.Details.QuantizationLevel != "" {
		fmt.Fprintf(&b, "- quantization: `%s`\n", resp.Details.QuantizationLevel)
	}
	if resp.Details.ContextLength > 0 {
		fmt.Fprintf(&b, "- context length: `%d`\n", resp.Details.ContextLength)
	}
	if len(resp.Capabilities) > 0 {
		parts := make([]string, 0, len(resp.Capabilities))
		for _, capability := range resp.Capabilities {
			parts = append(parts, string(capability))
		}
		sort.Strings(parts)
		fmt.Fprintf(&b, "- capabilities: `%s`\n", strings.Join(parts, ", "))
	}
	if b.Len() == len("**Model info**\n\n") {
		b.WriteString("No additional model information was returned.")
	}
	return strings.TrimRight(b.String(), "\n")
}

func legacyShowParameters(modelParameters string, userOptions map[string]any) string {
	var b strings.Builder
	b.WriteString("**Model defined parameters**\n\n")
	if strings.TrimSpace(modelParameters) == "" {
		b.WriteString("No additional parameters were specified for this model.\n")
	} else {
		b.WriteString("```text\n")
		b.WriteString(strings.TrimSpace(modelParameters))
		b.WriteString("\n```\n")
	}
	if len(userOptions) > 0 {
		b.WriteString("\n**User defined parameters**\n\n")
		keys := make([]string, 0, len(userOptions))
		for key := range userOptions {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		for _, key := range keys {
			fmt.Fprintf(&b, "- `%s`: `%v`\n", key, userOptions[key])
		}
	}
	return strings.TrimRight(b.String(), "\n")
}

func (m *chatModel) legacyUsageError(message string) (tea.Model, tea.Cmd) {
	m.entries = append(m.entries, newChatEntry(chatEntry{role: "error", content: message, err: message}))
	m.status = "error"
	return *m, nil
}

func (m *chatModel) refreshContextEstimate() {
	m.contextTokens = m.estimatePromptTokens(m.messages, "")
	m.contextEstimate = true
}

func legacySetUsage() string {
	return strings.Join([]string{
		"**Legacy set commands**",
		"",
		"- `/set parameter <name> <value...>`",
		"- `/set format json`",
		"- `/set noformat`",
		"- `/set verbose`",
		"- `/set quiet`",
		"- `/set think [low|medium|high|max]`",
		"- `/set nothink`",
	}, "\n")
}

func legacyParameterUsage() string {
	return strings.Join([]string{
		"**Legacy parameters**",
		"",
		"- `/set parameter seed <int>`",
		"- `/set parameter num_predict <int>`",
		"- `/set parameter top_k <int>`",
		"- `/set parameter top_p <float>`",
		"- `/set parameter min_p <float>`",
		"- `/set parameter num_ctx <int>`",
		"- `/set parameter temperature <float>`",
		"- `/set parameter repeat_penalty <float>`",
		"- `/set parameter repeat_last_n <int>`",
		"- `/set parameter num_gpu <int>`",
		"- `/set parameter stop <string> <string> ...`",
	}, "\n")
}

func legacyShowUsage() string {
	return strings.Join([]string{
		"**Legacy show commands**",
		"",
		"- `/show info`",
		"- `/show license`",
		"- `/show modelfile`",
		"- `/show parameters`",
		"- `/show system`",
		"- `/show template`",
	}, "\n")
}

func legacyShortcutUsage() string {
	return strings.Join([]string{
		"**Shortcuts**",
		"",
		"- `ctrl+o`: open tool details",
		"- `shift+enter`: insert a newline",
		"- `shift+tab`: toggle permission mode",
		"- `cmd+backspace`, `option+backspace`, `ctrl+w`: delete previous word",
		"- `ctrl+c`: clear input, cancel current response, or confirm quit",
	}, "\n")
}
