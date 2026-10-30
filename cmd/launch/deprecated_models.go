package launch

import (
	"fmt"
	"strings"

	"github.com/ollama/ollama/internal/modelref"
)

var deprecatedLaunchModels = map[string]struct{}{
	"codellama":     {},
	"qwen2.5":       {},
	"qwen2.5-coder": {},
	"llama3":        {},
	"llama3.1":      {},
	"llama3.2":      {},
	"llama3.3":      {},
	"mistral":       {},
	"starcoder":     {},
}

var deprecatedLaunchModelTags = map[string]map[string]struct{}{
	"deepseek-r1": {
		"":       {},
		"latest": {},
		"1.5b":   {},
		"7b":     {},
		"8b":     {},
		"14b":    {},
		"32b":    {},
	},
}

var errDeprecatedLaunchModelDeclined = fmt.Errorf("%w: deprecated launch model declined", ErrCancelled)

func isDeprecatedLaunchModel(name string) bool {
	family, tag := normalizedLaunchModelRef(name)
	if _, ok := deprecatedLaunchModels[family]; ok {
		return true
	}
	tags, ok := deprecatedLaunchModelTags[family]
	if !ok {
		return false
	}
	_, ok = tags[tag]
	return ok
}

func deprecatedLaunchModelPrompt(name, label, commandName, cloudRec, localRec string) string {
	if !isDeprecatedLaunchModel(name) {
		return ""
	}
	if label = strings.TrimSpace(label); label == "" {
		label = "ollama launch"
	}

	var b strings.Builder
	fmt.Fprintf(&b, "%s does not work well with %s. ", name, label)
	switch {
	case cloudRec != "" && localRec != "":
		fmt.Fprintf(&b, "Try an agent-capable model like %s or %s instead", cloudRec, localRec)
	case cloudRec != "":
		fmt.Fprintf(&b, "Try an agent-capable model like %s instead", cloudRec)
	case localRec != "":
		fmt.Fprintf(&b, "Try an agent-capable model like %s instead", localRec)
	default:
		b.WriteString("Try a newer recommended agent-capable model instead")
	}
	if command := launchReplacementCommand(commandName, firstNonEmpty(cloudRec, localRec)); command != "" {
		fmt.Fprintf(&b, ":\n  %s", command)
	} else {
		b.WriteString(".")
	}
	fmt.Fprintf(&b, "\n\nLaunch with %s anyway?", name)
	return b.String()
}

func launchReplacementCommand(commandName, model string) string {
	commandName = strings.TrimSpace(commandName)
	model = strings.TrimSpace(model)
	if commandName == "" || model == "" {
		return ""
	}
	return fmt.Sprintf("ollama launch %s --model %s", commandName, model)
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}

func normalizedLaunchModelRef(name string) (string, string) {
	name = strings.TrimSpace(strings.ToLower(name))
	if name == "" {
		return "", ""
	}
	if base, stripped := modelref.StripCloudSourceTag(name); stripped {
		name = base
	}
	if idx := strings.LastIndex(name, "/"); idx >= 0 {
		name = name[idx+1:]
	}
	tag := ""
	if idx := strings.Index(name, ":"); idx >= 0 {
		tag = strings.TrimSpace(name[idx+1:])
		name = name[:idx]
	}
	return strings.TrimSpace(name), tag
}

func filterDeprecatedLaunchModelItems(items []ModelItem) []ModelItem {
	filtered := items[:0]
	for _, item := range items {
		if !isDeprecatedLaunchModel(item.Name) {
			filtered = append(filtered, item)
		}
	}
	return filtered
}

func filterDeprecatedLaunchModelNames(models []string) []string {
	filtered := models[:0]
	for _, model := range models {
		if !isDeprecatedLaunchModel(model) {
			filtered = append(filtered, model)
		}
	}
	return filtered
}
