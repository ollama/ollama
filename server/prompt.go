package server

import (
	"fmt"
	"log/slog"
	"strings"
	"text/template"
	"text/template/parse"

	"github.com/ollama/ollama/api"
)

// isResponseNode checks if the node contains .Response
func isResponseNode(node *parse.ActionNode) bool {
	for _, cmd := range node.Pipe.Cmds {
		for _, arg := range cmd.Args {
			if fieldNode, ok := arg.(*parse.FieldNode); ok && len(fieldNode.Ident) > 0 {
				if fieldNode.Ident[0] == "Response" {
					return true
				}
			}
		}
	}
	return false
}

// formatTemplateForResponse formats the template AST to:
// 1. remove all nodes after the first .Response (if generate=true)
// 2. add a .Response node to the end if it doesn't exist
// TODO(jmorganca): this should recursively cut the template before the first .Response
func formatTemplateForResponse(tmpl *template.Template, generate bool) {
	var found bool
	for i, node := range tmpl.Tree.Root.Nodes {
		if actionNode, ok := node.(*parse.ActionNode); ok {
			if isResponseNode(actionNode) {
				found = true
				if generate {
					tmpl.Tree.Root.Nodes = tmpl.Tree.Root.Nodes[:i+1]
					break
				}
			}
		}
	}

	if !found {
		// add the response node if it doesn't exist
		responseFieldNode := &parse.FieldNode{NodeType: parse.NodeField, Ident: []string{"Response"}}
		responsePipeNode := &parse.PipeNode{NodeType: parse.NodePipe, Cmds: []*parse.CommandNode{{NodeType: parse.NodeCommand, Args: []parse.Node{responseFieldNode}}}}
		responseActionNode := &parse.ActionNode{NodeType: parse.NodeAction, Pipe: responsePipeNode}
		tmpl.Tree.Root.Nodes = append(tmpl.Tree.Root.Nodes, responseActionNode)
	}
}

// Prompt renders a prompt from a template. If generate is set to true,
// the response and parts of the template following it are not rendered
func Prompt(tmpl, tools, system, prompt, results string, response string, generate bool) (string, error) {
	parsed, err := template.New("").Option("missingkey=zero").Parse(tmpl)
	if err != nil {
		return "", err
	}

	formatTemplateForResponse(parsed, generate)

	vars := map[string]any{
		"Tools":    tools,
		"System":   system,
		"Prompt":   prompt,
		"Results":  results,
		"Response": response,
	}

	var sb strings.Builder
	if err := parsed.Execute(&sb, vars); err != nil {
		return "", err
	}

	return sb.String(), nil
}

func countTokens(tmpl string, tools string, system string, prompt string, results string, response string, encode func(string) ([]int, error)) (int, error) {
	rendered, err := Prompt(tmpl, tools, system, prompt, results, response, false)
	if err != nil {
		return 0, err
	}

	tokens, err := encode(rendered)
	if err != nil {
		slog.Error("failed to encode prompt", "err", err)
		return 0, err
	}

	return len(tokens), err
}

// ChatPrompt builds up a prompt from a series of messages, truncating based on context window size
func ChatPrompt(tmpl string, messages []api.Message, tools string, window int, encode func(string) ([]int, error)) (string, error) {
	type prompt struct {
		Tools    string
		System   string
		Prompt   string
		Results  string
		Response string

		images []int
		tokens int
	}

	slog.Debug("Available tools", "tools", tools)

	var p prompt

	// iterate through messages to build up {system,user,response} prompts
	var imgId int
	var prompts []prompt
	for _, msg := range messages {
		switch strings.ToLower(msg.Role) {
		case "system":
			if p.System != "" || p.Prompt != "" || p.Response != "" {
				prompts = append(prompts, p)
				p = prompt{}
			}

			p.System = msg.Content
		case "user":
			if p.Prompt != "" || p.Response != "" {
				prompts = append(prompts, p)
				p = prompt{}
			}

			var sb strings.Builder
			for range msg.Images {
				fmt.Fprintf(&sb, "[img-%d] ", imgId)
				p.images = append(p.images, imgId)
				imgId += 1
			}

			sb.WriteString(msg.Content)
			p.Prompt = sb.String()
		case "tool":
			if p.Results != "" {
				prompts = append(prompts, p)
				p = prompt{}
			}

			p.Results = msg.Content
		case "assistant":
			if p.Response != "" {
				prompts = append(prompts, p)
				p = prompt{}
			}

			p.Response = msg.Content
		default:
			return "", fmt.Errorf("invalid role: %s, role must be one of [system, user, tool, assistant]", msg.Role)
		}
	}

	// add final prompt
	if p.Tools != "" || p.System != "" || p.Prompt != "" || p.Results != "" || p.Response != "" {
		prompts = append(prompts, p)
	}

	// add available tools to the first prompt
	if len(prompts) > 0 {
		prompts[0].Tools = tools
	}

	// calculate token lengths for each prompt, estimating 768 tokens per images
	for i, p := range prompts {
		tokens, err := countTokens(tmpl, p.Tools, p.System, p.Prompt, p.Results, p.Response, encode)
		if err != nil {
			return "", err
		}

		prompts[i].tokens = tokens + len(prompts[i].images)*768
	}

	// truncate images and prompts starting from the beginning of the list
	// until either one prompt remains or the total tokens fits the context window
	// TODO (jmorganca): this doesn't account for the context window room required for the response
	for {
		var required int
		for _, p := range prompts {
			required += p.tokens
		}

		required += 1 // for bos token

		if required <= window {
			slog.Debug("prompt now fits in context window", "required", required, "window", window)
			break
		}

		prompt := &prompts[0]

		if len(prompt.images) > 1 {
			img := prompt.images[0]
			slog.Debug("prompt longer than context window, removing image", "id", img, "required", required, "window", window)
			prompt.images = prompt.images[1:]
			prompt.Prompt = strings.Replace(prompt.Prompt, fmt.Sprintf(" [img-%d]", img), "", 1)
			prompt.tokens -= 768
			continue
		}

		if len(prompts) > 1 {
			slog.Debug("required tokens longer than context window, removing first prompt", "prompt", prompts[0].tokens, "required", required, "window", window)
			system := prompt.System
			prompts = prompts[1:]

			if system != "" && prompts[0].System == "" {
				prompts[0].System = system

				tokens, err := countTokens(tmpl, prompts[0].Tools, prompts[0].System, prompts[0].Prompt, prompts[0].Results, prompts[0].Response, encode)
				if err != nil {
					return "", err
				}

				prompts[0].tokens = tokens + len(prompts[0].images)*768
			}

			continue
		}

		// stop truncating if there's only one prompt left
		break
	}

	var sb strings.Builder
	for i, p := range prompts {
		// last prompt should leave the response unrendered (for completion)
		rendered, err := Prompt(tmpl, p.Tools, p.System, p.Prompt, p.Results, p.Response, i == len(prompts)-1)
		if err != nil {
			return "", err
		}
		sb.WriteString(rendered)
	}

	return sb.String(), nil
}
