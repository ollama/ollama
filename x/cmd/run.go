package cmd

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/url"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/spf13/cobra"
	"golang.org/x/term"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/progress"
	"github.com/ollama/ollama/readline"
	"github.com/ollama/ollama/types/model"
	"github.com/ollama/ollama/x/agent"
	"github.com/ollama/ollama/x/tools"
)

// Tool output capping constants
const (
	// localModelTokenLimit is the token limit for local models (smaller context).
	localModelTokenLimit = 4000

	// defaultTokenLimit is the token limit for cloud/remote models.
	defaultTokenLimit = 10000

	// charsPerToken is a rough estimate of characters per token.
	// TODO: Estimate tokens more accurately using tokenizer if available
	charsPerToken = 4
)

// isLocalModel checks if the model is running locally (not a cloud model).
// TODO: Improve local/cloud model identification - could check model metadata
func isLocalModel(modelName string) bool {
	return !strings.HasSuffix(modelName, "-cloud")
}

// isLocalServer checks if connecting to a local Ollama server.
// TODO: Could also check other indicators of local vs cloud server
func isLocalServer() bool {
	host := os.Getenv("OLLAMA_HOST")
	if host == "" {
		return true // Default is localhost:11434
	}

	// Parse the URL to check host
	parsed, err := url.Parse(host)
	if err != nil {
		return true // If can't parse, assume local
	}

	hostname := parsed.Hostname()
	return hostname == "localhost" || hostname == "127.0.0.1" || strings.Contains(parsed.Host, ":11434")
}

// truncateToolOutput truncates tool output to prevent context overflow.
// Uses a smaller limit (4k tokens) for local models, larger (10k) for cloud/remote.
func truncateToolOutput(output, modelName string) string {
	var tokenLimit int
	if isLocalModel(modelName) && isLocalServer() {
		tokenLimit = localModelTokenLimit
	} else {
		tokenLimit = defaultTokenLimit
	}

	maxChars := tokenLimit * charsPerToken
	if len(output) > maxChars {
		return output[:maxChars] + "\n... (output truncated)"
	}
	return output
}

// waitForOllamaSignin shows the signin URL and polls until authentication completes.
func waitForOllamaSignin(ctx context.Context) error {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}

	// Get signin URL from initial Whoami call
	_, err = client.Whoami(ctx)
	if err != nil {
		var aErr api.AuthorizationError
		if errors.As(err, &aErr) && aErr.SigninURL != "" {
			fmt.Fprintf(os.Stderr, "\n  To sign in, navigate to:\n")
			fmt.Fprintf(os.Stderr, "      %s\n\n", aErr.SigninURL)
			fmt.Fprintf(os.Stderr, "  \033[90mwaiting for sign in to complete...\033[0m")

			// Poll until auth succeeds
			ticker := time.NewTicker(2 * time.Second)
			defer ticker.Stop()

			for {
				select {
				case <-ctx.Done():
					fmt.Fprintf(os.Stderr, "\n")
					return ctx.Err()
				case <-ticker.C:
					user, whoamiErr := client.Whoami(ctx)
					if whoamiErr == nil && user != nil && user.Name != "" {
						fmt.Fprintf(os.Stderr, "\r\033[K\033[A\r\033[K  \033[1msigned in:\033[0m %s\n", user.Name)
						return nil
					}
					// Still waiting, show dot
					fmt.Fprintf(os.Stderr, ".")
				}
			}
		}
		return err
	}
	return nil
}

// RunOptions contains options for running an interactive agent session.
type RunOptions struct {
	Model        string
	Messages     []api.Message
	WordWrap     bool
	Format       string
	System       string
	Options      map[string]any
	KeepAlive    *api.Duration
	Think        *api.ThinkValue
	HideThinking bool

	// Agent fields (managed externally for session persistence)
	Tools    *tools.Registry
	Approval *agent.ApprovalManager

	// YoloMode skips all tool approval prompts
	YoloMode bool
}

// Chat runs an agent chat loop with tool support.
// This is the experimental version of chat that supports tool calling.
func Chat(ctx context.Context, opts RunOptions) (*api.Message, error) {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return nil, err
	}

	// Use tools registry and approval from opts (managed by caller for session persistence)
	toolRegistry := opts.Tools
	approval := opts.Approval
	if approval == nil {
		approval = agent.NewApprovalManager()
	}

	p := progress.NewProgress(os.Stderr)
	defer p.StopAndClear()

	spinner := progress.NewSpinner("")
	p.Add("", spinner)

	cancelCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT)

	go func() {
		<-sigChan
		cancel()
	}()

	var state *displayResponseState = &displayResponseState{}
	var thinkingContent strings.Builder
	var fullResponse strings.Builder
	var thinkTagOpened bool = false
	var thinkTagClosed bool = false
	var pendingToolCalls []api.ToolCall
	var consecutiveErrors int // Track consecutive 500 errors for retry limit

	role := "assistant"
	messages := opts.Messages

	fn := func(response api.ChatResponse) error {
		if response.Message.Content != "" || !opts.HideThinking {
			p.StopAndClear()
		}

		role = response.Message.Role
		if response.Message.Thinking != "" && !opts.HideThinking {
			if !thinkTagOpened {
				fmt.Print(thinkingOutputOpeningText(false))
				thinkTagOpened = true
				thinkTagClosed = false
			}
			thinkingContent.WriteString(response.Message.Thinking)
			displayResponse(response.Message.Thinking, opts.WordWrap, state)
		}

		content := response.Message.Content
		if thinkTagOpened && !thinkTagClosed && (content != "" || len(response.Message.ToolCalls) > 0) {
			if !strings.HasSuffix(thinkingContent.String(), "\n") {
				fmt.Println()
			}
			fmt.Print(thinkingOutputClosingText(false))
			thinkTagOpened = false
			thinkTagClosed = true
			state = &displayResponseState{}
		}

		fullResponse.WriteString(content)

		if response.Message.ToolCalls != nil {
			toolCalls := response.Message.ToolCalls
			if len(toolCalls) > 0 {
				if toolRegistry != nil {
					// Store tool calls for execution after response is complete
					pendingToolCalls = append(pendingToolCalls, toolCalls...)
				} else {
					// No tools registry, just display tool calls
					fmt.Print(renderToolCalls(toolCalls, false))
				}
			}
		}

		displayResponse(content, opts.WordWrap, state)

		return nil
	}

	if opts.Format == "json" {
		opts.Format = `"` + opts.Format + `"`
	}

	// Agentic loop: continue until no more tool calls
	for {
		req := &api.ChatRequest{
			Model:    opts.Model,
			Messages: messages,
			Format:   json.RawMessage(opts.Format),
			Options:  opts.Options,
			Think:    opts.Think,
		}

		// Add tools
		if toolRegistry != nil {
			apiTools := toolRegistry.Tools()
			if len(apiTools) > 0 {
				req.Tools = apiTools
			}
		}

		if opts.KeepAlive != nil {
			req.KeepAlive = opts.KeepAlive
		}

		if err := client.Chat(cancelCtx, req, fn); err != nil {
			if errors.Is(err, context.Canceled) {
				return nil, nil
			}

			// Check for 401 Unauthorized - prompt user to sign in
			var authErr api.AuthorizationError
			if errors.As(err, &authErr) {
				p.StopAndClear()
				fmt.Fprintf(os.Stderr, "\033[1mauth required:\033[0m cloud model requires authentication\n")
				result, promptErr := agent.PromptYesNo("Sign in to Ollama?")
				if promptErr == nil && result {
					if signinErr := waitForOllamaSignin(ctx); signinErr == nil {
						// Retry the chat request
						fmt.Fprintf(os.Stderr, "\033[90mretrying...\033[0m\n")
						continue // Retry the loop
					}
				}
				return nil, fmt.Errorf("authentication required - run 'ollama signin' to authenticate")
			}

			// Check for 500 errors (often tool parsing failures) - inform the model
			var statusErr api.StatusError
			if errors.As(err, &statusErr) && statusErr.StatusCode >= 500 {
				consecutiveErrors++
				p.StopAndClear()

				if consecutiveErrors >= 3 {
					fmt.Fprintf(os.Stderr, "\033[1merror:\033[0m too many consecutive errors, giving up\n")
					return nil, fmt.Errorf("too many consecutive server errors: %s", statusErr.ErrorMessage)
				}

				fmt.Fprintf(os.Stderr, "\033[1mwarning:\033[0m server error (attempt %d/3): %s\n", consecutiveErrors, statusErr.ErrorMessage)

				// Include both the model's response and the error so it can learn
				assistantContent := fullResponse.String()
				if assistantContent == "" {
					assistantContent = "(empty response)"
				}
				errorMsg := fmt.Sprintf("Your previous response caused an error: %s\n\nYour response was:\n%s\n\nPlease try again with a valid response.", statusErr.ErrorMessage, assistantContent)
				messages = append(messages,
					api.Message{Role: "user", Content: errorMsg},
				)

				// Reset state and retry
				fullResponse.Reset()
				thinkingContent.Reset()
				thinkTagOpened = false
				thinkTagClosed = false
				pendingToolCalls = nil
				state = &displayResponseState{}
				p = progress.NewProgress(os.Stderr)
				spinner = progress.NewSpinner("")
				p.Add("", spinner)
				continue
			}

			if strings.Contains(err.Error(), "upstream error") {
				p.StopAndClear()
				fmt.Println("An error occurred while processing your message. Please try again.")
				fmt.Println()
				return nil, nil
			}
			return nil, err
		}

		// Reset consecutive error counter on success
		consecutiveErrors = 0

		// If no tool calls, we're done
		if len(pendingToolCalls) == 0 || toolRegistry == nil {
			break
		}

		// Execute tool calls and continue the conversation
		fmt.Fprintf(os.Stderr, "\n")

		// Add assistant's tool call message to history
		assistantMsg := api.Message{
			Role:      "assistant",
			Content:   fullResponse.String(),
			Thinking:  thinkingContent.String(),
			ToolCalls: pendingToolCalls,
		}
		messages = append(messages, assistantMsg)

		// Execute each tool call and collect results
		var toolResults []api.Message
		for _, call := range pendingToolCalls {
			toolName := call.Function.Name
			args := call.Function.Arguments.ToMap()

			// For bash commands, check denylist first
			skipApproval := false
			if toolName == "bash" {
				if cmd, ok := args["command"].(string); ok {
					// Check if command is denied (dangerous pattern)
					if denied, pattern := agent.IsDenied(cmd); denied {
						fmt.Fprintf(os.Stderr, "\033[1mblocked:\033[0m %s\n", formatToolShort(toolName, args))
						fmt.Fprintf(os.Stderr, "  matches dangerous pattern: %s\n", pattern)
						toolResults = append(toolResults, api.Message{
							Role:       "tool",
							Content:    agent.FormatDeniedResult(cmd, pattern),
							ToolCallID: call.ID,
						})
						continue
					}

					// Check if command is auto-allowed (safe command)
					// TODO(parthsareen): re-enable with tighter scoped allowlist
					// if agent.IsAutoAllowed(cmd) {
					// 	fmt.Fprintf(os.Stderr, "\033[1mauto-allowed:\033[0m %s\n", formatToolShort(toolName, args))
					// 	skipApproval = true
					// }
				}
			}

			// Check approval (uses prefix matching for bash commands)
			// In yolo mode, skip all approval prompts
			if opts.YoloMode {
				if !skipApproval {
					fmt.Fprintf(os.Stderr, "\033[1mrunning:\033[0m %s\n", formatToolShort(toolName, args))
				}
			} else if !skipApproval && !approval.IsAllowed(toolName, args) {
				result, err := approval.RequestApproval(toolName, args)
				if err != nil {
					fmt.Fprintf(os.Stderr, "Error requesting approval: %v\n", err)
					toolResults = append(toolResults, api.Message{
						Role:       "tool",
						Content:    fmt.Sprintf("Error: %v", err),
						ToolCallID: call.ID,
					})
					continue
				}

				// Show collapsed result
				fmt.Fprintln(os.Stderr, agent.FormatApprovalResult(toolName, args, result))

				switch result.Decision {
				case agent.ApprovalDeny:
					toolResults = append(toolResults, api.Message{
						Role:       "tool",
						Content:    agent.FormatDenyResult(toolName, result.DenyReason),
						ToolCallID: call.ID,
					})
					continue
				case agent.ApprovalAlways:
					approval.AddToAllowlist(toolName, args)
				}
			} else if !skipApproval {
				// Already allowed - show running indicator
				fmt.Fprintf(os.Stderr, "\033[1mrunning:\033[0m %s\n", formatToolShort(toolName, args))
			}

			// Execute the tool
			toolResult, err := toolRegistry.Execute(call)
			if err != nil {
				// Check if web search needs authentication
				if errors.Is(err, tools.ErrWebSearchAuthRequired) {
					// Prompt user to sign in
					fmt.Fprintf(os.Stderr, "\033[1mauth required:\033[0m web search requires authentication\n")
					result, promptErr := agent.PromptYesNo("Sign in to Ollama?")
					if promptErr == nil && result {
						// Get signin URL and wait for auth completion
						if signinErr := waitForOllamaSignin(ctx); signinErr == nil {
							// Retry the web search
							fmt.Fprintf(os.Stderr, "\033[90mretrying web search...\033[0m\n")
							toolResult, err = toolRegistry.Execute(call)
							if err == nil {
								goto toolSuccess
							}
						}
					}
				}
				fmt.Fprintf(os.Stderr, "\033[1merror:\033[0m %v\n", err)
				toolResults = append(toolResults, api.Message{
					Role:       "tool",
					Content:    fmt.Sprintf("Error: %v", err),
					ToolCallID: call.ID,
				})
				continue
			}
		toolSuccess:

			// Display tool output (truncated for display)
			if toolResult != "" {
				output := toolResult
				if len(output) > 300 {
					output = output[:300] + "... (truncated)"
				}
				// Show result in grey, indented
				fmt.Fprintf(os.Stderr, "\033[90m  %s\033[0m\n", strings.ReplaceAll(output, "\n", "\n  "))
			}

			// Truncate output to prevent context overflow
			toolResultForLLM := truncateToolOutput(toolResult, opts.Model)

			toolResults = append(toolResults, api.Message{
				Role:       "tool",
				Content:    toolResultForLLM,
				ToolCallID: call.ID,
			})
		}

		// Add tool results to message history
		messages = append(messages, toolResults...)

		fmt.Fprintf(os.Stderr, "\n")

		// Reset state for next iteration
		fullResponse.Reset()
		thinkingContent.Reset()
		thinkTagOpened = false
		thinkTagClosed = false
		pendingToolCalls = nil
		state = &displayResponseState{}

		// Start new progress spinner for next API call
		p = progress.NewProgress(os.Stderr)
		spinner = progress.NewSpinner("")
		p.Add("", spinner)
	}

	if len(opts.Messages) > 0 {
		fmt.Println()
		fmt.Println()
	}

	return &api.Message{Role: role, Thinking: thinkingContent.String(), Content: fullResponse.String()}, nil
}

// truncateUTF8 safely truncates a string to at most limit runes, adding "..." if truncated.
func truncateUTF8(s string, limit int) string {
	runes := []rune(s)
	if len(runes) <= limit {
		return s
	}
	if limit <= 3 {
		return string(runes[:limit])
	}
	return string(runes[:limit-3]) + "..."
}

// formatToolShort returns a short description of a tool call.
func formatToolShort(toolName string, args map[string]any) string {
	displayName := agent.ToolDisplayName(toolName)
	if toolName == "bash" {
		if cmd, ok := args["command"].(string); ok {
			return fmt.Sprintf("%s: %s", displayName, truncateUTF8(cmd, 50))
		}
	}
	if toolName == "web_search" {
		if query, ok := args["query"].(string); ok {
			return fmt.Sprintf("%s: %s", displayName, truncateUTF8(query, 50))
		}
	}
	return displayName
}

// Helper types and functions for display

type displayResponseState struct {
	lineLength int
	wordBuffer string
}

func displayResponse(content string, wordWrap bool, state *displayResponseState) {
	termWidth, _, _ := term.GetSize(int(os.Stdout.Fd()))
	if wordWrap && termWidth >= 10 {
		for _, ch := range content {
			if state.lineLength+1 > termWidth-5 {
				if len(state.wordBuffer) > termWidth-10 {
					fmt.Printf("%s%c", state.wordBuffer, ch)
					state.wordBuffer = ""
					state.lineLength = 0
					continue
				}

				// backtrack the length of the last word and clear to the end of the line
				a := len(state.wordBuffer)
				if a > 0 {
					fmt.Printf("\x1b[%dD", a)
				}
				fmt.Printf("\x1b[K\n")
				fmt.Printf("%s%c", state.wordBuffer, ch)

				state.lineLength = len(state.wordBuffer) + 1
			} else {
				fmt.Print(string(ch))
				state.lineLength++

				switch ch {
				case ' ', '\t':
					state.wordBuffer = ""
				case '\n', '\r':
					state.lineLength = 0
					state.wordBuffer = ""
				default:
					state.wordBuffer += string(ch)
				}
			}
		}
	} else {
		fmt.Printf("%s%s", state.wordBuffer, content)
		if len(state.wordBuffer) > 0 {
			state.wordBuffer = ""
		}
	}
}

func thinkingOutputOpeningText(plainText bool) string {
	text := "Thinking...\n"

	if plainText {
		return text
	}

	return readline.ColorGrey + readline.ColorBold + text + readline.ColorDefault + readline.ColorGrey
}

func thinkingOutputClosingText(plainText bool) string {
	text := "...done thinking.\n\n"

	if plainText {
		return text
	}

	return readline.ColorGrey + readline.ColorBold + text + readline.ColorDefault
}

func renderToolCalls(toolCalls []api.ToolCall, plainText bool) string {
	out := ""
	formatExplanation := ""
	formatValues := ""
	if !plainText {
		formatExplanation = readline.ColorGrey + readline.ColorBold
		formatValues = readline.ColorDefault
		out += formatExplanation
	}
	for i, toolCall := range toolCalls {
		argsAsJSON, err := json.Marshal(toolCall.Function.Arguments)
		if err != nil {
			return ""
		}
		if i > 0 {
			out += "\n"
		}
		out += fmt.Sprintf("  Tool call: %s(%s)", formatValues+toolCall.Function.Name+formatExplanation, formatValues+string(argsAsJSON)+formatExplanation)
	}
	if !plainText {
		out += readline.ColorDefault
	}
	return out
}

// checkModelCapabilities checks if the model supports tools.
func checkModelCapabilities(ctx context.Context, modelName string) (supportsTools bool, err error) {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return false, err
	}

	resp, err := client.Show(ctx, &api.ShowRequest{Model: modelName})
	if err != nil {
		return false, err
	}

	for _, cap := range resp.Capabilities {
		if cap == model.CapabilityTools {
			return true, nil
		}
	}

	return false, nil
}

// GenerateInteractive runs an interactive agent session.
// This is called from cmd.go when --experimental flag is set.
// If yoloMode is true, all tool approvals are skipped.
func GenerateInteractive(cmd *cobra.Command, modelName string, wordWrap bool, options map[string]any, think *api.ThinkValue, hideThinking bool, keepAlive *api.Duration, yoloMode bool) error {
	scanner, err := readline.New(readline.Prompt{
		Prompt:         ">>> ",
		AltPrompt:      "... ",
		Placeholder:    "Send a message (/? for help)",
		AltPlaceholder: `Use """ to end multi-line input`,
	})
	if err != nil {
		return err
	}

	fmt.Print(readline.StartBracketedPaste)
	defer fmt.Printf(readline.EndBracketedPaste)

	// Check if model supports tools
	supportsTools, err := checkModelCapabilities(cmd.Context(), modelName)
	if err != nil {
		fmt.Fprintf(os.Stderr, "\033[1mwarning:\033[0m could not check model capabilities: %v\n", err)
		supportsTools = false
	}

	// Create tool registry only if model supports tools
	var toolRegistry *tools.Registry
	if supportsTools {
		toolRegistry = tools.DefaultRegistry()

		if toolRegistry.Has("bash") {
			fmt.Fprintln(os.Stderr)
			fmt.Fprintln(os.Stderr, "This experimental version of Ollama has the \033[1mbash\033[0m tool enabled.")
			fmt.Fprintln(os.Stderr, "Models can read files on your computer, or run commands (after you allow them).")
			fmt.Fprintln(os.Stderr)
		}

		if yoloMode {
			fmt.Fprintf(os.Stderr, "\033[1mwarning:\033[0m yolo mode - all tool approvals will be skipped\n")
		}
	}

	// Create approval manager for session
	approval := agent.NewApprovalManager()

	var messages []api.Message
	var sb strings.Builder

	for {
		line, err := scanner.Readline()
		switch {
		case errors.Is(err, io.EOF):
			fmt.Println()
			return nil
		case errors.Is(err, readline.ErrInterrupt):
			if line == "" {
				fmt.Println("\nUse Ctrl + d or /bye to exit.")
			}
			sb.Reset()
			continue
		case err != nil:
			return err
		}

		switch {
		case strings.HasPrefix(line, "/exit"), strings.HasPrefix(line, "/bye"):
			return nil
		case strings.HasPrefix(line, "/clear"):
			messages = []api.Message{}
			approval.Reset()
			fmt.Println("Cleared session context and tool approvals")
			continue
		case strings.HasPrefix(line, "/tools"):
			showToolsStatus(toolRegistry, approval, supportsTools)
			continue
		case strings.HasPrefix(line, "/help"), strings.HasPrefix(line, "/?"):
			fmt.Fprintln(os.Stderr, "Available Commands:")
			fmt.Fprintln(os.Stderr, "  /tools          Show available tools and approvals")
			fmt.Fprintln(os.Stderr, "  /clear          Clear session context and approvals")
			fmt.Fprintln(os.Stderr, "  /bye            Exit")
			fmt.Fprintln(os.Stderr, "  /?, /help       Help for a command")
			fmt.Fprintln(os.Stderr, "")
			fmt.Fprintln(os.Stderr, "Keyboard Shortcuts:")
			fmt.Fprintln(os.Stderr, "  Ctrl+O          Expand last tool output")
			fmt.Fprintln(os.Stderr, "")
			continue
		case strings.HasPrefix(line, "/"):
			fmt.Printf("Unknown command '%s'. Type /? for help\n", strings.Fields(line)[0])
			continue
		default:
			sb.WriteString(line)
		}

		if sb.Len() > 0 {
			newMessage := api.Message{Role: "user", Content: sb.String()}
			messages = append(messages, newMessage)

			opts := RunOptions{
				Model:        modelName,
				Messages:     messages,
				WordWrap:     wordWrap,
				Options:      options,
				Think:        think,
				HideThinking: hideThinking,
				KeepAlive:    keepAlive,
				Tools:        toolRegistry,
				Approval:     approval,
				YoloMode:     yoloMode,
			}

			assistant, err := Chat(cmd.Context(), opts)
			if err != nil {
				return err
			}
			if assistant != nil {
				messages = append(messages, *assistant)
			}

			sb.Reset()
		}
	}
}

// showToolsStatus displays the current tools and approval status.
func showToolsStatus(registry *tools.Registry, approval *agent.ApprovalManager, supportsTools bool) {
	if !supportsTools || registry == nil {
		fmt.Println("Tools not available - model does not support tool calling")
		fmt.Println()
		return
	}

	fmt.Println("Available tools:")
	for _, name := range registry.Names() {
		tool, _ := registry.Get(name)
		fmt.Printf("  %s - %s\n", name, tool.Description())
	}

	allowed := approval.AllowedTools()
	if len(allowed) > 0 {
		fmt.Println("\nSession approvals:")
		for _, key := range allowed {
			fmt.Printf("  %s\n", key)
		}
	} else {
		fmt.Println("\nNo tools approved for this session yet")
	}
	fmt.Println()
}
