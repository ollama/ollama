//go:build integration

package integration

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

// TestAPIToolCallingStress tests tool calling with complex, agent-style prompts
// that include large system messages, multiple tools, and multi-turn conversations.
// This catches cache corruption and parser bugs that simple tool tests miss.
func TestAPIToolCallingStress(t *testing.T) {
	initialTimeout := 120 * time.Second
	streamTimeout := 120 * time.Second
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Minute)
	defer cancel()

	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()

	minVRAM := map[string]uint64{
		"qwen3-vl":      16,
		"gpt-oss:20b":   16,
		"gpt-oss:120b":  70,
		"qwen3":         6,
		"llama3.1":      8,
		"llama3.2":      4,
		"mistral":       6,
		"qwen2.5":       6,
		"qwen2":         6,
		"ministral-3":   20,
		"mistral-nemo":  9,
		"mistral-small": 16,
		"mixtral:8x22b": 80,
		"qwq":           20,
		"granite3.3":    7,
	}

	// Models that don't reliably produce tool calls with complex/multi-tool prompts.
	// The stress test uses a large system prompt with many tools, simulating coding agents.
	// Some models are too small, too slow, or not designed for this use case.
	skipModels := map[string]string{
		"lfm2.5-thinking": "returns text instead of tool calls with complex system prompts",
		"qwen3-vl":        "vision model, extremely slow with complex tool prompts",
		"llama3.2":        "3B model too small for reliable multi-tool agent prompts",
		"mistral":         "7B v0.3 returns text instead of tool calls with complex prompts",
		"mixtral:8x22b":   "returns text instead of tool calls with complex prompts",
		"qwen2":           "returns text instead of tool calls with complex prompts",
		"granite3.3":      "returns text instead of tool calls with complex prompts",
	}

	models := testModels(libraryToolsModels)

	for _, model := range models {
		t.Run(model, func(t *testing.T) {
			// Skip known-bad models unless explicitly requested via env var
			if reason, ok := skipModels[model]; ok && testModel == "" {
				t.Skipf("skipping: %s", reason)
			}
			if testModel != "" {
				requireCapability(ctx, t, client, model, "tools")
			}
			if v, ok := minVRAM[model]; ok {
				skipUnderMinVRAM(t, v)
			}

			pullOrSkip(ctx, t, client, model)

			tools := stressTestTools()

			// Large system prompt that mimics real coding agents (opencode, Claude Code, etc.)
			// This is intentionally very long (~5000+ tokens) to match the prompt sizes that
			// real coding agents send. The combination of a large system prompt, many tools,
			// and thinking mode is what triggers failures in some models.
			systemPrompt := stressTestSystemPrompt()

			// Test 1: First request (fresh prompt processing)
			// Use a direct prompt that tells the model exactly what tool to use,
			// reducing the chance it asks for clarification instead.
			t.Run("first_request", func(t *testing.T) {
				testToolCall(t, ctx, client, model, systemPrompt, tools,
					"Run git diff main to review the code changes on the current branch.",
					initialTimeout, streamTimeout)
			})

			// Test 2: Repeat with same prompt (tests cache reuse)
			t.Run("cached_request", func(t *testing.T) {
				testToolCall(t, ctx, client, model, systemPrompt, tools,
					"Run git diff main to review the code changes on the current branch.",
					initialTimeout, streamTimeout)
			})

			// Test 3: Different user message (partial cache hit)
			t.Run("different_user_message", func(t *testing.T) {
				testToolCall(t, ctx, client, model, systemPrompt, tools,
					"Read the file at ./go.mod and tell me what dependencies we have.",
					initialTimeout, streamTimeout)
			})

			// Test 4: Multi-turn with tool response
			t.Run("multi_turn", func(t *testing.T) {
				testToolCallMultiTurn(t, ctx, client, model, systemPrompt, tools,
					initialTimeout, streamTimeout)
			})
		})
	}
}

func newTool(name, description string, required []string, props map[string]api.ToolProperty) api.Tool {
	return api.Tool{
		Type: "function",
		Function: api.ToolFunction{
			Name:        name,
			Description: description,
			Parameters: api.ToolFunctionParameters{
				Type:       "object",
				Required:   required,
				Properties: testPropsMap(props),
			},
		},
	}
}

// stressTestTools returns a set of tools matching the scale and verbosity of
// real coding agent tool definitions (opencode, Claude Code, etc.). The tool
// descriptions are intentionally verbose to match real-world prompt sizes.
func stressTestTools() []api.Tool {
	return []api.Tool{
		newTool("bash", "Executes a given bash command in a persistent shell session with optional timeout, ensuring proper handling and security measures. All commands run in the working directory by default. Before executing the command, verify that the parent directory exists. Always quote file paths that contain spaces with double quotes. After ensuring proper quoting, execute the command and capture the output. Avoid using bash with find, grep, cat, head, tail, sed, awk, or echo commands unless explicitly instructed. Instead, always prefer using the dedicated tools for these commands. When issuing multiple commands, if they are independent and can run in parallel, make multiple tool calls in a single message.",
			[]string{"command"},
			map[string]api.ToolProperty{
				"command":     {Type: api.PropertyType{"string"}, Description: "The bash command to execute"},
				"description": {Type: api.PropertyType{"string"}, Description: "Short description of what this command does in 5-10 words"},
				"timeout":     {Type: api.PropertyType{"number"}, Description: "Optional timeout in milliseconds. If not specified, commands will time out after 120000ms (2 minutes)"},
			}),
		newTool("read", "Read a file or directory from the local filesystem. If the path does not exist, an error is returned. By default, this tool returns up to 2000 lines from the start of the file. The offset parameter is the line number to start from (1-indexed). To read later sections, call this tool again with a larger offset. Use the grep tool to find specific content in large files or files with long lines. If you are unsure of the correct file path, use the glob tool to look up filenames by glob pattern. Contents are returned with each line prefixed by its line number. Any line longer than 2000 characters is truncated. Call this tool in parallel when you know there are multiple files you want to read. Avoid tiny repeated slices (30 line chunks). If you need more context, read a larger window. This tool can read image files and PDFs and return them as file attachments.",
			[]string{"path"},
			map[string]api.ToolProperty{
				"path":   {Type: api.PropertyType{"string"}, Description: "The absolute path to the file to read"},
				"offset": {Type: api.PropertyType{"number"}, Description: "Line number to start reading from (1-indexed)"},
				"limit":  {Type: api.PropertyType{"number"}, Description: "Maximum number of lines to read"},
			}),
		newTool("glob", "Fast file pattern matching tool that works with any codebase size. Supports glob patterns like '**/*.js' or 'src/**/*.ts'. Returns matching file paths sorted by modification time. Use this tool when you need to find files by name patterns. When you are doing an open-ended search that may require multiple rounds of globbing and grepping, use the task tool instead. You have the capability to call multiple tools in a single response. It is always better to speculatively perform multiple searches as a batch that are potentially useful.",
			[]string{"pattern"},
			map[string]api.ToolProperty{
				"pattern": {Type: api.PropertyType{"string"}, Description: "The glob pattern to match files against"},
				"path":    {Type: api.PropertyType{"string"}, Description: "The directory to search in"},
			}),
		newTool("grep", "Fast content search tool that works with any codebase size. Searches file contents using regular expressions. Supports full regex syntax (eg. 'log.*Error', 'function\\s+\\w+'). Filter files by pattern with the include parameter (eg. '*.js', '*.{ts,tsx}'). Returns file paths and line numbers with at least one match sorted by modification time. Use this tool when you need to find files containing specific patterns. If you need to identify or count the number of matches within files, use the bash tool with rg (ripgrep) directly. When you are doing an open-ended search that may require multiple rounds of globbing and grepping, use the task tool instead.",
			[]string{"pattern"},
			map[string]api.ToolProperty{
				"pattern": {Type: api.PropertyType{"string"}, Description: "The regex pattern to search for in file contents"},
				"path":    {Type: api.PropertyType{"string"}, Description: "The directory to search in"},
				"include": {Type: api.PropertyType{"string"}, Description: "File pattern to include (eg. '*.js', '*.{ts,tsx}')"},
			}),
		newTool("edit", "Performs exact string replacements in files. You must use your read tool at least once in the conversation before editing. This tool will error if you attempt an edit without reading the file. When editing text from read tool output, ensure you preserve the exact indentation (tabs/spaces) as it appears after the line number prefix. Always prefer editing existing files in the codebase. Never write new files unless explicitly required. Only use emojis if the user explicitly requests it. The edit will fail if oldString is not found in the file. The edit will fail if oldString is found multiple times in the file. Use replaceAll for replacing and renaming strings across the file.",
			[]string{"path", "old_string", "new_string"},
			map[string]api.ToolProperty{
				"path":       {Type: api.PropertyType{"string"}, Description: "The absolute path to the file to modify"},
				"old_string": {Type: api.PropertyType{"string"}, Description: "The text to replace (must be unique in the file)"},
				"new_string": {Type: api.PropertyType{"string"}, Description: "The replacement text"},
			}),
		newTool("write", "Writes a file to the local filesystem. This tool will overwrite the existing file if there is one at the provided path. If this is an existing file, you must use the read tool first to read the file contents. This tool will fail if you did not read the file first. Always prefer editing existing files in the codebase. Never write new files unless explicitly required. Never proactively create documentation files or README files. Only create documentation files if explicitly requested by the user.",
			[]string{"path", "content"},
			map[string]api.ToolProperty{
				"path":    {Type: api.PropertyType{"string"}, Description: "The absolute path to the file to write"},
				"content": {Type: api.PropertyType{"string"}, Description: "The content to write to the file"},
			}),
		newTool("question", "Use this tool when you need to ask the user questions during execution. This allows you to gather user preferences or requirements, clarify ambiguous instructions, get decisions on implementation choices as you work, and offer choices to the user about what direction to take. When custom is enabled (default), a 'Type your own answer' option is added automatically. Answers are returned as arrays of labels. Set multiple to true to allow selecting more than one answer. If you recommend a specific option, make that the first option in the list and add '(Recommended)' at the end of the label.",
			[]string{"questions"},
			map[string]api.ToolProperty{
				"questions": {Type: api.PropertyType{"string"}, Description: "The question to ask the user"},
			}),
		newTool("task", "Launch a new agent to handle complex, multistep tasks autonomously. Available agent types: general (general-purpose agent for researching complex questions and executing multi-step tasks, use this to execute multiple units of work in parallel) and explore (fast agent specialized for exploring codebases, use this when you need to quickly find files by patterns, search code for keywords, or answer questions about the codebase). Launch multiple agents concurrently whenever possible to maximize performance. When the agent is done, it will return a single message back to you. Each agent invocation starts with a fresh context unless you provide task_id to resume the same subagent session.",
			[]string{"description", "prompt", "subagent_type"},
			map[string]api.ToolProperty{
				"description":   {Type: api.PropertyType{"string"}, Description: "A short (3-5 word) description of the task"},
				"prompt":        {Type: api.PropertyType{"string"}, Description: "The task for the agent to perform"},
				"subagent_type": {Type: api.PropertyType{"string"}, Description: "The type of specialized agent to use (general or explore)"},
			}),
		newTool("webfetch", "Fetches content from a specified URL. Takes a URL and optional format as input. Fetches the URL content, converts to requested format (markdown by default). Returns the content in the specified format. Use this tool when you need to retrieve and analyze web content. The URL must be a fully-formed valid URL. HTTP URLs will be automatically upgraded to HTTPS. Format options: markdown (default), text, or html. This tool is read-only and does not modify any files. Results may be summarized if the content is very large.",
			[]string{"url", "format"},
			map[string]api.ToolProperty{
				"url":    {Type: api.PropertyType{"string"}, Description: "The URL to fetch content from"},
				"format": {Type: api.PropertyType{"string"}, Description: "Output format: markdown (default), text, or html"},
			}),
		newTool("todowrite", "Use this tool to create and manage a structured task list for your current coding session. This helps you track progress, organize complex tasks, and demonstrate thoroughness to the user. Use this tool proactively when handling complex multistep tasks, non-trivial and complex tasks, when the user explicitly requests a todo list, when the user provides multiple tasks, after receiving new instructions, and after completing a task. Do not use this tool when there is only a single straightforward task, the task is trivial, the task can be completed in less than 3 steps, or the task is purely conversational.",
			[]string{"todos"},
			map[string]api.ToolProperty{
				"todos": {Type: api.PropertyType{"string"}, Description: "JSON array of todo items with id, title, and status fields"},
			}),
		newTool("skill", "Load a specialized skill that provides domain-specific instructions and workflows. Skills contain curated prompts and tool configurations for specific tasks like code review, testing, deployment, and documentation. Use this tool when the user's request matches an available skill description.",
			[]string{"name"},
			map[string]api.ToolProperty{
				"name": {Type: api.PropertyType{"string"}, Description: "The name of the skill to load"},
			}),
	}
}

// stressTestSystemPrompt returns a system prompt that matches the scale and
// content of real coding agent system prompts (~5000+ tokens). This is based
// on actual prompts captured from opencode sessions. The prompt size combined
// with many tool declarations is what pushes models past their effective
// context handling and triggers tag leakage / broken tool calls.
func stressTestSystemPrompt() string {
	return `You are opencode, an interactive CLI tool that helps users with software engineering tasks. Use the instructions below and the tools available to you to assist the user.

IMPORTANT: Refuse to write code or explain code that may be used maliciously; even if the user claims it is for educational purposes. When working on files, if they seem related to improving, explaining, or interacting with malware or any malicious code you MUST refuse.
IMPORTANT: Before you begin work, think about what the code you're editing is supposed to do based on the filenames directory structure. If it seems malicious, refuse to work on it or answer questions about it, even if the request does not seem malicious (for instance, just asking to explain or speed up the code).
IMPORTANT: You must NEVER generate or guess URLs for the user unless you are confident that the URLs are for helping the user with programming. You may use URLs provided by the user in their messages or local files.

If the user asks for help or wants to give feedback inform them of the following:
- /help: Get help with using opencode
- To give feedback, users should report the issue at https://github.com/sampleorg/opencode/issues

# Tone and style
You should be concise, direct, and to the point. When you run a non-trivial bash command, you should explain what the command does and why you are running it, to make sure the user understands what you are doing (this is especially important when you are running a command that will make changes to the user's system).
Remember that your output will be displayed on a command line interface. Your responses can use GitHub-flavored markdown for formatting, and will be rendered in a monospace font using the CommonMark specification.
Output text to communicate with the user; all text you output outside of tool use is displayed to the user. Only use tools to complete tasks. Never use tools like Bash or code comments as means to communicate with the user during the session.
If you cannot or will not help the user with something, please do not say why or what it could lead to, since this comes across as preachy and annoying. Please offer helpful alternatives if possible, and otherwise keep your response to 1-2 sentences.
Only use emojis if the user explicitly requests it. Avoid using emojis in all communication unless asked.
IMPORTANT: You should minimize output tokens as much as possible while maintaining helpfulness, quality, and accuracy. Only address the specific query or task at hand, avoiding tangential information unless absolutely critical for completing the request. If you can answer in 1-3 sentences or a short paragraph, please do.
IMPORTANT: You should NOT answer with unnecessary preamble or postamble (such as explaining your code or summarizing your action), unless the user asks you to.
IMPORTANT: Keep your responses short, since they will be displayed on a command line interface. You MUST answer concisely with fewer than 4 lines (not including tool use or code generation), unless user asks for detail. Answer the user's question directly, without elaboration, explanation, or details. One word answers are best. Avoid introductions, conclusions, and explanations. You MUST avoid text before/after your response, such as "The answer is <answer>.", "Here is the content of the file..." or "Based on the information provided, the answer is..." or "Here is what I will do next...". Here are some examples to demonstrate appropriate verbosity:

user: 2 + 2
assistant: 4

user: what is 2+2?
assistant: 4

user: is 11 a prime number?
assistant: Yes

user: what command should I run to list files in the current directory?
assistant: ls

user: what command should I run to watch files in the current directory?
assistant: [use the ls tool to list the files in the current directory, then read docs/commands in the relevant file to find out how to watch files]
npm run dev

user: How many golf balls fit inside a jetta?
assistant: 150000

user: what files are in the directory src/?
assistant: [runs ls and sees foo.c, bar.c, baz.c]
user: which file contains the implementation of foo?
assistant: src/foo.c

user: write tests for new feature
assistant: [uses grep and glob search tools to find where similar tests are defined, uses concurrent read file tool use blocks in one tool call to read relevant files at the same time, uses edit file tool to write new tests]

# Proactiveness
You are allowed to be proactive, but only when the user asks you to do something. You should strive to strike a balance between:
1. Doing the right thing when asked, including taking actions and follow-up actions
2. Not surprising the user with actions you take without asking
For example, if the user asks you how to approach something, you should do your best to answer their question first, and not immediately jump into taking actions.
3. Do not add additional code explanation summary unless requested by the user. After working on a file, just stop, rather than providing an explanation of what you did.

# Following conventions
When making changes to files, first understand the file's code conventions. Mimic code style, use existing libraries and utilities, and follow existing patterns.
- NEVER assume that a given library is available, even if it is well known. Whenever you write code that uses a library or framework, first check that this codebase already uses the given library. For example, you might look at neighboring files, or check the package.json (or cargo.toml, and so on depending on the language).
- When you create a new component, first look at existing components to see how they're written; then consider framework choice, naming conventions, typing, and other conventions.
- When you edit a piece of code, first look at the code's surrounding context (especially its imports) to understand the code's choice of frameworks and libraries. Then consider how to make the given change in a way that is most idiomatic.
- Always follow security best practices. Never introduce code that exposes or logs secrets and keys. Never commit secrets or keys to the repository.

# Code style
- IMPORTANT: DO NOT ADD ANY COMMENTS unless asked

# Doing tasks
The user will primarily request you perform software engineering tasks. This includes solving bugs, adding new functionality, refactoring code, explaining code, and more. For these tasks the following steps are recommended:
- Use the available search tools to understand the codebase and the user's query. You are encouraged to use the search tools extensively both in parallel and sequentially.
- Implement the solution using all tools available to you
- Verify the solution if possible with tests. NEVER assume specific test framework or test script. Check the README or search codebase to determine the testing approach.
- VERY IMPORTANT: When you have completed a task, you MUST run the lint and typecheck commands (e.g. npm run lint, npm run typecheck, ruff, etc.) with Bash if they were provided to you to ensure your code is correct. If you are unable to find the correct command, ask the user for the command to run and if they supply it, proactively suggest writing it to AGENTS.md so that you will know to run it next time.
NEVER commit changes unless the user explicitly asks you to. It is VERY IMPORTANT to only commit when explicitly asked, otherwise the user will feel that you are being too proactive.

# Tool usage policy
- When doing file search, prefer to use the Task tool in order to reduce context usage.
- You have the capability to call multiple tools in a single response. When multiple independent pieces of information are requested, batch your tool calls together for optimal performance. When making multiple bash tool calls, you MUST send a single message with multiple tools calls to run the calls in parallel.

You MUST answer concisely with fewer than 4 lines of text (not including tool use or code generation), unless user asks for detail.

# Code References
When referencing specific functions or pieces of code include the pattern file_path:line_number to allow the user to easily navigate to the source code location.

# Git workflow
When working with git:
- Create descriptive commit messages that explain WHY not just WHAT
- Use conventional commit format: feat:, fix:, refactor:, docs:, test:, chore:
- Check git status before and after operations
- Never force push to main/master
- Review diffs before committing
- NEVER update the git config
- NEVER run destructive/irreversible git commands unless the user explicitly requests them
- NEVER skip hooks (--no-verify, --no-gpg-sign, etc) unless the user explicitly requests it
- Avoid git commit --amend unless explicitly requested by the user
- NEVER commit changes unless the user explicitly asks you to

# Safety
- Never delete files without confirmation
- Never run destructive commands (rm -rf, DROP TABLE, etc.) without confirmation
- Always validate inputs before using them in shell commands
- Be careful with environment variables and secrets
- Do not expose API keys, passwords, or tokens in code or logs

# Environment
Working directory: /Users/test/code/myproject
Platform: darwin
Shell: zsh
Is directory a git repo: yes
The project uses Go 1.22 with modules. Run tests with 'go test ./...' and build with 'go build ./...'.
The CI pipeline runs golangci-lint, go vet, and go test with race detector enabled.

# User instructions
Never use cd to change into the repo root or any other directory in Bash commands. The working directory is always the repo root — use relative paths directly.
Never use heredoc-style inline bash or python scripts in Bash tool calls. Instead, write the script to an ephemeral file under ./.tmp/ in the repo, then run it as a separate command.`
}

// validStressTools is the set of tool names used in the stress test.
var validStressTools = map[string]bool{
	"bash": true, "read": true, "glob": true, "grep": true,
	"edit": true, "write": true, "question": true, "task": true,
	"webfetch": true, "todowrite": true, "skill": true,
}

func testToolCall(t *testing.T, ctx context.Context, client *api.Client, model, systemPrompt string, tools []api.Tool, userMessage string, initialTimeout, streamTimeout time.Duration) {
	t.Helper()

	req := api.ChatRequest{
		Model: model,
		Messages: []api.Message{
			{Role: "system", Content: systemPrompt},
			{Role: "user", Content: userMessage},
		},
		Tools: tools,
		Options: map[string]any{
			"temperature": 0,
			"num_ctx":     contextLength(16384),
		},
	}

	stallTimer := time.NewTimer(initialTimeout)
	var gotToolCall bool
	var lastToolCall api.ToolCall
	var allContent string

	fn := func(response api.ChatResponse) error {
		if len(response.Message.ToolCalls) > 0 {
			gotToolCall = true
			lastToolCall = response.Message.ToolCalls[len(response.Message.ToolCalls)-1]
		}
		allContent += response.Message.Content
		if !stallTimer.Reset(streamTimeout) {
			return fmt.Errorf("stall detected while streaming")
		}
		return nil
	}

	stream := true
	req.Stream = &stream
	done := make(chan int)
	var genErr error
	go func() {
		genErr = client.Chat(ctx, &req, fn)
		done <- 0
	}()

	select {
	case <-stallTimer.C:
		t.Fatalf("chat stalled after %s", initialTimeout)
	case <-done:
		if genErr != nil {
			t.Fatalf("chat failed: %v", genErr)
		}

		// Check for leaked special tags in content — these should never
		// appear in user-visible output regardless of model quality.
		checkNoLeakedTags(t, allContent)

		// The model must produce either a tool call or a text response.
		// A text response (e.g. asking for clarification) is legitimate.
		// Empty output with no tool call indicates a parser or model failure
		// (e.g. malformed tool call that gets dropped).
		if !gotToolCall && allContent == "" {
			t.Fatal("model produced neither a tool call nor text content")
		}
		if gotToolCall {
			if !validStressTools[lastToolCall.Function.Name] {
				t.Errorf("unexpected tool: %q", lastToolCall.Function.Name)
			}
			argsJSON, _ := json.Marshal(lastToolCall.Function.Arguments)
			t.Logf("tool call: %s(%s)", lastToolCall.Function.Name, string(argsJSON))
		} else {
			t.Logf("text response (no tool call): %q", truncate(allContent, 200))
		}
	case <-ctx.Done():
		t.Fatal("context cancelled")
	}
}

func testToolCallMultiTurn(t *testing.T, ctx context.Context, client *api.Client, model, systemPrompt string, tools []api.Tool, initialTimeout, streamTimeout time.Duration) {
	t.Helper()

	req := api.ChatRequest{
		Model: model,
		Messages: []api.Message{
			{Role: "system", Content: systemPrompt},
			{Role: "user", Content: "What files are in the current directory?"},
			{Role: "assistant", Content: "", ToolCalls: []api.ToolCall{{
				Function: api.ToolCallFunction{
					Name:      "bash",
					Arguments: api.ToolCallFunctionArguments{},
				},
			}}},
			{Role: "tool", Content: "go.mod\ngo.sum\nmain.go\nREADME.md\n"},
			// The model should now respond with content or another tool call
		},
		Tools: tools,
		Options: map[string]any{
			"temperature": 0,
			"num_ctx":     contextLength(16384),
		},
	}

	// For the tool response arguments, set the command
	req.Messages[2].ToolCalls[0].Function.Arguments.Set("command", "ls")

	stallTimer := time.NewTimer(initialTimeout)
	var gotResponse bool
	var allContent string
	var gotToolCall bool

	fn := func(response api.ChatResponse) error {
		if response.Message.Content != "" {
			gotResponse = true
			allContent += response.Message.Content
		}
		if len(response.Message.ToolCalls) > 0 {
			gotToolCall = true
			gotResponse = true
		}
		if !stallTimer.Reset(streamTimeout) {
			return fmt.Errorf("stall detected")
		}
		return nil
	}

	stream := true
	req.Stream = &stream
	done := make(chan int)
	var genErr error
	go func() {
		genErr = client.Chat(ctx, &req, fn)
		done <- 0
	}()

	select {
	case <-stallTimer.C:
		t.Fatalf("chat stalled after %s", initialTimeout)
	case <-done:
		if genErr != nil {
			t.Fatalf("chat failed: %v", genErr)
		}

		checkNoLeakedTags(t, allContent)

		if !gotResponse {
			t.Fatal("expected response (content or tool call), got nothing")
		}
		if gotToolCall {
			t.Log("multi-turn: got follow-up tool call")
		} else {
			t.Logf("multi-turn: got content response: %q", truncate(allContent, 200))
		}
	case <-ctx.Done():
		t.Fatal("context cancelled")
	}
}

// checkNoLeakedTags verifies that model-internal special tags do not appear in
// user-visible content. These tags should be consumed by the parser and never
// passed through. If they appear, either the parser has a bug or the model is
// generating malformed output that the parser fails to handle.
func checkNoLeakedTags(t *testing.T, content string) {
	t.Helper()
	leakedTags := []string{
		"<|channel>", "<channel|>",
		"<|tool_call>", "<tool_call|>",
		"<|tool>", "<tool|>",
		"<|turn>", "<turn|>",
	}
	for _, tag := range leakedTags {
		if strings.Contains(content, tag) {
			t.Errorf("leaked special tag %q in content: %q", tag, truncate(content, 300))
		}
	}
}

func contextLength(defaultVal int) int {
	if s := os.Getenv("OLLAMA_CONTEXT_LENGTH"); s != "" {
		if n, err := strconv.Atoi(s); err == nil {
			return n
		}
	}
	return defaultVal
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
