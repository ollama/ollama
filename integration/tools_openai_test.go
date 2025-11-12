//go:build integration

package integration

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/ollama/ollama/api"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/shared"
)

var agenticModels = []string{
	"gpt-oss:20b",
	"gpt-oss:120b",
	"qwen3-coder:30b",
	"qwen3:4b",
	"qwen3:8b",
}

var cloudModels = []string{
	"gpt-oss:120b-cloud",
	"gpt-oss:20b-cloud",
	"qwen3-vl:235b-cloud",
	"qwen3-coder:480b-cloud",
	"kimi-k2-thinking:cloud",
	"kimi-k2:1t-cloud",
}

// validateBashCommand validates a bash command with flexible matching
// It checks that the core command matches and required arguments are present
func validateBashCommand(cmd string, expectedCmd string, requiredArgs []string) error {
	parts := strings.Fields(cmd)
	if len(parts) == 0 {
		return fmt.Errorf("empty command")
	}

	actualCmd := parts[0]
	if actualCmd != expectedCmd {
		return fmt.Errorf("expected command '%s', got '%s'", expectedCmd, actualCmd)
	}

	cmdStr := strings.Join(parts[1:], " ")
	for _, arg := range requiredArgs {
		if !strings.Contains(cmdStr, arg) {
			return fmt.Errorf("missing required argument: %s", arg)
		}
	}

	return nil
}

// validateBashCommandFlexible validates a bash command with flexible matching
// It accepts alternative command forms (e.g., find vs ls) and checks required patterns
func validateBashCommandFlexible(cmd string, allowedCommands []string, requiredPatterns []string) error {
	parts := strings.Fields(cmd)
	if len(parts) == 0 {
		return fmt.Errorf("empty command")
	}

	actualCmd := parts[0]
	commandMatched := false
	for _, allowedCmd := range allowedCommands {
		if actualCmd == allowedCmd {
			commandMatched = true
			break
		}
	}
	if !commandMatched {
		return fmt.Errorf("expected one of commands %v, got '%s'", allowedCommands, actualCmd)
	}

	cmdStr := strings.ToLower(strings.Join(parts[1:], " "))
	for _, pattern := range requiredPatterns {
		if !strings.Contains(cmdStr, strings.ToLower(pattern)) {
			return fmt.Errorf("missing required pattern: %s", pattern)
		}
	}

	return nil
}

func TestOpenAIToolCallingMultiStep(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	var baseURL string
	var apiKey string
	var modelsToTest []string
	var cleanup func()

	if openaiBaseURL := os.Getenv("OPENAI_BASE_URL"); openaiBaseURL != "" {
		baseURL = openaiBaseURL
		apiKey = os.Getenv("OLLAMA_API_KEY")
		if apiKey == "" {
			t.Fatal("OPENAI_API_KEY must be set when using OPENAI_BASE_URL")
		}

		// only test cloud models unless OPENAI_TEST_MODELS is set
		modelsToTest = cloudModels
		if modelsEnv := os.Getenv("OPENAI_TEST_MODELS"); modelsEnv != "" {
			modelsToTest = []string{modelsEnv}
		}
		cleanup = func() {}
	} else {
		_, testEndpoint, cleanupFn := InitServerConnection(ctx, t)
		cleanup = cleanupFn
		baseURL = fmt.Sprintf("http://%s/v1", testEndpoint)
		apiKey = "ollama"
		modelsToTest = append(agenticModels, cloudModels...)
	}
	t.Cleanup(cleanup)

	opts := []option.RequestOption{
		option.WithBaseURL(baseURL),
		option.WithAPIKey(apiKey),
	}
	openaiClient := openai.NewClient(opts...)

	var ollamaClient *api.Client
	if baseURL == "" {
		ollamaClient, _, _ = InitServerConnection(ctx, t)
	}

	for _, model := range modelsToTest {
		t.Run(model, func(t *testing.T) {
			testCtx := ctx
			if slices.Contains(cloudModels, model) {
				t.Parallel()
				// Create a new context for parallel tests to avoid cancellation
				var cancel context.CancelFunc
				testCtx, cancel = context.WithTimeout(context.Background(), 10*time.Minute)
				defer cancel()
			}
			if v, ok := minVRAM[model]; ok {
				skipUnderMinVRAM(t, v)
			}

			if ollamaClient != nil {
				if err := PullIfMissing(testCtx, ollamaClient, model); err != nil {
					t.Fatalf("pull failed %s", err)
				}
			}

			tools := []openai.ChatCompletionToolUnionParam{
				openai.ChatCompletionFunctionTool(shared.FunctionDefinitionParam{
					Name:        "list_files",
					Description: openai.Opt("List all files in a directory"),
					Parameters: shared.FunctionParameters{
						"type": "object",
						"properties": map[string]any{
							"path": map[string]any{
								"type":        "string",
								"description": "The directory path to list files from",
							},
						},
						"required": []string{"path"},
					},
				}),
				openai.ChatCompletionFunctionTool(shared.FunctionDefinitionParam{
					Name:        "read_file",
					Description: openai.Opt("Read the contents of a file"),
					Parameters: shared.FunctionParameters{
						"type": "object",
						"properties": map[string]any{
							"path": map[string]any{
								"type":        "string",
								"description": "The file path to read",
							},
						},
						"required": []string{"path"},
					},
				}),
			}

			mockFileContents := "line 1\nline 2\nline 3\nline 4\nline 5"
			userContent := "Find the file named 'config.json' in /tmp and read its contents"
			userMessage := openai.UserMessage(userContent)

			messages := []openai.ChatCompletionMessageParamUnion{
				userMessage,
			}
			stepCount := 0
			maxSteps := 10

			normalizePath := func(path string) string {
				if path != "" && path[0] != '/' {
					return "/" + path
				}
				return path
			}

			expectedSteps := []struct {
				functionName string
				validateArgs func(map[string]any) error
				result       string
			}{
				{
					functionName: "list_files",
					validateArgs: func(args map[string]any) error {
						path, ok := args["path"]
						if !ok {
							return fmt.Errorf("missing required argument 'path'")
						}
						pathStr, ok := path.(string)
						if !ok {
							return fmt.Errorf("expected 'path' to be string, got %T", path)
						}
						normalizedPath := normalizePath(pathStr)
						if normalizedPath != "/tmp" {
							return fmt.Errorf("expected list_files(\"/tmp\"), got list_files(%q)", pathStr)
						}
						return nil
					},
					result: `["config.json", "other.txt", "data.log"]`,
				},
				{
					functionName: "read_file",
					validateArgs: func(args map[string]any) error {
						path, ok := args["path"]
						if !ok {
							return fmt.Errorf("missing required argument 'path'")
						}
						pathStr, ok := path.(string)
						if !ok {
							return fmt.Errorf("expected 'path' to be string, got %T", path)
						}
						normalizedPath := normalizePath(pathStr)
						if normalizedPath != "/tmp/config.json" {
							return fmt.Errorf("expected read_file(\"/tmp/config.json\"), got read_file(%q)", pathStr)
						}
						return nil
					},
					result: mockFileContents,
				},
			}

			for stepCount < maxSteps {
				req := openai.ChatCompletionNewParams{
					Model:       shared.ChatModel(model),
					Messages:    messages,
					Tools:       tools,
					Temperature: openai.Opt(0.0),
				}

				completion, err := openaiClient.Chat.Completions.New(testCtx, req)
				if err != nil {
					t.Fatalf("step %d chat failed: %v", stepCount+1, err)
				}

				if len(completion.Choices) == 0 {
					t.Fatalf("step %d: no choices in response", stepCount+1)
				}

				choice := completion.Choices[0]
				message := choice.Message

				toolCalls := message.ToolCalls
				content := message.Content
				gotToolCall := len(toolCalls) > 0
				var toolCallID string
				if gotToolCall && toolCalls[0].ID != "" {
					toolCallID = toolCalls[0].ID
				}

				var assistantMessage openai.ChatCompletionMessageParamUnion
				if gotToolCall {
					toolCallsJSON, err := json.Marshal(toolCalls)
					if err != nil {
						t.Fatalf("step %d: failed to marshal tool calls: %v", stepCount+1, err)
					}
					var toolCallParams []openai.ChatCompletionMessageToolCallUnionParam
					if err := json.Unmarshal(toolCallsJSON, &toolCallParams); err != nil {
						t.Fatalf("step %d: failed to unmarshal tool calls: %v", stepCount+1, err)
					}
					contentUnion := openai.ChatCompletionAssistantMessageParamContentUnion{
						OfString: openai.Opt(content),
					}
					assistantMsg := openai.ChatCompletionAssistantMessageParam{
						Content:   contentUnion,
						ToolCalls: toolCallParams,
					}
					assistantMessage = openai.ChatCompletionMessageParamUnion{
						OfAssistant: &assistantMsg,
					}
				} else {
					assistantMessage = openai.AssistantMessage(content)
				}

				if !gotToolCall && content != "" {
					if stepCount < len(expectedSteps) {
						t.Logf("EXPECTED: Step %d should call '%s'", stepCount+1, expectedSteps[stepCount].functionName)
						t.Logf("ACTUAL: Model stopped with content: %s", content)
						t.Fatalf("model stopped making tool calls after %d steps, expected %d steps. Final response: %s", stepCount, len(expectedSteps), content)
					}
					return
				}

				if !gotToolCall || len(toolCalls) == 0 {
					if stepCount < len(expectedSteps) {
						expectedStep := expectedSteps[stepCount]
						t.Logf("EXPECTED: Step %d should call '%s'", stepCount+1, expectedStep.functionName)
						t.Logf("ACTUAL: No tool call, got content: %s", content)
						t.Fatalf("step %d: expected tool call but got none. Response: %s", stepCount+1, content)
					}
					return
				}

				if stepCount >= len(expectedSteps) {
					actualCallJSON, _ := json.MarshalIndent(toolCalls[0], "", "  ")
					t.Logf("EXPECTED: All %d steps completed", len(expectedSteps))
					t.Logf("ACTUAL: Extra step %d with tool call:\n%s", stepCount+1, string(actualCallJSON))
					funcName := "unknown"
					if toolCalls[0].Function.Name != "" {
						funcName = toolCalls[0].Function.Name
					}
					t.Fatalf("model made more tool calls than expected. Expected %d steps, got step %d with tool call: %s", len(expectedSteps), stepCount+1, funcName)
				}

				expectedStep := expectedSteps[stepCount]
				firstToolCall := toolCalls[0]
				funcCall := firstToolCall.Function
				if funcCall.Name == "" {
					t.Fatalf("step %d: tool call missing function name", stepCount+1)
				}

				funcName := funcCall.Name

				var args map[string]any
				if funcCall.Arguments != "" {
					if err := json.Unmarshal([]byte(funcCall.Arguments), &args); err != nil {
						t.Fatalf("step %d: failed to parse tool call arguments: %v", stepCount+1, err)
					}
				}

				if funcName != expectedStep.functionName {
					t.Logf("DIFF: Function name mismatch")
					t.Logf("  Expected: %s", expectedStep.functionName)
					t.Logf("  Got:      %s", funcName)
					t.Logf("  Arguments: %v", args)
					t.Fatalf("step %d: expected tool call '%s', got '%s'. Arguments: %v", stepCount+1, expectedStep.functionName, funcName, args)
				}

				if err := expectedStep.validateArgs(args); err != nil {
					expectedArgsForDisplay := map[string]any{}
					if expectedStep.functionName == "list_files" {
						expectedArgsForDisplay = map[string]any{"path": "/tmp"}
					} else if expectedStep.functionName == "read_file" {
						expectedArgsForDisplay = map[string]any{"path": "/tmp/config.json"}
					}
					if diff := cmp.Diff(expectedArgsForDisplay, args); diff != "" {
						t.Logf("DIFF: Arguments mismatch for function '%s' (-want +got):\n%s", expectedStep.functionName, diff)
					}
					t.Logf("Error: %v", err)
					t.Fatalf("step %d: tool call '%s' has invalid arguments: %v. Arguments: %v", stepCount+1, expectedStep.functionName, err, args)
				}

				toolMessage := openai.ToolMessage(expectedStep.result, toolCallID)
				messages = append(messages, assistantMessage, toolMessage)
				stepCount++
			}

			if stepCount < len(expectedSteps) {
				t.Fatalf("test exceeded max steps (%d) before completing all expected steps (%d)", maxSteps, len(expectedSteps))
			}
		})
	}
}

func TestOpenAIToolCallingBash(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	var baseURL string
	var apiKey string
	var modelsToTest []string
	var cleanup func()

	if openaiBaseURL := os.Getenv("OPENAI_BASE_URL"); openaiBaseURL != "" {
		baseURL = openaiBaseURL
		apiKey = os.Getenv("OLLAMA_API_KEY")
		if apiKey == "" {
			t.Fatal("OPENAI_API_KEY must be set when using OPENAI_BASE_URL")
		}
		modelsToTest = cloudModels
		if modelsEnv := os.Getenv("OPENAI_TEST_MODELS"); modelsEnv != "" {
			modelsToTest = []string{modelsEnv}
		}
		cleanup = func() {}
	} else {
		_, testEndpoint, cleanupFn := InitServerConnection(ctx, t)
		cleanup = cleanupFn
		baseURL = fmt.Sprintf("http://%s/v1", testEndpoint)
		apiKey = "ollama"
		modelsToTest = append(agenticModels, cloudModels...)
	}
	t.Cleanup(cleanup)

	opts := []option.RequestOption{
		option.WithBaseURL(baseURL),
		option.WithAPIKey(apiKey),
	}
	openaiClient := openai.NewClient(opts...)

	var ollamaClient *api.Client
	if baseURL == "" {
		ollamaClient, _, _ = InitServerConnection(ctx, t)
	}

	for _, model := range modelsToTest {
		t.Run(model, func(t *testing.T) {
			testCtx := ctx
			if slices.Contains(cloudModels, model) {
				t.Parallel()
				// Create a new context for parallel tests to avoid cancellation
				var cancel context.CancelFunc
				testCtx, cancel = context.WithTimeout(context.Background(), 10*time.Minute)
				defer cancel()
			}
			if v, ok := minVRAM[model]; ok {
				skipUnderMinVRAM(t, v)
			}

			if ollamaClient != nil {
				if err := PullIfMissing(testCtx, ollamaClient, model); err != nil {
					t.Fatalf("pull failed %s", err)
				}
			}

			tools := []openai.ChatCompletionToolUnionParam{
				openai.ChatCompletionFunctionTool(shared.FunctionDefinitionParam{
					Name:        "execute_bash",
					Description: openai.Opt("Execute a bash/shell command and return stdout, stderr, and exit code"),
					Parameters: shared.FunctionParameters{
						"type": "object",
						"properties": map[string]any{
							"command": map[string]any{
								"type":        "string",
								"description": "The bash command to execute",
							},
							"working_directory": map[string]any{
								"type":        "string",
								"description": "Optional working directory for command execution",
							},
						},
						"required": []string{"command"},
					},
				}),
			}

			userContent := "List all files in /tmp directory"
			userMessage := openai.UserMessage(userContent)

			req := openai.ChatCompletionNewParams{
				Model:       shared.ChatModel(model),
				Messages:    []openai.ChatCompletionMessageParamUnion{userMessage},
				Tools:       tools,
				Temperature: openai.Opt(0.0),
			}

			completion, err := openaiClient.Chat.Completions.New(testCtx, req)
			if err != nil {
				t.Fatalf("chat failed: %v", err)
			}

			if len(completion.Choices) == 0 {
				t.Fatalf("no choices in response")
			}

			choice := completion.Choices[0]
			message := choice.Message

			if len(message.ToolCalls) == 0 {
				finishReason := choice.FinishReason
				if finishReason == "" {
					finishReason = "unknown"
				}
				content := message.Content
				if content == "" {
					content = "(empty)"
				}
				t.Logf("User prompt: %q", userContent)
				t.Logf("Finish reason: %s", finishReason)
				t.Logf("Message content: %q", content)
				t.Logf("Tool calls count: %d", len(message.ToolCalls))
				if messageJSON, err := json.MarshalIndent(message, "", "  "); err == nil {
					t.Logf("Full message: %s", string(messageJSON))
				}
				t.Fatalf("expected at least one tool call, got none. Finish reason: %s, Content: %q", finishReason, content)
			}

			firstToolCall := message.ToolCalls[0]
			if firstToolCall.Function.Name != "execute_bash" {
				t.Fatalf("unexpected tool called: got %q want %q", firstToolCall.Function.Name, "execute_bash")
			}

			var args map[string]any
			if firstToolCall.Function.Arguments != "" {
				if err := json.Unmarshal([]byte(firstToolCall.Function.Arguments), &args); err != nil {
					t.Fatalf("failed to parse tool call arguments: %v", err)
				}
			}

			command, ok := args["command"]
			if !ok {
				t.Fatalf("expected tool arguments to include 'command', got: %v", args)
			}

			cmdStr, ok := command.(string)
			if !ok {
				t.Fatalf("expected command to be string, got %T", command)
			}

			if err := validateBashCommand(cmdStr, "ls", []string{"/tmp"}); err != nil {
				t.Errorf("bash command validation failed: %v. Command: %q", err, cmdStr)
			}
		})
	}
}

func TestOpenAIToolCallingBashMultiStep(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	var baseURL string
	var apiKey string
	var modelsToTest []string
	var cleanup func()

	if openaiBaseURL := os.Getenv("OPENAI_BASE_URL"); openaiBaseURL != "" {
		baseURL = openaiBaseURL
		apiKey = os.Getenv("OLLAMA_API_KEY")
		if apiKey == "" {
			t.Fatal("OPENAI_API_KEY must be set when using OPENAI_BASE_URL")
		}
		modelsToTest = cloudModels
		if modelsEnv := os.Getenv("OPENAI_TEST_MODELS"); modelsEnv != "" {
			modelsToTest = []string{modelsEnv}
		}
		cleanup = func() {}
	} else {
		_, testEndpoint, cleanupFn := InitServerConnection(ctx, t)
		cleanup = cleanupFn
		baseURL = fmt.Sprintf("http://%s/v1", testEndpoint)
		apiKey = "ollama"
		modelsToTest = append(agenticModels, cloudModels...)
	}
	t.Cleanup(cleanup)

	opts := []option.RequestOption{
		option.WithBaseURL(baseURL),
		option.WithAPIKey(apiKey),
	}
	openaiClient := openai.NewClient(opts...)

	var ollamaClient *api.Client
	if baseURL == "" {
		ollamaClient, _, _ = InitServerConnection(ctx, t)
	}

	for _, model := range modelsToTest {
		t.Run(model, func(t *testing.T) {
			testCtx := ctx
			if slices.Contains(cloudModels, model) {
				t.Parallel()
				// Create a new context for parallel tests to avoid cancellation
				var cancel context.CancelFunc
				testCtx, cancel = context.WithTimeout(context.Background(), 10*time.Minute)
				defer cancel()
			}
			if v, ok := minVRAM[model]; ok {
				skipUnderMinVRAM(t, v)
			}

			if ollamaClient != nil {
				if err := PullIfMissing(testCtx, ollamaClient, model); err != nil {
					t.Fatalf("pull failed %s", err)
				}
			}

			tools := []openai.ChatCompletionToolUnionParam{
				openai.ChatCompletionFunctionTool(shared.FunctionDefinitionParam{
					Name:        "execute_bash",
					Description: openai.Opt("Execute a bash/shell command and return stdout, stderr, and exit code"),
					Parameters: shared.FunctionParameters{
						"type": "object",
						"properties": map[string]any{
							"command": map[string]any{
								"type":        "string",
								"description": "The bash command to execute",
							},
							"working_directory": map[string]any{
								"type":        "string",
								"description": "Optional working directory for command execution",
							},
						},
						"required": []string{"command"},
					},
				}),
			}

			userContent := "Find all log files in /tmp. use the bash tool"
			userMessage := openai.UserMessage(userContent)

			req := openai.ChatCompletionNewParams{
				Model:       shared.ChatModel(model),
				Messages:    []openai.ChatCompletionMessageParamUnion{userMessage},
				Tools:       tools,
				Temperature: openai.Opt(0.0),
			}

			completion, err := openaiClient.Chat.Completions.New(testCtx, req)
			if err != nil {
				t.Fatalf("chat failed: %v", err)
			}

			if len(completion.Choices) == 0 {
				t.Fatalf("no choices in response")
			}

			choice := completion.Choices[0]
			message := choice.Message

			if len(message.ToolCalls) == 0 {
				finishReason := choice.FinishReason
				if finishReason == "" {
					finishReason = "unknown"
				}
				content := message.Content
				if content == "" {
					content = "(empty)"
				}
				t.Logf("User prompt: %q", userContent)
				t.Logf("Finish reason: %s", finishReason)
				t.Logf("Message content: %q", content)
				t.Logf("Tool calls count: %d", len(message.ToolCalls))
				if messageJSON, err := json.MarshalIndent(message, "", "  "); err == nil {
					t.Logf("Full message: %s", string(messageJSON))
				}
				t.Fatalf("expected at least one tool call, got none. Finish reason: %s, Content: %q", finishReason, content)
			}

			firstToolCall := message.ToolCalls[0]
			if firstToolCall.Function.Name != "execute_bash" {
				t.Fatalf("unexpected tool called: got %q want %q", firstToolCall.Function.Name, "execute_bash")
			}

			var args map[string]any
			if firstToolCall.Function.Arguments != "" {
				if err := json.Unmarshal([]byte(firstToolCall.Function.Arguments), &args); err != nil {
					t.Fatalf("failed to parse tool call arguments: %v", err)
				}
			}

			command, ok := args["command"]
			if !ok {
				t.Fatalf("expected tool arguments to include 'command', got: %v", args)
			}

			cmdStr, ok := command.(string)
			if !ok {
				t.Fatalf("expected command to be string, got %T", command)
			}

			if err := validateBashCommandFlexible(cmdStr, []string{"find", "ls"}, []string{"/tmp"}); err != nil {
				t.Errorf("bash command validation failed: %v. Command: %q", err, cmdStr)
			}
		})
	}
}

func TestOpenAIToolCallingBashAmpersand(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	var baseURL string
	var apiKey string
	var modelsToTest []string
	var cleanup func()

	if openaiBaseURL := os.Getenv("OPENAI_BASE_URL"); openaiBaseURL != "" {
		baseURL = openaiBaseURL
		apiKey = os.Getenv("OLLAMA_API_KEY")
		if apiKey == "" {
			t.Fatal("OPENAI_API_KEY must be set when using OPENAI_BASE_URL")
		}
		modelsToTest = cloudModels
		if modelsEnv := os.Getenv("OPENAI_TEST_MODELS"); modelsEnv != "" {
			modelsToTest = []string{modelsEnv}
		}
		cleanup = func() {}
	} else {
		_, testEndpoint, cleanupFn := InitServerConnection(ctx, t)
		cleanup = cleanupFn
		baseURL = fmt.Sprintf("http://%s/v1", testEndpoint)
		apiKey = "ollama"
		modelsToTest = append(agenticModels, cloudModels...)
	}
	t.Cleanup(cleanup)

	opts := []option.RequestOption{
		option.WithBaseURL(baseURL),
		option.WithAPIKey(apiKey),
	}
	openaiClient := openai.NewClient(opts...)

	var ollamaClient *api.Client
	if baseURL == "" {
		ollamaClient, _, _ = InitServerConnection(ctx, t)
	}

	for _, model := range modelsToTest {
		t.Run(model, func(t *testing.T) {
			testCtx := ctx
			if slices.Contains(cloudModels, model) {
				t.Parallel()
				// Create a new context for parallel tests to avoid cancellation
				var cancel context.CancelFunc
				testCtx, cancel = context.WithTimeout(context.Background(), 10*time.Minute)
				defer cancel()
			}
			if v, ok := minVRAM[model]; ok {
				skipUnderMinVRAM(t, v)
			}

			if ollamaClient != nil {
				if err := PullIfMissing(testCtx, ollamaClient, model); err != nil {
					t.Fatalf("pull failed %s", err)
				}
			}

			tools := []openai.ChatCompletionToolUnionParam{
				openai.ChatCompletionFunctionTool(shared.FunctionDefinitionParam{
					Name:        "execute_bash",
					Description: openai.Opt("Execute a bash/shell command and return stdout, stderr, and exit code"),
					Parameters: shared.FunctionParameters{
						"type": "object",
						"properties": map[string]any{
							"command": map[string]any{
								"type":        "string",
								"description": "The bash command to execute",
							},
							"working_directory": map[string]any{
								"type":        "string",
								"description": "Optional working directory for command execution",
							},
						},
						"required": []string{"command"},
					},
				}),
			}

			userContent := "Echo the text 'A & B' using bash with the bash tool"
			userMessage := openai.UserMessage(userContent)

			req := openai.ChatCompletionNewParams{
				Model:       shared.ChatModel(model),
				Messages:    []openai.ChatCompletionMessageParamUnion{userMessage},
				Tools:       tools,
				Temperature: openai.Opt(0.0),
			}

			completion, err := openaiClient.Chat.Completions.New(testCtx, req)
			if err != nil {
				t.Fatalf("chat failed: %v", err)
			}

			if len(completion.Choices) == 0 {
				t.Fatalf("no choices in response")
			}

			choice := completion.Choices[0]
			message := choice.Message

			if len(message.ToolCalls) == 0 {
				finishReason := choice.FinishReason
				if finishReason == "" {
					finishReason = "unknown"
				}
				content := message.Content
				if content == "" {
					content = "(empty)"
				}
				t.Logf("User prompt: %q", userContent)
				t.Logf("Finish reason: %s", finishReason)
				t.Logf("Message content: %q", content)
				t.Logf("Tool calls count: %d", len(message.ToolCalls))
				if messageJSON, err := json.MarshalIndent(message, "", "  "); err == nil {
					t.Logf("Full message: %s", string(messageJSON))
				}
				t.Fatalf("expected at least one tool call, got none. Finish reason: %s, Content: %q", finishReason, content)
			}

			firstToolCall := message.ToolCalls[0]
			if firstToolCall.Function.Name != "execute_bash" {
				t.Fatalf("unexpected tool called: got %q want %q", firstToolCall.Function.Name, "execute_bash")
			}

			var args map[string]any
			if firstToolCall.Function.Arguments != "" {
				if err := json.Unmarshal([]byte(firstToolCall.Function.Arguments), &args); err != nil {
					t.Fatalf("failed to parse tool call arguments: %v", err)
				}
			}

			command, ok := args["command"]
			if !ok {
				t.Fatalf("expected tool arguments to include 'command', got: %v", args)
			}

			cmdStr, ok := command.(string)
			if !ok {
				t.Fatalf("expected command to be string, got %T", command)
			}

			if !strings.Contains(cmdStr, "&") {
				t.Errorf("expected command to contain '&' character for parsing test, got: %q", cmdStr)
			}

			if !strings.Contains(cmdStr, "echo") && !strings.Contains(cmdStr, "printf") {
				t.Errorf("expected command to use echo or printf, got: %q", cmdStr)
			}
		})
	}
}
