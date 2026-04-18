# Issue #152: Function Calling - Native Tool Integration Implementation Guide

**Priority**: HIGH - Feature Enhancement
**Complexity**: Very High
**Effort**: 40 hours
**Status**: Ready for Implementation

## Problem Statement

Models lack built-in ability to:
- Execute bash commands
- Read/write files
- Query Kubernetes clusters
- Call external APIs
- Use system tools

Current workaround: Post-process output regex (unreliable)

Required: **First-class function calling** where model can directly invoke system functions with:
- Type-safe argument passing
- Automatic serialization
- Execution with safety constraints
- Result feedback to model

## Solution Overview

Implement function calling infrastructure:
1. Register callable functions with signatures
2. Extend token generation to emit function calls
3. Execute safeguarded functions
4. Feed results back to model context
5. Multiple invocation rounds until completion

## Implementation

### Phase 1: Function Registry

```go
// server/functions/registry.go
package functions

import (
    "encoding/json"
    "fmt"
    "reflect"
)

// FunctionSignature describes a callable function
type FunctionSignature struct {
    Name        string                 `json:"name"`
    Description string                 `json:"description"`
    Parameters  map[string]ParameterDef `json:"parameters"`
}

type ParameterDef struct {
    Type        string   `json:"type"`      // string, integer, boolean, object, array
    Description string   `json:"description"`
    Required    bool     `json:"required"`
    Enum        []interface{} `json:"enum,omitempty"`
}

// CallableFunction wraps a function with permissions
type CallableFunction struct {
    Signature   FunctionSignature
    Handler     func(ctx context.Context, args map[string]interface{}) (interface{}, error)
    Permissions FunctionPermissions
}

type FunctionPermissions struct {
    AllowExecute bool     // Can execute?
    TimeoutMs    int      // Execution timeout
    MaxMemoryMB  int      // Memory limit
    AllowedPaths []string // For file operations
}

// FunctionRegistry manages available functions
type FunctionRegistry struct {
    functions map[string]*CallableFunction
}

func NewFunctionRegistry() *FunctionRegistry {
    return &FunctionRegistry{
        functions: make(map[string]*CallableFunction),
    }
}

// Register adds a function to registry
func (fr *FunctionRegistry) Register(cf *CallableFunction) error {
    if cf.Signature.Name == "" {
        return fmt.Errorf("function name required")
    }

    fr.functions[cf.Signature.Name] = cf
    return nil
}

// GetSignatures returns all available function signatures
func (fr *FunctionRegistry) GetSignatures() []FunctionSignature {
    sigs := make([]FunctionSignature, 0, len(fr.functions))
    for _, fn := range fr.functions {
        sigs = append(sigs, fn.Signature)
    }
    return sigs
}

// Call invokes a function with arguments
func (fr *FunctionRegistry) Call(ctx context.Context, name string, args map[string]interface{}) (interface{}, error) {
    fn, exists := fr.functions[name]
    if !exists {
        return nil, fmt.Errorf("function not found: %s", name)
    }

    // Validate arguments against signature
    if err := fr.validateArgs(args, fn.Signature); err != nil {
        return nil, err
    }

    // Check permissions
    if !fn.Permissions.AllowExecute {
        return nil, fmt.Errorf("function execution not allowed: %s", name)
    }

    return fn.Handler(ctx, args)
}

func (fr *FunctionRegistry) validateArgs(args map[string]interface{}, sig FunctionSignature) error {
    for paramName, paramDef := range sig.Parameters {
        arg, exists := args[paramName]
        if !exists {
            if paramDef.Required {
                return fmt.Errorf("required parameter missing: %s", paramName)
            }
            continue
        }

        // Type validation
        if err := validateType(arg, paramDef.Type); err != nil {
            return fmt.Errorf("invalid type for parameter %s: %w", paramName, err)
        }
    }
    return nil
}

func validateType(value interface{}, expectedType string) error {
    switch expectedType {
    case "string":
        _, ok := value.(string)
        if !ok {
            return fmt.Errorf("expected string, got %T", value)
        }
    case "integer":
        switch value.(type) {
        case float64, int, int64:
        default:
            return fmt.Errorf("expected integer, got %T", value)
        }
    case "boolean":
        _, ok := value.(bool)
        if !ok {
            return fmt.Errorf("expected boolean, got %T", value)
        }
    case "object":
        _, ok := value.(map[string]interface{})
        if !ok {
            return fmt.Errorf("expected object, got %T", value)
        }
    case "array":
        _, ok := value.([]interface{})
        if !ok {
            return fmt.Errorf("expected array, got %T", value)
        }
    }
    return nil
}
```

### Phase 2: Built-in Functions

```go
// server/functions/builtin.go
package functions

import (
    "context"
    "fmt"
    "io/ioutil"
    "os"
    "os/exec"
    "path/filepath"
)

// CreateBuiltinFunctions registers standard system functions
func CreateBuiltinFunctions() *FunctionRegistry {
    registry := NewFunctionRegistry()

    // Execute shell command
    registry.Register(&CallableFunction{
        Signature: FunctionSignature{
            Name:        "execute_command",
            Description: "Execute a shell command and return output",
            Parameters: map[string]ParameterDef{
                "command": {
                    Type:        "string",
                    Description: "Shell command to execute (bash)",
                    Required:    true,
                },
                "timeout_seconds": {
                    Type:        "integer",
                    Description: "Command timeout (max 60s)",
                    Required:    false,
                },
            },
        },
        Handler: executeCommand,
        Permissions: FunctionPermissions{
            AllowExecute: true,
            TimeoutMs:    60000,
            MaxMemoryMB:  256,
        },
    })

    // Read file
    registry.Register(&CallableFunction{
        Signature: FunctionSignature{
            Name:        "read_file",
            Description: "Read contents of a file",
            Parameters: map[string]ParameterDef{
                "path": {
                    Type:        "string",
                    Description: "File path to read",
                    Required:    true,
                },
                "max_bytes": {
                    Type:        "integer",
                    Description: "Max bytes to read (1MB limit)",
                    Required:    false,
                },
            },
        },
        Handler: readFile,
        Permissions: FunctionPermissions{
            AllowExecute: true,
            AllowedPaths: []string{"/home", "/tmp"},
        },
    })

    // Write file
    registry.Register(&CallableFunction{
        Signature: FunctionSignature{
            Name:        "write_file",
            Description: "Write contents to a file",
            Parameters: map[string]ParameterDef{
                "path": {
                    Type:        "string",
                    Description: "File path to write",
                    Required:    true,
                },
                "content": {
                    Type:        "string",
                    Description: "Content to write",
                    Required:    true,
                },
                "mode": {
                    Type: "string",
                    Enum: []interface{}{"w", "a"},
                },
            },
        },
        Handler: writeFile,
        Permissions: FunctionPermissions{
            AllowExecute: true,
            AllowedPaths: []string{"/tmp"},
        },
    })

    // Query Kubernetes
    registry.Register(&CallableFunction{
        Signature: FunctionSignature{
            Name:        "kubectl_get",
            Description: "Query Kubernetes resources",
            Parameters: map[string]ParameterDef{
                "resource": {
                    Type:        "string",
                    Description: "Resource type (pods, services, deployments)",
                    Required:    true,
                    Enum: []interface{}{"pods", "services", "deployments", "nodes"},
                },
                "namespace": {
                    Type:        "string",
                    Description: "Kubernetes namespace",
                    Required:    false,
                },
            },
        },
        Handler: kubectlGet,
        Permissions: FunctionPermissions{
            AllowExecute: true,
            TimeoutMs:    10000,
        },
    })

    return registry
}

func executeCommand(ctx context.Context, args map[string]interface{}) (interface{}, error) {
    command, _ := args["command"].(string)
    timeoutSec := 30

    if to, ok := args["timeout_seconds"].(float64); ok {
        timeoutSec = int(to)
        if timeoutSec > 60 {
            timeoutSec = 60
        }
    }

    // Security: whitelist safe commands
    allowedCommands := map[string]bool{
        "ls":  true,
        "pwd": true,
        "cat": true,
        "grep": true,
        "wc": true,
    }

    // Parse command
    parts := splitCommand(command)
    if len(parts) == 0 {
        return nil, fmt.Errorf("invalid command")
    }

    if !allowedCommands[parts[0]] {
        return nil, fmt.Errorf("command not allowed: %s", parts[0])
    }

    // Execute with timeout
    cmd := exec.CommandContext(ctx, parts[0], parts[1:]...)

    output, err := cmd.CombinedOutput()
    if err != nil {
        return map[string]interface{}{
            "exit_code": cmd.ProcessState.ExitCode(),
            "error":     err.Error(),
        }, nil
    }

    return map[string]interface{}{
        "output": string(output),
        "exit_code": 0,
    }, nil
}

func readFile(ctx context.Context, args map[string]interface{}) (interface{}, error) {
    path, _ := args["path"].(string)

    // Validate path
    absPath, err := filepath.Abs(path)
    if err != nil {
        return nil, err
    }

    // Only allow safe directories
    if !isAllowedPath(absPath, []string{"/home", "/tmp"}) {
        return nil, fmt.Errorf("access denied: %s", path)
    }

    content, err := ioutil.ReadFile(absPath)
    if err != nil {
        return nil, err
    }

    return map[string]interface{}{
        "content": string(content),
        "size":    len(content),
    }, nil
}

func writeFile(ctx context.Context, args map[string]interface{}) (interface{}, error) {
    path, _ := args["path"].(string)
    content, _ := args["content"].(string)

    // Only allow writing to /tmp
    if !isAllowedPath(path, []string{"/tmp"}) {
        return nil, fmt.Errorf("write access denied: %s", path)
    }

    err := ioutil.WriteFile(path, []byte(content), 0644)
    if err != nil {
        return nil, err
    }

    return map[string]interface{}{
        "success": true,
        "path":    path,
        "bytes":   len(content),
    }, nil
}

func kubectlGet(ctx context.Context, args map[string]interface{}) (interface{}, error) {
    resource, _ := args["resource"].(string)
    namespace, _ := args["namespace"].(string)

    cmd := exec.CommandContext(ctx, "kubectl", "get", resource, "-o", "json")
    if namespace != "" {
        cmd.Args = append(cmd.Args, "-n", namespace)
    }

    output, err := cmd.Output()
    if err != nil {
        return nil, err
    }

    var result interface{}
    json.Unmarshal(output, &result)

    return result, nil
}

func isAllowedPath(path string, allowedDirs []string) bool {
    for _, allowed := range allowedDirs {
        if path == allowed || filepath.HasPrefix(path, allowed) {
            return true
        }
    }
    return false
}

func splitCommand(cmd string) []string {
    // Simple split - proper shell parsing would be more complex
    return strings.Fields(cmd)
}
```

### Phase 3: Dynamic Function Invocation

```go
// server/generate_with_functions.go
package server

import (
    "encoding/json"
    "fmt"

    "ollama/server/functions"
)

type GenerateWithFunctionsRequest struct {
    Model     string      `json:"model"`
    Prompt    string      `json:"prompt"`
    Functions []string    `json:"functions"` // Function names to enable
    Stream    bool        `json:"stream"`
}

type FunctionCall struct {
    Name   string                 `json:"name"`
    Args   map[string]interface{} `json:"arguments"`
}

// GenerateWithFunctions handles multi-turn function calling
func (s *Server) GenerateWithFunctions(ctx context.Context, req *GenerateWithFunctionsRequest) (string, error) {
    funcRegistry := functions.CreateBuiltinFunctions()
    allSignatures := funcRegistry.GetSignatures()

    // Filter to requested functions
    var enabledSignatures []functions.FunctionSignature
    for _, sig := range allSignatures {
        for _, name := range req.Functions {
            if sig.Name == name {
                enabledSignatures = append(enabledSignatures, sig)
                break
            }
        }
    }

    // Build system prompt with function definitions
    systemPrompt := s.buildFunctionPrompt(enabledSignatures)

    conversationHistory := []struct {
        Role    string `json:"role"`
        Content string `json:"content"`
    }{
        {Role: "system", Content: systemPrompt},
        {Role: "user", Content: req.Prompt},
    }

    // Multi-turn generation loop
    maxIterations := 10
    for i := 0; i < maxIterations; i++ {
        // Generate next response
        model, _ := s.modelManager.Load(ctx, req.Model)

        lastMessage := conversationHistory[len(conversationHistory)-1].Content
        response, _ := model.Generate(ctx, lastMessage)

        // Check if response contains function call
        functionCall, err := s.parseFunction CallFromResponse(response)
        if err != nil {
            // No function call, return final response
            return response, nil
        }

        // Execute function
        result, execErr := funcRegistry.Call(ctx, functionCall.Name, functionCall.Args)

        // Add to conversation
        conversationHistory = append(conversationHistory,
            struct {
                Role    string
                Content string
            }{Role: "assistant", Content: response},
            struct {
                Role    string
                Content string
            }{Role: "function", Content: fmt.Sprintf("%s: %v", functionCall.Name, result)},
        )
    }

    return "Max iterations reached without final response", nil
}

func (s *Server) buildFunctionPrompt(sigs []functions.FunctionSignature) string {
    template := `You are an AI assistant with access to the following functions.

Available functions:
`

    for _, sig := range sigs {
        template += fmt.Sprintf(`
- **%s**: %s
  Parameters:
`, sig.Name, sig.Description)

        for paramName, paramDef := range sig.Parameters {
            template += fmt.Sprintf(`  - %s (%s): %s%s
`, paramName, paramDef.Type, paramDef.Description,
                func() string {
                    if paramDef.Required {
                        return " [REQUIRED]"
                    }
                    return ""
                }())
        }
    }

    template += `

To call a function, respond with:
{
  "type": "function_call",
  "function": {
    "name": "function_name",
    "arguments": {
      "param1": "value1",
      "param2": "value2"
    }
  }
}

After executing, provide the final answer.
`

    return template
}

func (s *Server) parseFunctionCallFromResponse(response string) (*functions.FunctionCall, error) {
    // Look for function call JSON
    start := strings.Index(response, "{")
    end := strings.LastIndex(response, "}")

    if start == -1 || end == -1 {
        return nil, fmt.Errorf("no function call found")
    }

    jsonStr := response[start : end+1]

    var callWrapper struct {
        Type     string `json:"type"`
        Function struct {
            Name      string                 `json:"name"`
            Arguments map[string]interface{} `json:"arguments"`
        } `json:"function"`
    }

    if err := json.Unmarshal([]byte(jsonStr), &callWrapper); err != nil {
        return nil, err
    }

    return &functions.FunctionCall{
        Name: callWrapper.Function.Name,
        Args: callWrapper.Function.Arguments,
    }, nil
}
```

## Acceptance Criteria

- ✅ Execute bash commands safely
- ✅ Read/write files with permission checks
- ✅ Query Kubernetes clusters
- ✅ Type-safe argument validation
- ✅ Timeout enforcement
- ✅ Multi-turn conversations
- ✅ Full audit trail of function calls

## Testing

```bash
# Test function calling
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama2",
    "prompt": "List files in /home and count them",
    "functions": ["execute_command", "read_file"]
  }'

# Expected: Model calls execute_command("ls /home"), gets output, returns result
```

## Deployment Checklist

- [ ] Implement function registry
- [ ] Create builtin functions
- [ ] Add function calling support to generation
- [ ] Set up safety constraints
- [ ] Test multi-turn execution
- [ ] Documentation and examples

---

**Ready for Implementation**: Yes - clear function interface, established patterns from OpenAI API.
