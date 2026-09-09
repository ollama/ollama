package tools

// ThinkTool is a no-op reasoning primitive that allows a model to plan before taking an external action.
// It simply validates that a `thought` string is provided and returns an "ok" status.
//
// Usage example (JSON input):
// { "thought": "I should check the weather before asking for a hike." }

import (
    "encoding/json"
    "errors"
    "fmt"
)

type ThinkInput struct {
    Thought string `json:"thought"`
}

type ThinkOutput struct {
    Status string `json:"status"`
}

// Execute validates the input and returns a simple success response.
func Execute(inputJSON []byte) (string, error) {
    var in ThinkInput
    if err := json.Unmarshal(inputJSON, &in); err != nil {
        return "", fmt.Errorf("invalid JSON: %w", err)
    }
    if in.Thought == "" {
        return "", errors.New("thought cannot be empty")
    }
    out := ThinkOutput{Status: "ok"}
    result, err := json.Marshal(out)
    if err != nil {
        return "", fmt.Errorf("failed to marshal output: %w", err)
    }
    return string(result), nil
}

// Register adds the tool to the global registry. The registry implementation
// lives in `x/tools/registry.go` (if it exists) – otherwise this is a stub
// placeholder for future integration.
func Register() {
    // Placeholder: actual registration logic will depend on the existing tool registry.
}
