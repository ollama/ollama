package tools

import (
	"context"
	"strings"
	"testing"
)

func TestBashCommand_Name(t *testing.T) {
	cmd := &BashCommand{}
	if name := cmd.Name(); name != "bash_command" {
		t.Errorf("Expected name 'bash_command', got %s", name)
	}
}

func TestBashCommand_Execute(t *testing.T) {
	cmd := &BashCommand{}
	ctx := context.Background()

	tests := []struct {
		name        string
		input       map[string]any
		wantErr     bool
		errContains string
		wantOutput  string
	}{
		{
			name: "valid echo command",
			input: map[string]any{
				"command": "echo 'hello world'",
			},
			wantErr:    false,
			wantOutput: "hello world\n",
		},
		{
			name: "valid ls command",
			input: map[string]any{
				"command": "ls -l",
			},
			wantErr: false,
		},
		{
			name: "invalid command",
			input: map[string]any{
				"command": "rm -rf /",
			},
			wantErr:     true,
			errContains: "command not in allowed list",
		},
		{
			name: "dangerous flag",
			input: map[string]any{
				"command": "find . --delete",
			},
			wantErr:     true,
			errContains: "dangerous flag",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := cmd.Execute(ctx, tt.input)

			if tt.wantErr {
				if err == nil {
					t.Error("Expected error but got none")
				} else if !strings.Contains(err.Error(), tt.errContains) {
					t.Errorf("Expected error containing '%s', got '%s'", tt.errContains, err.Error())
				}
				return
			}

			if err != nil {
				t.Errorf("Unexpected error: %v", err)
				return
			}

			// Check result type and fields
			response, ok := result.(map[string]any)
			if !ok {
				t.Fatal("Expected result to be map[string]any")
			}

			// Check required fields
			success, ok := response["success"].(bool)
			if !ok || !success {
				t.Error("Expected success to be true")
			}

			command, ok := response["command"].(string)
			if !ok || command == "" {
				t.Error("Expected command to be non-empty string")
			}

			output, ok := response["output"].(string)
			if !ok {
				t.Error("Expected output to be string")
			} else if tt.wantOutput != "" && output != tt.wantOutput {
				t.Errorf("Expected output '%s', got '%s'", tt.wantOutput, output)
			}
		})
	}
}

func TestBashCommand_InvalidInput(t *testing.T) {
	cmd := &BashCommand{}
	ctx := context.Background()

	tests := []struct {
		name        string
		input       map[string]any
		errContains string
	}{
		{
			name:        "missing command",
			input:       map[string]any{},
			errContains: "command parameter is required",
		},
		{
			name: "empty command",
			input: map[string]any{
				"command": "",
			},
			errContains: "empty command",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := cmd.Execute(ctx, tt.input)
			if err == nil {
				t.Error("Expected error but got none")
			} else if !strings.Contains(err.Error(), tt.errContains) {
				t.Errorf("Expected error containing '%s', got '%s'", tt.errContains, err.Error())
			}
		})
	}
}

func TestBashCommand_OutputFormat(t *testing.T) {
	cmd := &BashCommand{}
	ctx := context.Background()

	// Test with a simple echo command
	input := map[string]any{
		"command": "echo 'test output'",
	}

	result, err := cmd.Execute(ctx, input)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	// Verify the result is a map[string]any
	response, ok := result.(map[string]any)
	if !ok {
		t.Fatal("Result is not a map[string]any")
	}

	// Check all expected fields exist
	requiredFields := []string{"command", "output", "success"}
	for _, field := range requiredFields {
		if _, ok := response[field]; !ok {
			t.Errorf("Missing required field: %s", field)
		}
	}

	// Verify output is plain text
	output, ok := response["output"].(string)
	if !ok {
		t.Error("Output field is not a string")
	} else {
		// Output should contain 'test output' and a newline
		expectedOutput := "test output\n"
		if output != expectedOutput {
			t.Errorf("Expected output '%s', got '%s'", expectedOutput, output)
		}

		// Verify output is not base64 encoded
		if strings.Contains(output, "base64") ||
			(len(output) > 0 && output[0] == 'e' && strings.ContainsAny(output, "+/=")) {
			t.Error("Output appears to be base64 encoded")
		}
	}
}
