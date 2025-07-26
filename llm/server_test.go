package llm

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"golang.org/x/sync/semaphore"
)

func TestLLMServerCompletionFormat(t *testing.T) {
	// This test was written to fix an already deployed issue. It is a bit
	// of a mess, and but it's good enough, until we can refactoring the
	// Completion method to be more testable.

	ctx, cancel := context.WithCancel(t.Context())
	s := &llmServer{
		sem: semaphore.NewWeighted(1), // required to prevent nil panic
	}

	checkInvalid := func(format string) {
		t.Helper()
		err := s.Completion(ctx, CompletionRequest{
			Options: new(api.Options),
			Format:  []byte(format),
		}, nil)

		want := fmt.Sprintf("invalid format: %q; expected \"json\" or a valid JSON Schema", format)
		if err == nil || !strings.Contains(err.Error(), want) {
			t.Fatalf("err = %v; want %q", err, want)
		}
	}

	checkInvalid("X")   // invalid format
	checkInvalid(`"X"`) // invalid JSON Schema

	cancel() // prevent further processing if request makes it past the format check

	checkValid := func(err error) {
		t.Helper()
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("Completion: err = %v; expected context.Canceled", err)
		}
	}

	valids := []string{
		// "missing"
		``,
		`""`,
		`null`,

		// JSON
		`"json"`,
		`{"type":"object"}`,
	}
	for _, valid := range valids {
		err := s.Completion(ctx, CompletionRequest{
			Options: new(api.Options),
			Format:  []byte(valid),
		}, nil)
		checkValid(err)
	}

	err := s.Completion(ctx, CompletionRequest{
		Options: new(api.Options),
		Format:  nil, // missing format
	}, nil)
	checkValid(err)
}

func TestUnicodeBufferHandler(t *testing.T) {
	tests := []struct {
		name              string
		inputResponses    []CompletionResponse
		expectedResponses []CompletionResponse
		description       string
	}{
		{
			name: "complete_unicode",
			inputResponses: []CompletionResponse{
				{Content: "Hello", Done: false},
				{Content: " world", Done: false},
				{Content: "!", Done: true},
			},
			expectedResponses: []CompletionResponse{
				{Content: "Hello", Done: false},
				{Content: " world", Done: false},
				{Content: "!", Done: true},
			},
			description: "All responses with valid unicode should pass through unchanged",
		},
		{
			name: "incomplete_unicode_at_end_with_done",
			inputResponses: []CompletionResponse{
				{Content: "Hello", Done: false},
				{Content: string([]byte{0xF0, 0x9F}), Done: true}, // Incomplete emoji with Done=true
			},
			expectedResponses: []CompletionResponse{
				{Content: "Hello", Done: false},
				{Content: "", Done: true}, // Content is trimmed but response is still sent with Done=true
			},
			description: "When Done=true, incomplete Unicode at the end should be trimmed",
		},
		{
			name: "split_unicode_across_responses",
			inputResponses: []CompletionResponse{
				{Content: "Hello " + string([]byte{0xF0, 0x9F}), Done: false}, // First part of 😀
				{Content: string([]byte{0x98, 0x80}) + " world!", Done: true}, // Second part of 😀 and more text
			},
			expectedResponses: []CompletionResponse{
				{Content: "Hello ", Done: false},  // Incomplete Unicode trimmed
				{Content: "😀 world!", Done: true}, // Complete emoji in second response
			},
			description: "Unicode split across responses should be handled correctly",
		},
		{
			name: "incomplete_unicode_buffered",
			inputResponses: []CompletionResponse{
				{Content: "Test " + string([]byte{0xF0, 0x9F}), Done: false}, // Incomplete emoji
				{Content: string([]byte{0x98, 0x80}), Done: false},           // Complete the emoji
				{Content: " done", Done: true},
			},
			expectedResponses: []CompletionResponse{
				{Content: "Test ", Done: false}, // First part without incomplete unicode
				{Content: "😀", Done: false},     // Complete emoji
				{Content: " done", Done: true},
			},
			description: "Incomplete unicode should be buffered and combined with next response",
		},
		{
			name: "empty_response_with_done",
			inputResponses: []CompletionResponse{
				{Content: "Complete response", Done: false},
				{Content: "", Done: true}, // Empty response with Done=true
			},
			expectedResponses: []CompletionResponse{
				{Content: "Complete response", Done: false},
				{Content: "", Done: true}, // Should still be sent because Done=true
			},
			description: "Empty final response with Done=true should still be sent",
		},
		{
			name: "done_reason_preserved",
			inputResponses: []CompletionResponse{
				{Content: "Response", Done: false},
				{Content: " complete", Done: true, DoneReason: DoneReasonStop},
			},
			expectedResponses: []CompletionResponse{
				{Content: "Response", Done: false},
				{Content: " complete", Done: true, DoneReason: DoneReasonStop},
			},
			description: "DoneReason should be preserved in the final response",
		},
		{
			name: "only_incomplete_unicode_not_done",
			inputResponses: []CompletionResponse{
				{Content: string([]byte{0xF0, 0x9F}), Done: false}, // Only incomplete unicode
			},
			expectedResponses: []CompletionResponse{
				// No response expected - should be buffered
			},
			description: "Response with only incomplete unicode should be buffered if not done",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var actualResponses []CompletionResponse

			// Create a callback that collects responses
			callback := func(resp CompletionResponse) {
				actualResponses = append(actualResponses, resp)
			}

			// Create the unicode buffer handler
			handler := unicodeBufferHandler(callback)

			// Send all input responses through the handler
			for _, resp := range tt.inputResponses {
				handler(resp)
			}

			// Verify the number of responses
			if len(actualResponses) != len(tt.expectedResponses) {
				t.Fatalf("%s: got %d responses, want %d responses",
					tt.description, len(actualResponses), len(tt.expectedResponses))
			}

			// Verify each response matches the expected one
			for i, expected := range tt.expectedResponses {
				if i >= len(actualResponses) {
					t.Fatalf("%s: missing response at index %d", tt.description, i)
					continue
				}

				actual := actualResponses[i]

				// Verify content
				if actual.Content != expected.Content {
					t.Errorf("%s: response[%d].Content = %q, want %q",
						tt.description, i, actual.Content, expected.Content)
				}

				// Verify Done flag
				if actual.Done != expected.Done {
					t.Errorf("%s: response[%d].Done = %v, want %v",
						tt.description, i, actual.Done, expected.Done)
				}

				// Verify DoneReason if specified
				if actual.DoneReason != expected.DoneReason {
					t.Errorf("%s: response[%d].DoneReason = %v, want %v",
						tt.description, i, actual.DoneReason, expected.DoneReason)
				}
			}
		})
	}
}
