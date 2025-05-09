package convert

import (
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/goccy/go-yaml"
)

var (
	syntaxError             *yaml.SyntaxError             = new(yaml.SyntaxError)
	unexpectedNodeTypeError *yaml.UnexpectedNodeTypeError = new(yaml.UnexpectedNodeTypeError)
)

const TEST_DATA_PATH = "testdata"
const MODEL_CARD_PATH = "modelcard"

func TestParseHuggingFaceModelCardMetadata(t *testing.T) {
	tests := []struct {
		name          string
		filePath      string
		expectedError error
	}{
		{
			name:          "Valid-Phi-3.5-mini-instruct",
			filePath:      filepath.Join(TEST_DATA_PATH, MODEL_CARD_PATH, "README.md.Phi-3.5-mini-instruct"),
			expectedError: nil,
		},
		{
			name:          "Valid-Qwen2.5-7B-Instruct",
			filePath:      filepath.Join(TEST_DATA_PATH, MODEL_CARD_PATH, "README.md.Qwen2.5-7B-Instruct"),
			expectedError: nil,
		},
		{
			name:          "Valid-tiny-LlamaForCausalLM-3.2",
			filePath:      filepath.Join(TEST_DATA_PATH, MODEL_CARD_PATH, "README.md.tiny-LlamaForCausalLM-3.2"),
			expectedError: nil,
		},
		{
			name:          "Valid-Llama-4-Maverick-17B-128E-Instruct",
			filePath:      filepath.Join(TEST_DATA_PATH, MODEL_CARD_PATH, "README md.Llama-4-Maverick-17B-128E-Instruct"),
			expectedError: nil,
		},
		{
			name:          "Valid-OpenHermes-2-Mistral-7B",
			filePath:      filepath.Join(TEST_DATA_PATH, MODEL_CARD_PATH, "README.md.OpenHermes-2-Mistral-7B"),
			expectedError: nil,
		},
		{
			name:          "Invalid-YAML-empty-tags-array",
			filePath:      filepath.Join(TEST_DATA_PATH, MODEL_CARD_PATH, "README.md.empty_array"),
			expectedError: nil,
		},
		{
			name:          "Invalid-YAML-syntax-error",
			filePath:      filepath.Join(TEST_DATA_PATH, MODEL_CARD_PATH, "README.md.bad_delim"),
			expectedError: syntaxError,
		},
		{
			name:          "Invalid-YAML-missing-delimiter",
			filePath:      filepath.Join(TEST_DATA_PATH, MODEL_CARD_PATH, "README.md.missing_delim"),
			expectedError: unexpectedNodeTypeError,
		},
	}

	// need filesystem relative to this test file
	baseFS := os.DirFS(".")

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {

			var modelcard ModelCardMetadata
			err := parseModelCardMetadata(baseFS, tt.filePath, &modelcard)

			// short-circuit if no error expected and none returned
			if err == nil && tt.expectedError == nil {
				return
			}

			// simple mismatched cases
			if (err != nil && tt.expectedError == nil) ||
				(err == nil && tt.expectedError != nil) {
				t.Errorf("parseModelCardMetadata() error: %T, expectedError: %T", err, tt.expectedError)
			}

			// complex case: wrapped errors of where messages are not fixed
			unwrappedErr := errors.Unwrap(err)
			if unwrappedErr == nil || (reflect.TypeOf(tt.expectedError) != reflect.TypeOf(unwrappedErr)) {
				t.Errorf("parseModelCardMetadata() wrapped error: %T, expectedError: %T", unwrappedErr, tt.expectedError)
			}
		})
	}
}
