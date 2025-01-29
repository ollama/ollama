package llm

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestParseFileType(t *testing.T) {
	for str, expectedType := range fileTypeMap {
		parsedType, err := ParseFileType(str)
		require.NoError(t, err, "Unexpected error parsing file type: %s", str)
		assert.Equal(t, expectedType, parsedType, "Parsed file type does not match for %s", str)
	}

	unknownType := "UNKNOWN"
	parsedType, err := ParseFileType(unknownType)
	assert.Error(t, err, "Expected error for unknown file type")
	assert.Equal(t, fileTypeUnknown, parsedType, "Parsed file type for unknown input should be fileTypeUnknown")
}

func TestFileTypeString(t *testing.T) {
	for str, ft := range fileTypeMap {
		assert.Equal(t, str, ft.String(), "String representation mismatch for fileType %v", ft)
	}
}

func TestFileTypeMapOneToOne(t *testing.T) {
	reverseMap := make(map[fileType]string)

	for key, value := range fileTypeMap {
		if existingKey, exists := reverseMap[value]; exists {
			t.Errorf("fileType %v is mapped to multiple keys: %s and %s", value, existingKey, key)
		}

		reverseMap[value] = key
	}

	for value, key := range reverseMap {
		assert.Equal(t, value, fileTypeMap[key], "Reverse mapping mismatch for key: %s", key)
	}
}
