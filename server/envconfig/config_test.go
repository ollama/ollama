package envconfig

import (
	"os"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestConfig(t *testing.T) {
	os.Setenv("OLLAMA_DEBUG", "")
	LoadConfig()
	require.False(t, Debug)
	os.Setenv("OLLAMA_DEBUG", "false")
	LoadConfig()
	require.False(t, Debug)
	os.Setenv("OLLAMA_DEBUG", "1")
	LoadConfig()
	require.True(t, Debug)
}
