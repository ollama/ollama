package envconfig

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestConfig(t *testing.T) {
	Debug = false // Reset whatever was loaded in init()
	t.Setenv("OLLAMA_DEBUG", "")
	LoadConfig()
	require.False(t, Debug)
	t.Setenv("OLLAMA_DEBUG", "false")
	LoadConfig()
	require.False(t, Debug)
	t.Setenv("OLLAMA_DEBUG", "1")
	LoadConfig()
	require.True(t, Debug)
	t.Setenv("OLLAMA_FLASH_ATTENTION", "1")
	LoadConfig()
	require.True(t, FlashAttention)
}
