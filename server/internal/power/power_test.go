//go:build darwin

package power

import (
	"os/exec"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestPowerAssertion(t *testing.T) {
	// 1. Initial state: should not have assertion
	hasAssertion := checkAssertion()
	assert.False(t, hasAssertion, "Should not have assertion initially")

	// 2. Prevent Sleep
	PreventSleep()
	// Give it a moment to register with OS
	time.Sleep(100 * time.Millisecond)

	hasAssertion = checkAssertion()
	assert.True(t, hasAssertion, "Should have assertion after PreventSleep")

	// 3. Allow Sleep
	AllowSleep()
	time.Sleep(100 * time.Millisecond)

	hasAssertion = checkAssertion()
	assert.False(t, hasAssertion, "Should not have assertion after AllowSleep")
}

func checkAssertion() bool {
	cmd := exec.Command("pmset", "-g", "assertions")
	out, err := cmd.Output()
	if err != nil {
		return false
	}
	return strings.Contains(string(out), "Ollama Inference")
}
