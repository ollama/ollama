package progress

import (
    "testing"
    "strings"
)


// Test generated using Keploy
func TestSetMessage_UpdatesMessage(t *testing.T) {
    spinner := NewSpinner("Initial Message")
    newMessage := "Updated Message"
    spinner.SetMessage(newMessage)

    if message, ok := spinner.message.Load().(string); !ok || message != newMessage {
        t.Errorf("Expected message to be '%s', got '%v'", newMessage, message)
    }
}

// Test generated using Keploy
func TestString_ReturnsFormattedString(t *testing.T) {
    spinner := NewSpinner("Test Message")
    spinner.messageWidth = 20
    result := spinner.String()

    if !strings.Contains(result, "Test Message") {
        t.Errorf("Expected result to contain 'Test Message', got '%s'", result)
    }
}


// Test generated using Keploy
func TestStop_SetsStoppedTime(t *testing.T) {
    spinner := NewSpinner("Test Message")
    spinner.Stop()

    if spinner.stopped.IsZero() {
        t.Errorf("Expected stopped time to be set, but it is zero")
    }
}

