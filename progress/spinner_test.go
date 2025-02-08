package progress

import (
    "testing"
    "strings"
    "time"
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

// Test generated using Keploy
func TestString_MessageTruncation(t *testing.T) {
    spinner := NewSpinner("This is a very long message that should be truncated")
    spinner.messageWidth = 20
    result := spinner.String()

    if !strings.Contains(result, "This is a very long") {
        t.Errorf("Expected result to contain truncated message 'This is a very long', got '%s'", result)
    }
}

// Test generated using Keploy
func TestStart_IncrementsValueAndStops(t *testing.T) {
    spinner := NewSpinner("Test Start")
    spinner.ticker = time.NewTicker(10 * time.Millisecond) // Use a shorter interval for testing
    defer spinner.ticker.Stop()

    // Run the start method in a separate goroutine
    go spinner.start()

    // Allow the ticker to tick a few times
    time.Sleep(50 * time.Millisecond)

    // Stop the spinner
    spinner.Stop()

    // Capture the value after stopping
    finalValue := spinner.value

    // Allow some time to ensure the goroutine exits
    time.Sleep(20 * time.Millisecond)

    if spinner.stopped.IsZero() {
        t.Errorf("Expected spinner to be stopped, but it is not")
    }

    if finalValue == 0 {
        t.Errorf("Expected spinner value to increment, but it did not")
    }
}


// Test generated using Keploy
func TestStart_NoTickerNoIncrement(t *testing.T) {
    spinner := NewSpinner("Test Start")
    spinner.ticker = nil // Ensure ticker is nil

    // Run the start method in a separate goroutine
    go spinner.start()

    // Allow some time for the goroutine to run
    time.Sleep(50 * time.Millisecond)

    // Stop the spinner
    spinner.Stop()

    if spinner.value != 0 {
        t.Errorf("Expected spinner value to remain 0 when ticker is nil, but got %d", spinner.value)
    }
}



