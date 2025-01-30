package progress

import (
    "testing"
    "time"

)


// Test generated using Keploy
func TestNewBar_ValidInputs(t *testing.T) {
    bar := NewBar("Test Message", 100, 10)
    if bar == nil {
        t.Fatalf("Expected non-nil Bar instance, got nil")
    }
    if bar.message != "Test Message" {
        t.Errorf("Expected message 'Test Message', got '%s'", bar.message)
    }
    if bar.maxValue != 100 {
        t.Errorf("Expected maxValue 100, got %d", bar.maxValue)
    }
    if bar.currentValue != 10 {
        t.Errorf("Expected currentValue 10, got %d", bar.currentValue)
    }
}

// Test generated using Keploy
func TestBarPercent_ZeroMaxValue(t *testing.T) {
    bar := NewBar("Test Message", 0, 0)
    percent := bar.percent()
    if percent != 0 {
        t.Errorf("Expected percent 0, got %f", percent)
    }
}


// Test generated using Keploy
func TestBarSet_MaxValueReached(t *testing.T) {
    bar := NewBar("Test Message", 100, 0)
    bar.Set(100)
    if bar.currentValue != 100 {
        t.Errorf("Expected currentValue 100, got %d", bar.currentValue)
    }
    if bar.stopped.IsZero() {
        t.Errorf("Expected stopped time to be set, got zero value")
    }
}


// Test generated using Keploy
func TestRepeat_NegativeCount(t *testing.T) {
    result := repeat("x", -1)
    if result != "" {
        t.Errorf("Expected empty string for negative count, got '%s'", result)
    }
}


// Test generated using Keploy
func TestFormatDuration_Over100Hours(t *testing.T) {
    duration := 101 * time.Hour
    result := formatDuration(duration)
    if result != "99h+" {
        t.Errorf("Expected '99h+', got '%s'", result)
    }
}


// Test generated using Keploy
func TestBarPercent_NonZeroValues(t *testing.T) {
    bar := NewBar("Test Message", 100, 25)
    percent := bar.percent()
    if percent != 25 {
        t.Errorf("Expected percent 25, got %f", percent)
    }
}


// Test generated using Keploy
func TestRepeat_PositiveCount(t *testing.T) {
    result := repeat("x", 5)
    expected := "xxxxx"
    if result != expected {
        t.Errorf("Expected '%s', got '%s'", expected, result)
    }
}


// Test generated using Keploy
func TestFormatDuration_OneSecond(t *testing.T) {
    duration := 1 * time.Second
    result := formatDuration(duration)
    if result != "1s" {
        t.Errorf("Expected '1s', got '%s'", result)
    }
}

