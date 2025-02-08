package progress

import (
    "testing"
    "time"

    "strings"


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

// Test generated using Keploy
func TestBarRate_BarStopped(t *testing.T) {
    bar := NewBar("Test Message", 100, 0)
    bar.Set(50)
    bar.stopped = bar.started.Add(2 * time.Second) // Simulate stopping after 2 seconds

    rate := bar.rate()
    expectedRate := 25.0 // 50 units over 2 seconds
    if rate != expectedRate {
        t.Errorf("Expected rate %f, got %f", expectedRate, rate)
    }
}


// Test generated using Keploy
func TestBarRate_NotStarted(t *testing.T) {
    bar := NewBar("Test Message", 100, 0)
    rate := bar.rate()
    if rate != 0 {
        t.Errorf("Expected rate to be 0 for a bar that has not started, got %f", rate)
    }
}


// Test generated using Keploy
func TestBarString_EmptyProgress(t *testing.T) {
    bar := NewBar("Test Message", 100, 0)
    result := bar.String()
    if !strings.Contains(result, "Test Message") {
        t.Errorf("Expected result to contain 'Test Message', got '%s'", result)
    }
    if !strings.Contains(result, "0%") {
        t.Errorf("Expected result to contain '0%%', got '%s'", result)
    }
}



// Test generated using Keploy
func TestBarString_StoppedBar(t *testing.T) {
    bar := NewBar("Test Message", 100, 100)
    bar.stopped = time.Now() // Simulate a stopped bar

    result := bar.String()

    if !strings.Contains(result, "100%") {
        t.Errorf("Expected result to contain '100%%', got '%s'", result)
    }
    if !strings.Contains(result, "Test Message") {
        t.Errorf("Expected result to contain 'Test Message', got '%s'", result)
    }
}


// Test generated using Keploy
func TestBarRate_MultipleBuckets(t *testing.T) {
    bar := NewBar("Test Message", 100, 0)
    bar.buckets = []bucket{
        {updated: bar.started.Add(1 * time.Second), value: 10},
        {updated: bar.started.Add(3 * time.Second), value: 30},
    }

    rate := bar.rate()
    expectedRate := 10.0 // (30 - 10) / (3 - 1)

    if rate != expectedRate {
        t.Errorf("Expected rate %f, got %f", expectedRate, rate)
    }
}


// Test generated using Keploy
func TestBarSet_ExceedMaxValue(t *testing.T) {
    bar := NewBar("Test Message", 100, 0)
    bar.Set(150) // Exceeding max value

    if bar.currentValue != 100 {
        t.Errorf("Expected currentValue to be capped at 100, got %d", bar.currentValue)
    }
}

