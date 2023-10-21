package server

import (
	"os"
	"testing"
	"time"
)

func TestDefaultSessionDuration(t *testing.T) {
	const defaultValue = 5 * time.Minute
	testCases := []struct {
		env  string
		want time.Duration
	}{
		{"", defaultValue},
		{"haha hihi", defaultValue},
		{"-42", defaultValue},
		{"0", 0 * time.Second},
		{"42", 42 * time.Second},
	}

	for i, testCase := range testCases {
		err := os.Setenv(defaultSessionDurationEnvVar, testCase.env)
		if err != nil {
			t.Fatalf("could not set env var %q: %s", defaultSessionDurationEnvVar, err)
		}

		got := defaultSessionDuration()
		if got != testCase.want {
			t.Errorf("test case[%d]: got %q, want %q", i, got, testCase.want)
		}
	}
}
