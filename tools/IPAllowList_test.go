package tools

import (
	"fmt"
	"testing"
)

func TestIPAllowList(t *testing.T) {
	// Test empty AllowList (no restrictions)
	w, err := NewIPAllowList("")

	if err != nil {
		t.Fatalf("Expected no error for empty AllowList, got: %v", err)
	}
	if !w.IsAllowed("192.168.1.100") {
		t.Error("Empty AllowList should allow all IPs")
	}

	// Test valid AllowList
	AllowList := "192.168.1.0/24,10.0.0.1,127.0.0.1"
	w, err = NewIPAllowList(AllowList)
	if err != nil {
		t.Fatalf("Failed to create IP AllowList: %v", err)
	}

	tests := []struct {
		ip       string
		expected bool
	}{
		{"192.168.1.100", true},
		{"10.0.0.1", true},
		{"127.0.0.1", true},
		{"10.0.0.2", false},
		{"172.16.0.1", false},
	}

	for _, tc := range tests {
		t.Run(fmt.Sprintf("IP: %s", tc.ip), func(t *testing.T) {
			if got := w.IsAllowed(tc.ip); got != tc.expected {
				t.Errorf("isAllowedIP(%q) = %v, want %v", tc.ip, got, tc.expected)
			}
		})
	}
}
