package server

import (
	"errors"
	"net/http"
	"strings"
	"testing"
)

// A malformed client "options" value is a client error (400), not a server error (500).
func TestScheduleErrorInvalidRequestOptionIs400(t *testing.T) {
	var s Server
	for _, opts := range []map[string]any{
		{"temperature": "hot"},
		{"num_ctx": "big"},
		{"stop": "x"},
		{"use_mmap": "yes"},
	} {
		_, err := s.modelOptions(nil, opts)
		if err == nil {
			t.Fatalf("expected error for %v", opts)
		}
		status, msg := scheduleError("m", err)
		if status != http.StatusBadRequest {
			t.Errorf("options %v: status=%d (%s), want 400", opts, status, msg)
		}
		if !strings.Contains(msg, "must be of type") {
			t.Errorf("options %v: message %q lost its detail", opts, msg)
		}
	}
}

// Unrelated errors must keep mapping to 500 so real server faults are not masked.
func TestScheduleErrorOtherErrorsStay500(t *testing.T) {
	if status, _ := scheduleError("m", errors.New("boom")); status != http.StatusInternalServerError {
		t.Fatalf("status=%d, want 500", status)
	}
}

func TestModelOptionsValid(t *testing.T) {
	var s Server
	opts, err := s.modelOptions(nil, map[string]any{"temperature": 0.3, "num_predict": float64(12)})
	if err != nil {
		t.Fatal(err)
	}
	if opts.Temperature != 0.3 || opts.NumPredict != 12 {
		t.Fatalf("unexpected options %+v", opts)
	}
}
