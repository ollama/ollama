package types_test

import (
	"encoding/json"
	"testing"

	"github.com/ollama/ollama/types"
)

func TestNull(t *testing.T) {
	var s types.Null[string]
	if val := s.Value(); val != "" {
		t.Errorf("expected Value to return zero value '', got '%s'", val)
	}

	if val := s.Value("default"); val != "default" {
		t.Errorf("expected Value to return default value 'default', got '%s'", val)
	}

	if bts, err := json.Marshal(s); err != nil {
		t.Errorf("unexpected error during MarshalJSON: %v", err)
	} else if want := "null"; string(bts) != want {
		t.Errorf("expected marshaled JSON to be %s, got %s", want, string(bts))
	}

	s.SetValue("foo")
	if val := s.Value(); val != "foo" {
		t.Errorf("expected Value to return 'foo', got '%s'", val)
	}

	s = types.NullValue("bar")
	if val := s.Value(); val != "bar" {
		t.Errorf("expected Value to return 'bar', got '%s'", val)
	}

	if bts, err := json.Marshal(s); err != nil {
		t.Errorf("unexpected error during MarshalJSON: %v", err)
	} else if want := `"bar"`; string(bts) != want {
		t.Errorf("expected marshaled JSON to be %s, got %s", want, string(bts))
	}

	if err := json.Unmarshal([]byte(`null`), &s); err != nil {
		t.Errorf("unexpected error during UnmarshalJSON: %v", err)
	}

	if err := json.Unmarshal([]byte(`"baz"`), &s); err != nil {
		t.Errorf("unexpected error during UnmarshalJSON: %v", err)
	}

	if err := json.Unmarshal([]byte(`1.2345`), &s); err == nil {
		t.Error("expected error during UnmarshalJSON with invalid JSON, got nil")
	}
}
