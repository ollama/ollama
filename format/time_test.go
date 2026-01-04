package format

import (
	"testing"
	"time"
)

func assertEqual(t *testing.T, a any, b any) {
	if a != b {
		t.Errorf("Assert failed, expected %v, got %v", b, a)
	}
}

func TestHumanTime(t *testing.T) {
	now := time.Now()

	t.Run("zero value", func(t *testing.T) {
		assertEqual(t, HumanTime(time.Time{}, "never"), "never")
	})

	t.Run("time in the future", func(t *testing.T) {
		v := now.Add(48 * time.Hour)
		assertEqual(t, HumanTime(v, ""), "2 days from now")
	})

	t.Run("time in the past", func(t *testing.T) {
		v := now.Add(-48 * time.Hour)
		assertEqual(t, HumanTime(v, ""), "2 days ago")
	})

	t.Run("soon", func(t *testing.T) {
		v := now.Add(800 * time.Millisecond)
		assertEqual(t, HumanTime(v, ""), "Less than a second from now")
	})

	t.Run("time way in the future", func(t *testing.T) {
		v := now.Add(24 * time.Hour * 365 * 200)
		assertEqual(t, HumanTime(v, ""), "Forever")
	})

	t.Run("time way in the future lowercase", func(t *testing.T) {
		v := now.Add(24 * time.Hour * 365 * 200)
		assertEqual(t, HumanTimeLower(v, ""), "forever")
	})
}
