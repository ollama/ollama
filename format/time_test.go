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

func TestHumanDurationYearBoundary(t *testing.T) {
	// The switch reads the rounded hour count, so the years label has to divide
	// the same one. It used to divide the truncated count instead, and in the
	// last half hour before a year boundary the two disagree: the label went
	// backwards, "24 months" for a shorter age and "1 years" for a longer one,
	// before jumping to "2 years".
	for _, tc := range []struct {
		d    time.Duration
		want string
	}{
		{17519*time.Hour + 20*time.Minute, "24 months"},
		{17519*time.Hour + 30*time.Minute, "2 years"},
		{17519*time.Hour + 59*time.Minute, "2 years"},
		{17520 * time.Hour, "2 years"},
		{26279*time.Hour + 40*time.Minute, "3 years"},
		{26280 * time.Hour, "3 years"},
	} {
		assertEqual(t, humanDuration(tc.d), tc.want)
	}
}
