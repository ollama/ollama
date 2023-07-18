package format

import (
	"testing"
	"time"
)

func assertEqual(t *testing.T, a interface{}, b interface{}) {
	if a != b {
		t.Errorf("Assert failed, expected %v, got %v", b, a)
	}
}

func TestHumanDuration(t *testing.T) {
	day := 24 * time.Hour
	week := 7 * day
	month := 30 * day
	year := 365 * day

	assertEqual(t, "Less than a second", HumanDuration(450*time.Millisecond))
	assertEqual(t, "Less than a second", HumanDurationWithCase(450*time.Millisecond, true))
	assertEqual(t, "less than a second", HumanDurationWithCase(450*time.Millisecond, false))
	assertEqual(t, "1 second", HumanDuration(1*time.Second))
	assertEqual(t, "45 seconds", HumanDuration(45*time.Second))
	assertEqual(t, "46 seconds", HumanDuration(46*time.Second))
	assertEqual(t, "59 seconds", HumanDuration(59*time.Second))
	assertEqual(t, "About a minute", HumanDuration(60*time.Second))
	assertEqual(t, "About a minute", HumanDurationWithCase(1*time.Minute, true))
	assertEqual(t, "about a minute", HumanDurationWithCase(1*time.Minute, false))
	assertEqual(t, "3 minutes", HumanDuration(3*time.Minute))
	assertEqual(t, "35 minutes", HumanDuration(35*time.Minute))
	assertEqual(t, "35 minutes", HumanDuration(35*time.Minute+40*time.Second))
	assertEqual(t, "45 minutes", HumanDuration(45*time.Minute))
	assertEqual(t, "45 minutes", HumanDuration(45*time.Minute+40*time.Second))
	assertEqual(t, "46 minutes", HumanDuration(46*time.Minute))
	assertEqual(t, "59 minutes", HumanDuration(59*time.Minute))
	assertEqual(t, "About an hour", HumanDuration(1*time.Hour))
	assertEqual(t, "About an hour", HumanDurationWithCase(1*time.Hour+29*time.Minute, true))
	assertEqual(t, "about an hour", HumanDurationWithCase(1*time.Hour+29*time.Minute, false))
	assertEqual(t, "2 hours", HumanDuration(1*time.Hour+31*time.Minute))
	assertEqual(t, "2 hours", HumanDuration(1*time.Hour+59*time.Minute))
	assertEqual(t, "3 hours", HumanDuration(3*time.Hour))
	assertEqual(t, "3 hours", HumanDuration(3*time.Hour+29*time.Minute))
	assertEqual(t, "4 hours", HumanDuration(3*time.Hour+31*time.Minute))
	assertEqual(t, "4 hours", HumanDuration(3*time.Hour+59*time.Minute))
	assertEqual(t, "4 hours", HumanDuration(3*time.Hour+60*time.Minute))
	assertEqual(t, "24 hours", HumanDuration(24*time.Hour))
	assertEqual(t, "36 hours", HumanDuration(1*day+12*time.Hour))
	assertEqual(t, "2 days", HumanDuration(2*day))
	assertEqual(t, "7 days", HumanDuration(7*day))
	assertEqual(t, "13 days", HumanDuration(13*day+5*time.Hour))
	assertEqual(t, "2 weeks", HumanDuration(2*week))
	assertEqual(t, "2 weeks", HumanDuration(2*week+4*day))
	assertEqual(t, "3 weeks", HumanDuration(3*week))
	assertEqual(t, "4 weeks", HumanDuration(4*week))
	assertEqual(t, "4 weeks", HumanDuration(4*week+3*day))
	assertEqual(t, "4 weeks", HumanDuration(1*month))
	assertEqual(t, "6 weeks", HumanDuration(1*month+2*week))
	assertEqual(t, "2 months", HumanDuration(2*month))
	assertEqual(t, "2 months", HumanDuration(2*month+2*week))
	assertEqual(t, "3 months", HumanDuration(3*month))
	assertEqual(t, "3 months", HumanDuration(3*month+1*week))
	assertEqual(t, "5 months", HumanDuration(5*month+2*week))
	assertEqual(t, "13 months", HumanDuration(13*month))
	assertEqual(t, "23 months", HumanDuration(23*month))
	assertEqual(t, "24 months", HumanDuration(24*month))
	assertEqual(t, "2 years", HumanDuration(24*month+2*week))
	assertEqual(t, "3 years", HumanDuration(3*year+2*month))
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
}

func TestExactDuration(t *testing.T) {
	assertEqual(t, "1 millisecond", ExactDuration(1*time.Millisecond))
	assertEqual(t, "10 milliseconds", ExactDuration(10*time.Millisecond))
	assertEqual(t, "1 second", ExactDuration(1*time.Second))
	assertEqual(t, "10 seconds", ExactDuration(10*time.Second))
	assertEqual(t, "1 minute", ExactDuration(1*time.Minute))
	assertEqual(t, "10 minutes", ExactDuration(10*time.Minute))
	assertEqual(t, "1 hour", ExactDuration(1*time.Hour))
	assertEqual(t, "10 hours", ExactDuration(10*time.Hour))
	assertEqual(t, "1 hour 1 second", ExactDuration(1*time.Hour+1*time.Second))
	assertEqual(t, "1 hour 10 seconds", ExactDuration(1*time.Hour+10*time.Second))
	assertEqual(t, "1 hour 1 minute", ExactDuration(1*time.Hour+1*time.Minute))
	assertEqual(t, "1 hour 10 minutes", ExactDuration(1*time.Hour+10*time.Minute))
	assertEqual(t, "1 hour 1 minute 1 second", ExactDuration(1*time.Hour+1*time.Minute+1*time.Second))
	assertEqual(t, "10 hours 10 minutes 10 seconds", ExactDuration(10*time.Hour+10*time.Minute+10*time.Second))
}
