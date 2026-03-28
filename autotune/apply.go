package autotune

import (
	"fmt"
	"log/slog"
	"os"
	"strings"
	"time"
)

// Apply sets the environment variables from the TunePlan into the current
// process. It only sets variables that are not already explicitly set by the
// user (to avoid overriding intentional configuration).
// Returns the list of variables that were actually set.
func Apply(plan *TunePlan) []Recommendation {
	var applied []Recommendation
	for _, rec := range plan.Recommendations {
		if existing := os.Getenv(rec.Key); existing != "" {
			slog.Debug("autotune: skipping (user-set)",
				"key", rec.Key, "existing", existing, "recommended", rec.Value)
			continue
		}
		os.Setenv(rec.Key, rec.Value)
		applied = append(applied, rec)
		slog.Info("autotune: applied",
			"key", rec.Key, "value", rec.Value, "reason", rec.Reason)
	}
	return applied
}

// ApplyDefaults is the main entry point called during server startup.
// It detects hardware, generates a TunePlan for the given profile, and
// applies settings that the user hasn't explicitly set.
func ApplyDefaults(gpuDevices interface{}, profileStr string) *TunePlan {
	// Accept nil or []ml.DeviceInfo — the caller should pass the actual
	// gpu list after discover.GPUDevices(), but we handle nil gracefully.
	var hw HardwareProfile
	switch g := gpuDevices.(type) {
	case nil:
		hw = DetectHardware(nil)
	default:
		// We can't import ml directly here due to potential cycles in
		// some build configurations. The caller passes the right type.
		_ = g
		hw = DetectHardware(nil)
	}

	profile := ParseProfile(profileStr)
	plan := Tune(hw, profile)
	Apply(plan)
	return plan
}

// FormatPlan returns a human-readable summary of the tuning plan,
// suitable for logging at startup.
func FormatPlan(plan *TunePlan) string {
	var b strings.Builder
	b.WriteString(fmt.Sprintf("autotune profile: %s\n", plan.Profile))
	b.WriteString(fmt.Sprintf("hardware: %s\n", plan.Hardware.Summary()))
	if len(plan.Recommendations) > 0 {
		b.WriteString("recommendations:\n")
		for _, r := range plan.Recommendations {
			b.WriteString(fmt.Sprintf("  %s=%s (%s)\n", r.Key, r.Value, r.Reason))
		}
	} else {
		b.WriteString("no recommendations (all settings look good)\n")
	}
	return b.String()
}

// --- helpers for reading the current effective config from env ---

// EffectiveKeepAlive returns the current OLLAMA_KEEP_ALIVE as duration,
// or the plan's recommendation.
func EffectiveKeepAlive(plan *TunePlan) time.Duration {
	if s := os.Getenv("OLLAMA_KEEP_ALIVE"); s != "" {
		// The existing envconfig module will parse this; we just
		// indicate it's been set.
		return 0
	}
	return plan.KeepAlive
}
