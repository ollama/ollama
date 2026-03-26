package autotune

import (
	"encoding/json"
	"log/slog"
	"net/http"
	"sync"

	"github.com/gin-gonic/gin"
)

// --- Shared state ---

var (
	currentPlanMu sync.RWMutex
	currentPlan   *TunePlan
)

// SetCurrentPlan stores the last applied TunePlan for status queries.
func SetCurrentPlan(plan *TunePlan) {
	currentPlanMu.Lock()
	defer currentPlanMu.Unlock()
	currentPlan = plan
}

// GetCurrentPlan returns the last applied TunePlan, or nil.
func GetCurrentPlan() *TunePlan {
	currentPlanMu.RLock()
	defer currentPlanMu.RUnlock()
	return currentPlan
}

// --- API types ---

// StatusResponse is returned by GET /api/autotune.
type StatusResponse struct {
	Enabled  bool                 `json:"enabled"`
	Profile  string               `json:"profile,omitempty"`
	Hardware *HardwareSummary     `json:"hardware,omitempty"`
	Applied  []RecommendationJSON `json:"applied,omitempty"`
}

// HardwareSummary is a JSON-safe subset of HardwareProfile.
type HardwareSummary struct {
	CPUModel       string `json:"cpu_model"`
	CPUThreads     int    `json:"cpu_threads"`
	GPUCount       int    `json:"gpu_count"`
	TotalVRAMBytes uint64 `json:"total_vram_bytes"`
	TotalRAMBytes  uint64 `json:"total_ram_bytes"`
	OS             string `json:"os"`
	Arch           string `json:"arch"`
	Summary        string `json:"summary"`
}

// RecommendationJSON is a JSON-safe Recommendation.
type RecommendationJSON struct {
	Key    string `json:"key"`
	Value  string `json:"value"`
	Reason string `json:"reason"`
}

// TuneRequest is the body for POST /api/autotune.
type TuneRequest struct {
	Profile string `json:"profile"`
}

// --- HTTP handlers ---

// HandleGetAutotune is the GET /api/autotune handler.
func HandleGetAutotune(c *gin.Context) {
	plan := GetCurrentPlan()
	if plan == nil {
		c.JSON(http.StatusOK, StatusResponse{Enabled: false})
		return
	}

	resp := StatusResponse{
		Enabled:  true,
		Profile:  string(plan.Profile),
		Hardware: hwSummary(&plan.Hardware),
	}
	for _, r := range plan.Recommendations {
		resp.Applied = append(resp.Applied, RecommendationJSON{
			Key: r.Key, Value: r.Value, Reason: r.Reason,
		})
	}
	c.JSON(http.StatusOK, resp)
}

// HandlePostAutotune is the POST /api/autotune handler (re-tune at runtime).
func HandlePostAutotune(c *gin.Context) {
	var req TuneRequest
	if err := json.NewDecoder(c.Request.Body).Decode(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid request body"})
		return
	}

	profile := ParseProfile(req.Profile)
	hw := DetectHardware(nil) // refresh hardware snapshot
	plan := Tune(hw, profile)
	applied := Apply(plan)
	SetCurrentPlan(plan)

	slog.Info("autotune: re-tuned via API",
		"profile", string(profile),
		"applied", len(applied))

	resp := StatusResponse{
		Enabled:  true,
		Profile:  string(plan.Profile),
		Hardware: hwSummary(&plan.Hardware),
	}
	for _, r := range plan.Recommendations {
		resp.Applied = append(resp.Applied, RecommendationJSON{
			Key: r.Key, Value: r.Value, Reason: r.Reason,
		})
	}
	c.JSON(http.StatusOK, resp)
}

// HandleGetProfiles is the GET /api/autotune/profiles handler.
func HandleGetProfiles(c *gin.Context) {
	type profileInfo struct {
		Name        string `json:"name"`
		Description string `json:"description"`
	}
	profiles := []profileInfo{
		{"speed", "Maximize token generation speed (single-user, high VRAM usage)"},
		{"balanced", "Good tradeoff between speed and resource usage (default)"},
		{"memory", "Minimize memory usage for constrained hardware"},
		{"multiuser", "Optimize for concurrent users with shared cache"},
		{"max", "Squeeze maximum performance at the cost of resources"},
	}
	c.JSON(http.StatusOK, gin.H{"profiles": profiles})
}

func hwSummary(hw *HardwareProfile) *HardwareSummary {
	cpuModel := "unknown"
	if len(hw.CPUs) > 0 {
		cpuModel = hw.CPUs[0].ModelName
	}
	return &HardwareSummary{
		CPUModel:       cpuModel,
		CPUThreads:     hw.System.ThreadCount,
		GPUCount:       hw.DiscreteGPUCount(),
		TotalVRAMBytes: hw.TotalVRAM(),
		TotalRAMBytes:  hw.System.TotalMemory,
		OS:             hw.Platform.OS,
		Arch:           hw.Platform.Arch,
		Summary:        hw.Summary(),
	}
}
