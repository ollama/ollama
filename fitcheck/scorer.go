package fitcheck

import (
	"fmt"
	"sort"
)

const gb = uint64(1024 * 1024 * 1024)

// CompatibilityTier ranks how well a model will run on the given hardware.
type CompatibilityTier int

const (
	// TierIdeal means the model will run entirely on GPU with headroom to spare.
	TierIdeal CompatibilityTier = iota
	// TierGood means the model fits on GPU with minor CPU offloading.
	TierGood
	// TierMarginal means the model requires significant CPU offloading and will run slowly.
	TierMarginal
	// TierPossible means the model will run CPU-only, very slowly.
	TierPossible
	// TierTooLarge means the model cannot run on this hardware.
	TierTooLarge
)

// String returns the human-readable name for the tier.
func (t CompatibilityTier) String() string {
	switch t {
	case TierIdeal:
		return "ideal"
	case TierGood:
		return "good"
	case TierMarginal:
		return "marginal"
	case TierPossible:
		return "possible"
	case TierTooLarge:
		return "too_large"
	default:
		return "unknown"
	}
}

// ModelCandidate is a scored and annotated model entry.
type ModelCandidate struct {
	// Req is the original model requirement entry.
	Req ModelRequirement `json:"req"`

	// Tier is the compatibility tier for this hardware.
	Tier CompatibilityTier `json:"tier"`

	// Score is a 0.0–1.0 composite score (higher is better).
	Score float64 `json:"score"`

	// RunMode describes how the model will execute: "GPU", "GPU+CPU", or "CPU".
	RunMode string `json:"run_mode"`

	// EstTPS is the estimated tokens-per-second on this hardware.
	EstTPS int `json:"est_tps"`

	// Notes are human-readable warnings or suggestions about running this model.
	Notes []string `json:"notes,omitempty"`

	// Installed is true if the model blob is already present in ModelsDir.
	Installed bool `json:"installed"`
}

// Score evaluates all reqs against hw and returns sorted candidates (best first).
// Primary sort key: Tier ascending (lower tier = better). Secondary: Score descending.
func Score(hw HardwareProfile, reqs []ModelRequirement) []ModelCandidate {
	candidates := make([]ModelCandidate, 0, len(reqs))

	for _, req := range reqs {
		c := scoreOne(hw, req)
		candidates = append(candidates, c)
	}

	sort.SliceStable(candidates, func(i, j int) bool {
		if candidates[i].Tier != candidates[j].Tier {
			return candidates[i].Tier < candidates[j].Tier
		}
		return candidates[i].Score > candidates[j].Score
	})

	return candidates
}

func scoreOne(hw HardwareProfile, req ModelRequirement) ModelCandidate {
	var notes []string

	hasGPU := hw.BestGPU != nil && hw.BestGPU.FreeMemory > 0
	var vramFreeB uint64
	if hasGPU {
		vramFreeB = hw.BestGPU.FreeMemory
	}
	vramReqB := req.VRAMMinMB * 1024 * 1024
	ramFreeB := hw.RAMAvailableBytes
	ramReqB := req.RAMMinMB * 1024 * 1024
	diskFreeB := hw.DiskModelAvailBytes
	diskReqB := req.DiskSizeMB * 1024 * 1024

	// --- vramScore & runMode ---
	var vramScore float64
	var runMode string
	if !hasGPU {
		vramScore = 0.0
		runMode = "CPU"
	} else if vramFreeB >= vramReqB {
		vramScore = 1.0
		runMode = "GPU"
	} else if ramFreeB >= ramReqB {
		ratio := float64(vramFreeB) / float64(vramReqB)
		vramScore = 0.25 + (0.40 * ratio)
		runMode = "GPU+CPU"
	} else {
		vramScore = 0.0
		runMode = "CPU"
	}

	// --- ramScore ---
	var ramScore float64
	if ramFreeB >= ramReqB {
		ramScore = 1.0
	} else if hw.RAMTotalBytes >= uint64(float64(ramReqB)*0.85) {
		ramScore = 0.5
		notes = append(notes, "RAM is tight; close other apps")
	} else {
		ramScore = 0.0
		notes = append(notes, fmt.Sprintf("Needs %s RAM, only %s available",
			humanize(ramReqB), humanize(ramFreeB)))
	}

	// --- diskScore ---
	var diskScore float64
	if diskFreeB >= diskReqB {
		diskScore = 1.0
	} else {
		diskScore = 0.0
		notes = append(notes, fmt.Sprintf("Needs %s disk space, only %s free",
			humanize(diskReqB), humanize(diskFreeB)))
	}

	// --- speedScore & estTPS ---
	var speedScore float64
	var estTPS int
	if !hasGPU {
		// Estimate CPU TPS based on ~45 GB/s memory bandwidth
		if req.DiskSizeMB > 0 {
			estTPS = int(45000.0 / float64(req.DiskSizeMB))
		} else {
			estTPS = 3
		}
		if estTPS > 100 {
			estTPS = 100
		}
		if estTPS < 1 {
			estTPS = 1
		}
		
		switch {
		case estTPS >= 30:
			speedScore = 0.85
		case estTPS >= 10:
			speedScore = 0.60
		case estTPS >= 4:
			speedScore = 0.35
		default:
			speedScore = 0.15
		}
	} else {
		switch hw.BestGPU.Library {
		case "Metal":
			total := hw.BestGPU.TotalMemory
			switch {
			case total >= 36*gb:
				speedScore = 1.0
				estTPS = 120
			case total >= 18*gb:
				speedScore = 0.85
				estTPS = 90
			default:
				speedScore = 0.65
				estTPS = 55
			}
		case "CUDA":
			maj := hw.BestGPU.ComputeMajor
			switch {
			case maj >= 9:
				speedScore = 1.0
				estTPS = 150
			case maj >= 8:
				speedScore = 0.85
				estTPS = 100
			case maj >= 7:
				speedScore = 0.65
				estTPS = 60
			default:
				speedScore = 0.40
				estTPS = 25
			}
		case "ROCm":
			speedScore = 0.70
			estTPS = 70
		default:
			speedScore = 0.20
			estTPS = 8
		}
	}

	finalScore := (vramScore * 0.40) + (ramScore * 0.25) + (diskScore * 0.15) + (speedScore * 0.20)

	// For CPU-only runs, if it's super fast, give a little bump so tiny models can be 'Good'
	if !hasGPU && estTPS >= 20 {
		finalScore += 0.20
	}

	// --- tier ---
	var tier CompatibilityTier
	switch {
	case ramScore == 0.0 || diskScore == 0.0:
		tier = TierTooLarge
	case finalScore >= 0.82:
		tier = TierIdeal
	case finalScore >= 0.62:
		tier = TierGood
	case finalScore >= 0.38:
		tier = TierMarginal
	case finalScore >= 0.15:
		tier = TierPossible
	default:
		tier = TierTooLarge
	}

	// Scale estTPS by model size relative to GPU memory
	if hasGPU && vramReqB > 0 {
		usageRatio := float64(vramReqB) / float64(hw.BestGPU.TotalMemory)
		scale := 1.0 - usageRatio*0.5
		if scale < 0.3 {
			scale = 0.3
		}
		estTPS = int(float64(estTPS) * scale)
	}

	return ModelCandidate{
		Req:     req,
		Tier:    tier,
		Score:   finalScore,
		RunMode: runMode,
		EstTPS:  estTPS,
		Notes:   notes,
	}
}

// humanize formats byte counts in human-readable units.
func humanize(bytes uint64) string {
	const (
		kb = uint64(1024)
		mb = uint64(1024 * 1024)
		_gb = uint64(1024 * 1024 * 1024)
	)
	switch {
	case bytes < kb:
		return fmt.Sprintf("%d B", bytes)
	case bytes < mb:
		return fmt.Sprintf("%d KB", bytes/kb)
	case bytes < _gb:
		return fmt.Sprintf("%.1f MB", float64(bytes)/float64(mb))
	default:
		return fmt.Sprintf("%.1f GB", float64(bytes)/float64(_gb))
	}
}
