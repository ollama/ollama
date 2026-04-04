package server

import (
	"github.com/ollama/ollama/cmd"
)

// SleepPrevention manages system sleep state during inference
// It prevents the system from sleeping while models are being processed
type SleepPrevention struct {
	active bool
	token  uint32
}

// Start prevents system sleep during inference
func (sp *SleepPrevention) Start() {
	if sp.active {
		return // Already active
	}
	
	if !cmd.IsSystemSleepPreventionAvailable() {
		// Sleep prevention not available on this platform
		return
	}
	
	sp.token = cmd.PreventSystemSleep()
	sp.active = true
}

// Stop allows system sleep after inference completes
func (sp *SleepPrevention) Stop() {
	if !sp.active {
		return
	}
	
	if sp.token > 0 {
		cmd.AllowSystemSleep(sp.token)
	}
	sp.active = false
}

// NewSleepPrevention creates a new sleep prevention manager
func NewSleepPrevention() *SleepPrevention {
	return &SleepPrevention{
		active: false,
		token:  0,
	}
}
