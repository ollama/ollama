package model

type Capability string

const (
	CapabilityCompletion = Capability("completion")
	CapabilityTools      = Capability("tools")
	CapabilityInsert     = Capability("insert")
	CapabilityVision     = Capability("vision")
	CapabilityEmbedding  = Capability("embedding")
	CapabilityThinking   = Capability("thinking")
	CapabilityImage      = Capability("image")
	CapabilityAudio      = Capability("audio")
	CapabilityDecision   = Capability("decision")
)

// IsValid reports whether c is recognized, such as "decision"; empty or unknown labels are invalid.
func (c Capability) IsValid() bool {
	switch c {
	case CapabilityCompletion, CapabilityTools, CapabilityInsert, CapabilityVision,
		CapabilityEmbedding, CapabilityThinking, CapabilityImage, CapabilityAudio, CapabilityDecision:
		return true
	default:
		return false
	}
}

func (c Capability) String() string {
	return string(c)
}
