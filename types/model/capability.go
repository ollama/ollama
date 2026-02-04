package model

type Capability string

const (
	CapabilityCompletion      = Capability("completion")
	CapabilityTools           = Capability("tools")
	CapabilityInsert          = Capability("insert")
	CapabilityVision          = Capability("vision")
	CapabilityEmbedding       = Capability("embedding")
	CapabilityThinking        = Capability("thinking")
	CapabilityImage = Capability("image")
)

func (c Capability) String() string {
	return string(c)
}
