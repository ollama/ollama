package model

type Capability string

const (
	CapabilityCompletion = Capability("completion")
	CapabilityTools      = Capability("tools")
	CapabilityInsert     = Capability("insert")
	CapabilityVision     = Capability("vision")
	CapabilityEmbedding  = Capability("embedding")
	CapabilityThinking   = Capability("thinking")
	CapabilityReranking  = Capability("reranking")
)

func (c Capability) String() string {
	return string(c)
}
