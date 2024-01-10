package gpu

type memInfo struct {
	TotalMemory uint64 `json:"total_memory,omitempty"`
	FreeMemory  uint64 `json:"free_memory,omitempty"`
	DeviceCount uint32 `json:"device_count,omitempty"`
}

// Beginning of an `ollama info` command
type GpuInfo struct {
	memInfo
	Library string `json:"library,omitempty"`

	// TODO add other useful attributes about the card here for discovery information
}
