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

	// Optional variant to select (e.g. versions, cpu feature flags)
	Variant string `json:"variant,omitempty"`

	// MinimumMemory represents the minimum memory required to use the GPU
	MinimumMemory uint64 `json:"-"`

	// TODO add other useful attributes about the card here for discovery information
}

type Version struct {
	Major uint
	Minor uint
	Patch uint
}
