package fitcheck

// FitResponse is returned by GET /api/fit and contains hardware capabilities
// alongside a ranked list of model compatibility candidates.
type FitResponse struct {
	// System describes the hardware of the machine running Ollama.
	System HardwareProfile `json:"system"`

	// Models is the ranked list of model candidates for this hardware.
	Models []ModelCandidate `json:"models"`
}
