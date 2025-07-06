// Process a prompt along with optional suffix for FIM and format type.
func (m *Model) Generate(ctx context.Context, prompt, suffix, format string, options map[string]interface{}) (string, error) {
	if prompt == "" {
		return "", nil
	}

	// Handle empty suffix for models that require special handling (like qwen2.5-coder)
	if strings.Contains(m.ModelName(), "qwen2.5-coder") && suffix == "" {
		suffix = " " // Convert empty suffix to space for FIM functionality
	}

	// Apply bias adapters
	if err := m.setBiasAdapters(options); err != nil {
		return "", err
	}
