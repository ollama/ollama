// Template is the prompt template for Qwen2.5
func (Qwen2) Template() string {
	return DEFAULT_TEMPLATE
}

// HandleEmptySuffix ensures that empty string suffixes work correctly for FIM
// by converting them to a space character, which is known to work.
func (Qwen2) HandleEmptySuffix(suffix string) string {
	if suffix == "" {
		return " " // Replace empty suffix with a space for FIM to work properly
	}
	return suffix
}
