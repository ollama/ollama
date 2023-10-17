package llm

func refactModelType(layers uint32) string {
	switch layers {
	case 32:
		return "1B"
	}

	return "unknown"
}
