package llm

func persimmonModelType(layers uint32) string {
	switch layers {
	case 36:
		return "8B"
	}

	return "unknown"
}
