package llm

func falconModelType(layers uint32) string {
	switch layers {
	case 32:
		return "7B"
	case 60:
		return "40B"
	case 80:
		return "180B"
	}

	return "unknown"
}
