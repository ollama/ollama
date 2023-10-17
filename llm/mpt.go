package llm

func mptModelType(numLayer uint32) string {
	switch numLayer {
	case 32:
		return "7B"
	case 48:
		return "30B"
	}

	return "unknown"
}
