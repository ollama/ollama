package llm

func starCoderModelType(numLayer uint32) string {
	switch numLayer {
	case 24:
		return "1B"
	case 36:
		return "3B"
	case 42:
		return "7B"
	case 40:
		return "15B"
	default:
		return "unknown"
	}
}
