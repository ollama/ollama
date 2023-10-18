package llm

const (
	starCoderModelType1B  = 24
	starCoderModelType3B  = 36
	starCoderModelType7B  = 42
	starCoderModelType15B = 40
)

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
