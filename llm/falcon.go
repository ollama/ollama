package llm

const ModelFamilyFalcon = "falcon"

const (
	falconModelType7B   = 32
	falconModelType40B  = 60
	falconModelType180B = 80
)

func falconModelType(numLayer uint32) string {
	switch numLayer {
	case 32:
		return "7B"
	case 60:
		return "40B"
	case 80:
		return "180B"
	default:
		return "Unknown"
	}
}
