package llm

func bloomModelType(layers, embds uint32) string {
	switch layers {
	case 24:
		return "1B"
	case 30:
		switch embds {
		case 2560:
			return "3B"
		case 4096:
			return "7B"
		}
	}

	return "unknown"
}
