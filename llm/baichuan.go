package llm

func baichuanModelType(layers uint32) string {
	switch layers {
	case 32:
		return "7B"
	case 40:
		return "13B"
	}

	return "unknown"
}
