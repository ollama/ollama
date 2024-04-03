package build

func convertSafeTensorToGGUF(path string) (ggufPath string, err error) {
	// TODO: decine on hueristic for converting safetensor to gguf and
	// the errors that can be returned. For now, we just say
	// "unsupported", however it may be intended to be a valid safe
	// tensor but we hit an error in the conversion.
	//
	// I (bmizernay) think this will naturally evolve as we implement
	// the conversion.
	return "", ErrUnsupportedModelFormat
}
