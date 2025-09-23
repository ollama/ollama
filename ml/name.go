package ml

// lets callers do ml.SetName(t, "foo") without importing ggml
func SetName(t Tensor, name string) {
	if n, ok := t.(interface{ SetName(string) }); ok {
		n.SetName(name)
	}
}
