package middleware

func optionalIntValue(v *int) int {
	if v == nil {
		return 0
	}
	return *v
}

func addOptionalInts(a, b *int) *int {
	if a == nil && b == nil {
		return nil
	}
	value := optionalIntValue(a) + optionalIntValue(b)
	return &value
}

func maxOptionalInts(a, b *int) *int {
	if a == nil && b == nil {
		return nil
	}
	value := max(optionalIntValue(a), optionalIntValue(b))
	return &value
}
