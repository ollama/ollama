package create

import "strings"

// sourceFloatFileType names the float dtype holding most of an unquantized
// source's weight elements, in the spelling Kev and GGUF imports record, so
// the package states its precision rather than relying on config.json's
// optional (and sometimes wrong) dtype field.
func sourceFloatFileType(inv Inventory) string {
	elements := map[string]int64{}
	for _, t := range inv.Tensors {
		n := int64(1)
		for _, d := range t.Shape {
			n *= int64(d)
		}
		elements[strings.ToUpper(t.Dtype)] += n
	}
	var best string
	for dtype, n := range elements {
		if n > elements[best] || (n == elements[best] && dtype < best) {
			best = dtype
		}
	}
	switch best {
	case "BF16", "F16", "F32":
		return best
	}
	return ""
}
