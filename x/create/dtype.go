package create

import (
	"strings"
)

func sourceQuantType(mode string, bits int) string {
	switch strings.ToLower(mode) {
	case "affine":
		switch bits {
		case 4:
			return "int4"
		case 8:
			return "int8"
		}
	case "nvfp4":
		return "nvfp4"
	case "mxfp8":
		return "mxfp8"
	case "mxfp4":
		return "mxfp4"
	}
	return ""
}
