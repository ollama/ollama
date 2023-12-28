package format

import (
	"fmt"
	"math"
)

const (
	Thousand = 1000
	Million  = Thousand * 1000
	Billion  = Million * 1000
)

func HumanNumber(b uint64) string {
	switch {
	case b > Billion:
		return fmt.Sprintf("%.0fB", math.Round(float64(b)/Billion))
	case b > Million:
		return fmt.Sprintf("%.0fM", math.Round(float64(b)/Million))
	case b > Thousand:
		return fmt.Sprintf("%.0fK", math.Round(float64(b)/Thousand))
	default:
		return fmt.Sprintf("%d", b)
	}
}
