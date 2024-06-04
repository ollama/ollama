package format

import (
	"fmt"
)

const (
	Thousand = 1000
	Million  = Thousand * 1000
	Billion  = Million * 1000
	Trillion = Billion * 1000
)

func HumanNumber(b uint64) string {
	switch {
	case b >= Trillion:
		number := float64(b) / Trillion
		return fmt.Sprintf("%sT", DecimalPlace(number))
	case b >= Billion:
		number := float64(b) / Billion
		return fmt.Sprintf("%sB", DecimalPlace(number))
	case b >= Million:
		number := float64(b) / Million
		return fmt.Sprintf("%sM", DecimalPlace(number))
	case b >= Thousand:
		number := float64(b) / Thousand
		return fmt.Sprintf("%sK", DecimalPlace(number))
	default:
		return fmt.Sprintf("%d", b)
	}
}

func DecimalPlace(number float64) string {
	switch {
	case number >= 100:
		return fmt.Sprintf("%.0f", number)
	case number >= 10:
		return fmt.Sprintf("%.1f", number)
	default:
		return fmt.Sprintf("%.2f", number)
	}
}
