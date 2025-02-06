package format

import (
	"fmt"
	"math"
)

const (
	_   = iota
	KiB = 1 << (10 * iota)
	MiB
	GiB
	TiB
)

func HumanBytes(b int64) string {
	var value float64
	var unit string

	switch {
	case b >= TiB:
		value = float64(b) / TiB
		unit = "TB"
	case b >= GiB:
		value = float64(b) / GiB
		unit = "GB"
	case b >= MiB:
		value = float64(b) / MiB
		unit = "MB"
	case b >= KiB:
		value = float64(b) / KiB
		unit = "KB"
	default:
		return fmt.Sprintf("%d B", b)
	}

	switch {
	case value >= 10:
		return fmt.Sprintf("%d %s", int(value), unit)
	case value != math.Trunc(value):
		return fmt.Sprintf("%.1f %s", value, unit)
	default:
		return fmt.Sprintf("%d %s", int(value), unit)
	}
}

func HumanBytes2(b uint64) string {
	switch {
	case b >= GiB:
		return fmt.Sprintf("%.1f GiB", float64(b)/GiB)
	case b >= MiB:
		return fmt.Sprintf("%.1f MiB", float64(b)/MiB)
	case b >= KiB:
		return fmt.Sprintf("%.1f KiB", float64(b)/KiB)
	default:
		return fmt.Sprintf("%d B", b)
	}
}
