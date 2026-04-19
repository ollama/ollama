package format

import (
	"fmt"
	"math"
)

const (
	Byte = 1

	KiloByte = Byte * 1000
	MegaByte = KiloByte * 1000
	GigaByte = MegaByte * 1000
	TeraByte = GigaByte * 1000

	KibiByte = Byte * 1024
	MebiByte = KibiByte * 1024
	GibiByte = MebiByte * 1024
)

func HumanBytes(b int64) string {
	var value float64
	var unit string

	// Use binary prefixes (1024) for file sizes as is standard in computing
	switch {
	case b >= GibiByte*1024:
		value = float64(b) / (GibiByte * 1024)
		unit = "TB"
	case b >= GibiByte:
		value = float64(b) / GibiByte
		unit = "GB"
	case b >= MebiByte:
		value = float64(b) / MebiByte
		unit = "MB"
	case b >= KibiByte:
		value = float64(b) / KibiByte
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
	case b >= GibiByte:
		return fmt.Sprintf("%.1f GiB", float64(b)/GibiByte)
	case b >= MebiByte:
		return fmt.Sprintf("%.1f MiB", float64(b)/MebiByte)
	case b >= KibiByte:
		return fmt.Sprintf("%.1f KiB", float64(b)/KibiByte)
	default:
		return fmt.Sprintf("%d B", b)
	}
}
