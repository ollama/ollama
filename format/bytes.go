package format

import "fmt"

const (
	Byte     = 1
	KiloByte = Byte * 1000
	MegaByte = KiloByte * 1000
	GigaByte = MegaByte * 1000
)

func HumanBytes(b int64) string {
	switch {
	case b > GigaByte:
		return fmt.Sprintf("%.1f GB", float64(b)/GigaByte)
	case b > MegaByte:
		return fmt.Sprintf("%.1f MB", float64(b)/MegaByte)
	case b > KiloByte:
		return fmt.Sprintf("%.1f KB", float64(b)/KiloByte)
	default:
		return fmt.Sprintf("%d B", b)
	}
}
