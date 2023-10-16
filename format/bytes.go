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
		return fmt.Sprintf("%d GB", b/GigaByte)
	case b > MegaByte:
		return fmt.Sprintf("%d MB", b/MegaByte)
	case b > KiloByte:
		return fmt.Sprintf("%d KB", b/KiloByte)
	default:
		return fmt.Sprintf("%d B", b)
	}
}
