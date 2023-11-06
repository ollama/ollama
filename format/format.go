package format

import "fmt"

const (
	Thousand = 1000
	Million  = Thousand * 1000
	Billion  = Million * 1000
)

func Human(b uint64) string {
	switch {
	case b > Billion:
		return fmt.Sprintf("%dB", b/Billion)
	case b > Million:
		return fmt.Sprintf("%dM", b/Million)
	case b > Thousand:
		return fmt.Sprintf("%dK", b/Thousand)
	default:
		return fmt.Sprintf("%d", b)
	}
}
