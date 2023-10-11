package format

import "fmt"

func HumanBytes(b int64) string {
	switch {
	case b > 1000*1000*1000:
		return fmt.Sprintf("%d GB", b/1000/1000/1000)
	case b > 1000*1000:
		return fmt.Sprintf("%d MB", b/1000/1000)
	case b > 1000:
		return fmt.Sprintf("%d KB", b/1000)
	default:
		return fmt.Sprintf("%d B", b)
	}
}
