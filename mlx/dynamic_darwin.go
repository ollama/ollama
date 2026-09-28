package mlx

import (
	"strconv"
	"strings"
	"syscall"
)

func macOSMajorVersion() int {
	ver, err := syscall.Sysctl("kern.osproductversion")
	if err != nil {
		return 0
	}
	parts := strings.SplitN(ver, ".", 2)
	major, _ := strconv.Atoi(parts[0])
	return major
}
