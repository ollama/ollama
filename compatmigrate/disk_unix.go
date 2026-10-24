//go:build !windows

package compatmigrate

import "golang.org/x/sys/unix"

func availableSpace(path string) (uint64, error) {
	var st unix.Statfs_t
	if err := unix.Statfs(path, &st); err != nil {
		return 0, err
	}

	// Bavail is uint64 on Linux and int64 on FreeBSD and DragonFly.
	return saturatingCount(st.Bavail) * uint64(st.Bsize), nil
}

// saturatingCount converts a Statfs block count. A negative count means
// no space is available to an unprivileged caller.
func saturatingCount[N ~int64 | ~uint64](n N) uint64 {
	var zero N
	if n < zero {
		return 0
	}
	return uint64(n)
}
