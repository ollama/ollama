//go:build !windows

package compatmigrate

import "golang.org/x/sys/unix"

func availableSpace(path string) (uint64, error) {
	var st unix.Statfs_t
	if err := unix.Statfs(path, &st); err != nil {
		return 0, err
	}

	return uint64(st.Bavail) * uint64(st.Bsize), nil
}
