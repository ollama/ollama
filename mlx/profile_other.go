//go:build !darwin && !linux

package mlx

func profileMarkersAvailable() bool { return false }

func profileRangePush(string) {}

func profileRangePop() {}
