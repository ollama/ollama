//go:build !darwin && !linux

package mlx

func profileRangePush(string) {}

func profileRangePop() {}
