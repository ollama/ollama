//go:build !windows

package lifecycle

import "fmt"

func GetStarted() error {
	return fmt.Errorf("GetStarted not implemented")
}
