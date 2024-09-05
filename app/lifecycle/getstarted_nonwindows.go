//go:build !windows

package lifecycle

import "errors"

func GetStarted() error {
	return errors.New("not implemented")
}
