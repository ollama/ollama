package cmd

import "errors"

func localCopy(src, target string) error {
	return errors.New("no local copy implementation for linux")
}
