//go:build windows || darwin

// package logrotate provides utilities for rotating logs
// TODO (jmorgan): this most likely doesn't need it's own
// package and can be moved to app where log files are created
package logrotate

import (
	"log/slog"
	"os"
	"strconv"
	"strings"
)

const MaxLogFiles = 5

func Rotate(filename string) {
	if _, err := os.Stat(filename); os.IsNotExist(err) {
		return
	}

	index := strings.LastIndex(filename, ".")
	pre := filename[:index]
	post := "." + filename[index+1:]
	for i := MaxLogFiles; i > 0; i-- {
		older := pre + "-" + strconv.Itoa(i) + post
		newer := pre + "-" + strconv.Itoa(i-1) + post
		if i == 1 {
			newer = pre + post
		}
		if _, err := os.Stat(newer); err == nil {
			if _, err := os.Stat(older); err == nil {
				err := os.Remove(older)
				if err != nil {
					slog.Warn("Failed to remove older log", "older", older, "error", err)
					continue
				}
			}
			err := os.Rename(newer, older)
			if err != nil {
				slog.Warn("Failed to rotate log", "older", older, "newer", newer, "error", err)
			}
		}
	}
}
