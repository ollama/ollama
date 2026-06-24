package tui

import "regexp"

func stripANSI(s string) string {
	re := regexp.MustCompile(`\x1b\[[0-9;:]*[A-Za-z]`)
	return re.ReplaceAllString(s, "")
}
