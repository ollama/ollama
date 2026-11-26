//go:build windows || darwin

package assets

import (
	"embed"
	"io/fs"
)

//go:embed *.ico
var icons embed.FS

//go:embed hello.png
var images embed.FS

func ListIcons() ([]string, error) {
	return fs.Glob(icons, "*")
}

func GetIcon(filename string) ([]byte, error) {
	return icons.ReadFile(filename)
}

// GetImage returns an image the app's windows show, such as onboarding's.
func GetImage(filename string) ([]byte, error) {
	return images.ReadFile(filename)
}
