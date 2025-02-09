package assets

import (
    "embed"
    "io/fs"
)

//go:embed *.ico
var embeddedIcons embed.FS

var Icons fs.FS = embeddedIcons

func ListIcons() ([]string, error) {
    return fs.Glob(Icons, "*")
}

func GetIcon(filename string) ([]byte, error) {
    return fs.ReadFile(Icons, filename)
}
