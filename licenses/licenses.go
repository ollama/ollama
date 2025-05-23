// Package licenses embeds third-party licenses for release binaries.
// The licenses can be collected for embedding using
//
//	go run github.com/google/go-licenses@v1.6.0 save . --save_path=licenses/content
package licenses

import (
	"embed"
	"io/fs"
)

//go:embed content
var licenses embed.FS

// LicenseText is the concatenation of all the third-party licenses used by the app.
var LicenseText string

func init() {
	fs.WalkDir(licenses, ".", func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		content, err := fs.ReadFile(licenses, path)
		if err != nil {
			return err
		}
		LicenseText += string(content) + "\n"
		return nil
	})
}
