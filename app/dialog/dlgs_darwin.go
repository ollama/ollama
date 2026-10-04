package dialog

import (
	"github.com/ollama/ollama/app/dialog/cocoa"
)

func (b *FileBuilder) loadMultiple() ([]string, error) {
	return b.runMultiple()
}

func (b *FileBuilder) runMultiple() ([]string, error) {
	star := false
	var exts []string
	for _, filt := range b.Filters {
		for _, ext := range filt.Extensions {
			if ext == "*" {
				star = true
			} else {
				exts = append(exts, ext)
			}
		}
	}

	files, err := cocoa.MultiFileDlg(b.Dlg.Title, exts, star, "", false)
	if len(files) == 0 && err == nil {
		return nil, ErrCancelled
	}
	return files, err
}

func (b *DirectoryBuilder) browse() (string, error) {
	f, err := cocoa.DirDlg(b.Dlg.Title, "", b.ShowHiddenFiles)
	if f == "" && err == nil {
		return "", ErrCancelled
	}
	return f, err
}
