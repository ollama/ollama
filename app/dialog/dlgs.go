//go:build windows || darwin

// Package dialog provides a simple cross-platform common dialog API.
// Eg. to prompt the user for a directory:
//
//	dir, err := dialog.Directory().Title("Select a folder").Browse()
//
// The general usage pattern is to call one of the toplevel *Dlg functions
// which return a *Builder structure. From here you can optionally call
// configuration functions (eg. Title) to customise the dialog, before
// using a launcher function to run the dialog.
package dialog

import (
	"errors"
)

// ErrCancelled is an error returned when a user cancels/closes a dialog.
var ErrCancelled = errors.New("Cancelled")

// Dlg is the common type for dialogs.
type Dlg struct {
	Title string
}

// FileFilter represents a category of files (eg. audio files, spreadsheets).
type FileFilter struct {
	Desc       string
	Extensions []string
}

// FileBuilder is used for creating file browsing dialogs.
type FileBuilder struct {
	Dlg
	Filters []FileFilter
}

// File initialises a FileBuilder using the default configuration.
func File() *FileBuilder {
	return &FileBuilder{}
}

// Title specifies the title to be used for the dialog.
func (b *FileBuilder) Title(title string) *FileBuilder {
	b.Dlg.Title = title
	return b
}

// Filter adds a category of files to the types allowed by the dialog. Multiple
// calls to Filter are cumulative - any of the provided categories will be allowed.
// By default all files can be selected.
//
// The special extension '*' allows all files to be selected when the Filter is active.
func (b *FileBuilder) Filter(desc string, extensions ...string) *FileBuilder {
	filt := FileFilter{desc, extensions}
	if len(filt.Extensions) == 0 {
		filt.Extensions = append(filt.Extensions, "*")
	}
	b.Filters = append(b.Filters, filt)
	return b
}

// LoadMultiple spawns the file selection dialog using the configured settings,
// asking the user to select multiple files. Returns ErrCancelled as the error
// if the user cancels or closes the dialog.
func (b *FileBuilder) LoadMultiple() ([]string, error) {
	return b.loadMultiple()
}

// DirectoryBuilder is used for directory browse dialogs.
type DirectoryBuilder struct {
	Dlg
	ShowHiddenFiles bool
}

// Directory initialises a DirectoryBuilder using the default configuration.
func Directory() *DirectoryBuilder {
	return &DirectoryBuilder{}
}

// Browse spawns the directory selection dialog using the configured settings,
// asking the user to select a single folder. Returns ErrCancelled as the error
// if the user cancels or closes the dialog.
func (b *DirectoryBuilder) Browse() (string, error) {
	return b.browse()
}

// Title specifies the title to be used for the dialog.
func (b *DirectoryBuilder) Title(title string) *DirectoryBuilder {
	b.Dlg.Title = title
	return b
}

// ShowHiddenFiles sets whether hidden files should be visible in the dialog.
func (b *DirectoryBuilder) ShowHidden(show bool) *DirectoryBuilder {
	b.ShowHiddenFiles = show
	return b
}
