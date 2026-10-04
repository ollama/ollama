package dialog

import (
	"fmt"
	"reflect"
	"syscall"
	"unicode/utf16"
	"unsafe"

	"github.com/TheTitanrain/w32"
)

const multiFileBufferSize = w32.MAX_PATH * 10

type WinDlgError int

func (e WinDlgError) Error() string {
	return fmt.Sprintf("CommDlgExtendedError: %#x", int(e))
}

func err() error {
	e := w32.CommDlgExtendedError()
	if e == 0 {
		return ErrCancelled
	}
	return WinDlgError(e)
}

type filedlg struct {
	buf     []uint16
	filters []uint16
	opf     *w32.OPENFILENAME
}

func (d filedlg) parseMultipleFilenames() []string {
	var files []string
	i := 0

	// Find first null terminator (directory path)
	for i < len(d.buf) && d.buf[i] != 0 {
		i++
	}

	if i >= len(d.buf) {
		return files
	}

	// Get directory path
	dirPath := string(utf16.Decode(d.buf[:i]))
	i++ // Skip null terminator

	// Check if there are more files (multiple selection)
	if i < len(d.buf) && d.buf[i] != 0 {
		// Multiple files selected - parse filenames
		for i < len(d.buf) {
			start := i
			// Find next null terminator
			for i < len(d.buf) && d.buf[i] != 0 {
				i++
			}
			if i >= len(d.buf) {
				break
			}

			if start < i {
				filename := string(utf16.Decode(d.buf[start:i]))
				if dirPath != "" {
					files = append(files, dirPath+"\\"+filename)
				} else {
					files = append(files, filename)
				}
			}
			i++ // Skip null terminator
			if i >= len(d.buf) || d.buf[i] == 0 {
				break // End of list
			}
		}
	} else {
		// Single file selected
		files = append(files, dirPath)
	}

	return files
}

func (b *FileBuilder) loadMultiple() ([]string, error) {
	d := openfile(w32.OFN_FILEMUSTEXIST|w32.OFN_NOCHANGEDIR|w32.OFN_ALLOWMULTISELECT|w32.OFN_EXPLORER, b)
	d.buf = make([]uint16, multiFileBufferSize)
	d.opf.File = utf16ptr(d.buf)
	d.opf.MaxFile = uint32(len(d.buf))

	if w32.GetOpenFileName(d.opf) {
		return d.parseMultipleFilenames(), nil
	}
	return nil, err()
}

/* syscall.UTF16PtrFromString not sufficient because we need to encode embedded NUL bytes */
func utf16ptr(utf16 []uint16) *uint16 {
	if utf16[len(utf16)-1] != 0 {
		panic("refusing to make ptr to non-NUL terminated utf16 slice")
	}
	h := (*reflect.SliceHeader)(unsafe.Pointer(&utf16))
	return (*uint16)(unsafe.Pointer(h.Data))
}

func openfile(flags uint32, b *FileBuilder) (d filedlg) {
	d.buf = make([]uint16, w32.MAX_PATH)
	d.opf = &w32.OPENFILENAME{
		File:    utf16ptr(d.buf),
		MaxFile: uint32(len(d.buf)),
		Flags:   flags,
	}
	d.opf.StructSize = uint32(unsafe.Sizeof(*d.opf))
	if b.Dlg.Title != "" {
		d.opf.Title, _ = syscall.UTF16PtrFromString(b.Dlg.Title)
	}
	for _, filt := range b.Filters {
		/* build utf16 string of form "Music File\0*.mp3;*.ogg;*.wav;\0" */
		d.filters = append(d.filters, utf16.Encode([]rune(filt.Desc))...)
		d.filters = append(d.filters, 0)
		for _, ext := range filt.Extensions {
			s := fmt.Sprintf("*.%s;", ext)
			d.filters = append(d.filters, utf16.Encode([]rune(s))...)
		}
		d.filters = append(d.filters, 0)
	}
	if d.filters != nil {
		d.filters = append(d.filters, 0, 0) // two extra NUL chars to terminate the list
		d.opf.Filter = utf16ptr(d.filters)
	}
	return d
}

type dirdlg struct {
	bi *w32.BROWSEINFO
}

func selectdir(b *DirectoryBuilder) (d dirdlg) {
	d.bi = &w32.BROWSEINFO{Flags: w32.BIF_RETURNONLYFSDIRS | w32.BIF_NEWDIALOGSTYLE}
	if b.Dlg.Title != "" {
		d.bi.Title, _ = syscall.UTF16PtrFromString(b.Dlg.Title)
	}
	return d
}

func (b *DirectoryBuilder) browse() (string, error) {
	d := selectdir(b)
	res := w32.SHBrowseForFolder(d.bi)
	if res == 0 {
		return "", ErrCancelled
	}
	return w32.SHGetPathFromIDList(res), nil
}
