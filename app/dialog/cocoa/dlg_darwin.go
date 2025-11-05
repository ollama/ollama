package cocoa

// #cgo darwin LDFLAGS: -framework Cocoa -framework UniformTypeIdentifiers
// #include <stdlib.h>
// #include <sys/syslimits.h>
// #include "dlg.h"
import "C"

import (
	"bytes"
	"errors"
	"unsafe"
)

type AlertParams struct {
	p C.AlertDlgParams
}

func mkAlertParams(msg, title string, style C.AlertStyle) *AlertParams {
	a := AlertParams{C.AlertDlgParams{msg: C.CString(msg), style: style}}
	if title != "" {
		a.p.title = C.CString(title)
	}
	return &a
}

func (a *AlertParams) run() C.DlgResult {
	return C.alertDlg(&a.p)
}

func (a *AlertParams) free() {
	C.free(unsafe.Pointer(a.p.msg))
	if a.p.title != nil {
		C.free(unsafe.Pointer(a.p.title))
	}
}

func nsStr(s string) unsafe.Pointer {
	return C.NSStr(unsafe.Pointer(&[]byte(s)[0]), C.int(len(s)))
}

func YesNoDlg(msg, title string) bool {
	a := mkAlertParams(msg, title, C.MSG_YESNO)
	defer a.free()
	return a.run() == C.DLG_OK
}

func InfoDlg(msg, title string) {
	a := mkAlertParams(msg, title, C.MSG_INFO)
	defer a.free()
	a.run()
}

func ErrorDlg(msg, title string) {
	a := mkAlertParams(msg, title, C.MSG_ERROR)
	defer a.free()
	a.run()
}

const (
	BUFSIZE             = C.PATH_MAX
	MULTI_FILE_BUF_SIZE = 32768
)

// MultiFileDlg opens a file dialog that allows multiple file selection
func MultiFileDlg(title string, exts []string, relaxExt bool, startDir string, showHidden bool) ([]string, error) {
	return fileDlgWithOptions(C.LOADDLG, title, exts, relaxExt, startDir, "", showHidden, true)
}

// FileDlg opens a file dialog for single file selection (kept for compatibility)
func FileDlg(save bool, title string, exts []string, relaxExt bool, startDir string, filename string, showHidden bool) (string, error) {
	mode := C.LOADDLG
	if save {
		mode = C.SAVEDLG
	}
	files, err := fileDlgWithOptions(mode, title, exts, relaxExt, startDir, filename, showHidden, false)
	if err != nil {
		return "", err
	}
	if len(files) == 0 {
		return "", nil
	}
	return files[0], nil
}

func DirDlg(title string, startDir string, showHidden bool) (string, error) {
	files, err := fileDlgWithOptions(C.DIRDLG, title, nil, false, startDir, "", showHidden, false)
	if err != nil {
		return "", err
	}
	if len(files) == 0 {
		return "", nil
	}
	return files[0], nil
}

// fileDlgWithOptions is the unified file dialog function that handles both single and multiple selection
func fileDlgWithOptions(mode int, title string, exts []string, relaxExt bool, startDir, filename string, showHidden, allowMultiple bool) ([]string, error) {
	// Use larger buffer for multiple files, smaller for single
	bufSize := BUFSIZE
	if allowMultiple {
		bufSize = MULTI_FILE_BUF_SIZE
	}

	p := C.FileDlgParams{
		mode: C.int(mode),
		nbuf: C.int(bufSize),
	}

	if allowMultiple {
		p.allowMultiple = C.int(1) // Enable multiple selection //nolint:structcheck
	}
	if showHidden {
		p.showHidden = 1
	}

	p.buf = (*C.char)(C.malloc(C.size_t(bufSize)))
	defer C.free(unsafe.Pointer(p.buf))
	buf := (*(*[MULTI_FILE_BUF_SIZE]byte)(unsafe.Pointer(p.buf)))[:bufSize]

	if title != "" {
		p.title = C.CString(title)
		defer C.free(unsafe.Pointer(p.title))
	}
	if startDir != "" {
		p.startDir = C.CString(startDir)
		defer C.free(unsafe.Pointer(p.startDir))
	}
	if filename != "" {
		p.filename = C.CString(filename)
		defer C.free(unsafe.Pointer(p.filename))
	}

	if len(exts) > 0 {
		if len(exts) > 999 {
			panic("more than 999 extensions not supported")
		}
		ptrSize := int(unsafe.Sizeof(&title))
		p.exts = (*unsafe.Pointer)(C.malloc(C.size_t(ptrSize * len(exts))))
		defer C.free(unsafe.Pointer(p.exts))
		cext := (*(*[999]unsafe.Pointer)(unsafe.Pointer(p.exts)))[:]
		for i, ext := range exts {
			cext[i] = nsStr(ext)
			defer C.NSRelease(cext[i])
		}
		p.numext = C.int(len(exts))
		if relaxExt {
			p.relaxext = 1
		}
	}

	// Execute dialog and parse results
	switch C.fileDlg(&p) {
	case C.DLG_OK:
		if allowMultiple {
			// Parse multiple null-terminated strings from buffer
			var files []string
			start := 0
			for i := range len(buf) - 1 {
				if buf[i] == 0 {
					if i > start {
						files = append(files, string(buf[start:i]))
					}
					start = i + 1
					// Check for double null (end of list)
					if i+1 < len(buf) && buf[i+1] == 0 {
						break
					}
				}
			}
			return files, nil
		} else {
			// Single file - return as array for consistency
			filename := string(buf[:bytes.Index(buf, []byte{0})])
			return []string{filename}, nil
		}
	case C.DLG_CANCEL:
		return nil, nil
	case C.DLG_URLFAIL:
		return nil, errors.New("failed to get file-system representation for selected URL")
	}
	panic("unhandled case")
}
