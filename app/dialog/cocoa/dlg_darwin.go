package cocoa

// #cgo darwin LDFLAGS: -framework Cocoa
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

const BUFSIZE = C.PATH_MAX

func FileDlg(save bool, title string, exts []string, relaxExt bool, startDir string, filename string, showHidden bool) (string, error) {
	mode := C.LOADDLG
	if save {
		mode = C.SAVEDLG
	}
	return fileDlg(mode, title, exts, relaxExt, startDir, filename, showHidden)
}

func DirDlg(title string, startDir string, showHidden bool) (string, error) {
	return fileDlg(C.DIRDLG, title, nil, false, startDir, "", showHidden)
}

func fileDlg(mode int, title string, exts []string, relaxExt bool, startDir, filename string, showHidden bool) (string, error) {
	p := C.FileDlgParams{
		mode: C.int(mode),
		nbuf: BUFSIZE,
	}
	if showHidden {
		p.showHidden = 1
	}
	p.buf = (*C.char)(C.malloc(BUFSIZE))
	defer C.free(unsafe.Pointer(p.buf))
	buf := (*(*[BUFSIZE]byte)(unsafe.Pointer(p.buf)))[:]
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
			i := i
			cext[i] = nsStr(ext)
			defer C.NSRelease(cext[i])
		}
		p.numext = C.int(len(exts))
		if relaxExt {
			p.relaxext = 1
		}
	}
	switch C.fileDlg(&p) {
	case C.DLG_OK:
		// casting to string copies the [about-to-be-freed] bytes
		return string(buf[:bytes.Index(buf, []byte{0})]), nil
	case C.DLG_CANCEL:
		return "", nil
	case C.DLG_URLFAIL:
		return "", errors.New("failed to get file-system representation for selected URL")
	}
	panic("unhandled case")
}
