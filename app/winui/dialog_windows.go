package winui

import (
	"errors"
	"unsafe"

	"golang.org/x/sys/windows"
)

var (
	ole32   = windows.NewLazySystemDLL("ole32.dll")
	shell32 = windows.NewLazySystemDLL("shell32.dll")

	procCoCreateInstance            = ole32.NewProc("CoCreateInstance")
	procSHCreateItemFromParsingName = shell32.NewProc("SHCreateItemFromParsingName")

	clsidFileOpenDialog = guid("dc1c5a9c-e88a-4dde-a5a1-60f82a20aef7")
	iidIFileOpenDialog  = guid("d57c7288-d4ad-4768-be02-9d969532d960")
	iidIShellItem       = guid("43826d1e-e718-42ee-bc55-a1e261c37bfe")
)

// ErrCanceled is returned when people close a dialog without choosing.
var ErrCanceled = errors.New("canceled")

// errCanceled is HRESULT_FROM_WIN32(ERROR_CANCELLED).
const errCanceled hresult = -0x7ff8fb39 // 0x800704c7

// PickFolder asks people to choose a folder, starting in folder, and
// returns its path. It returns ErrCanceled if they don't choose one.
func (w *Window) PickFolder(title, okLabel, folder string) (string, error) {
	var dialog *object
	r, _, _ := procCoCreateInstance.Call(uintptr(unsafe.Pointer(&clsidFileOpenDialog)), 0, 1, // CLSCTX_INPROC_SERVER
		uintptr(unsafe.Pointer(&iidIFileOpenDialog)), uintptr(unsafe.Pointer(&dialog)))
	if err := hresult(uint32(r)).err(); err != nil {
		return "", err
	}
	defer dialog.release()

	var options uint32
	if err := dialog.call(10, uintptr(unsafe.Pointer(&options))); err != nil { // GetOptions
		return "", err
	}
	const pickFolders = 0x20 | 0x40 | 0x800                              // FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM | FOS_PATHMUSTEXIST
	if err := dialog.call(9, uintptr(options|pickFolders)); err != nil { // SetOptions
		return "", err
	}
	for slot, text := range map[int]string{17: title, 18: okLabel} { // SetTitle, SetOkButtonLabel
		if text == "" {
			continue
		}
		p, err := windows.UTF16PtrFromString(text)
		if err != nil {
			return "", err
		}
		if err := dialog.call(slot, uintptr(unsafe.Pointer(p))); err != nil {
			return "", err
		}
	}
	if folder != "" {
		if item, err := shellItem(folder); err == nil {
			dialog.call(12, uintptr(unsafe.Pointer(item))) // SetFolder
			item.release()
		}
	}

	hr := dialog.hresult(3, uintptr(w.hwnd())) // Show
	if hr == errCanceled {
		return "", ErrCanceled
	}
	if err := hr.err(); err != nil {
		return "", err
	}
	item, err := dialog.getObject(20) // GetResult
	if err != nil {
		return "", err
	}
	defer item.release()
	var path *uint16
	const fileSystemPath = 0x80058000                                                    // SIGDN_FILESYSPATH
	if err := item.call(5, fileSystemPath, uintptr(unsafe.Pointer(&path))); err != nil { // GetDisplayName
		return "", err
	}
	defer windows.CoTaskMemFree(unsafe.Pointer(path))
	return windows.UTF16PtrToString(path), nil
}

func shellItem(path string) (*object, error) {
	p, err := windows.UTF16PtrFromString(path)
	if err != nil {
		return nil, err
	}
	var item *object
	r, _, _ := procSHCreateItemFromParsingName.Call(uintptr(unsafe.Pointer(p)), 0,
		uintptr(unsafe.Pointer(&iidIShellItem)), uintptr(unsafe.Pointer(&item)))
	if err := hresult(uint32(r)).err(); err != nil {
		return nil, err
	}
	return item, nil
}
