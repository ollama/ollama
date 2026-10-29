package winui

import (
	"errors"
	"fmt"
	"runtime"
	"syscall"
	"unsafe"

	"golang.org/x/sys/windows"
)

var (
	combase  = windows.NewLazySystemDLL("combase.dll")
	oleaut32 = windows.NewLazySystemDLL("oleaut32.dll")

	procRoInitialize                      = combase.NewProc("RoInitialize")
	procRoGetActivationFactory            = combase.NewProc("RoGetActivationFactory")
	procRoActivateInstance                = combase.NewProc("RoActivateInstance")
	procWindowsCreateString               = combase.NewProc("WindowsCreateString")
	procWindowsDeleteString               = combase.NewProc("WindowsDeleteString")
	procWindowsGetStringRawBuffer         = combase.NewProc("WindowsGetStringRawBuffer")
	procGetRestrictedErrorInfo            = combase.NewProc("GetRestrictedErrorInfo")
	procSysFreeString                     = oleaut32.NewProc("SysFreeString")
	iidIUnknown                           = guid("00000000-0000-0000-c000-000000000046")
	iidIInspectable                       = guid("af86e2e0-b12d-4c6a-9c5a-d7aa65101e90")
	iidIAgileObject                       = guid("94ea2b94-e9cc-49e0-c0ff-ee64ca8f5b90")
	errNoInterface                hresult = -0x7fffbffe // E_NOINTERFACE
	errFail                       hresult = -0x7fffbffb // E_FAIL
	errPointer                    hresult = -0x7fffbffd // E_POINTER
)

// hresult is a COM status code.
type hresult int32

func (hr hresult) err() error {
	if hr >= 0 {
		return nil
	}
	return &Error{Code: hr, Message: restrictedErrorMessage()}
}

// Error is a failed COM call. Message is the description WinRT recorded for
// the failure, such as the line of XAML that didn't parse.
type Error struct {
	Code    hresult
	Message string
}

func (e *Error) Error() string {
	if e.Message != "" {
		return fmt.Sprintf("%s (%#08x)", e.Message, uint32(e.Code))
	}
	return fmt.Sprintf("HRESULT %#08x", uint32(e.Code))
}

// Is reports whether target is an error with the same code.
func (e *Error) Is(target error) bool {
	var t *Error
	return errors.As(target, &t) && t.Code == e.Code
}

// object is a COM interface pointer. Only the vtable pointer is read; the
// array bound just allows indexing.
type object struct {
	vtbl *[1024]uintptr
}

// call calls the method in vtable slot with args. Callers convert pointers to
// uintptr in the argument list, so Go keeps what they point to on the heap:
// XAML can call back into Go during a call, and moving the stack then would
// leave Windows writing through stale addresses.
//
//go:uintptrescapes
func (o *object) call(slot int, args ...uintptr) error {
	return o.hresult(slot, args...).err()
}

// hresult is call that returns the raw status code.
//
//go:uintptrescapes
func (o *object) hresult(slot int, args ...uintptr) hresult {
	all := append([]uintptr{uintptr(unsafe.Pointer(o))}, args...)
	r, _, _ := syscall.SyscallN(o.vtbl[slot], all...)
	return hresult(uint32(r))
}

func (o *object) release() {
	if o != nil {
		syscall.SyscallN(o.vtbl[2], uintptr(unsafe.Pointer(o)))
	}
}

// queryInterface returns the interface iid of o, which the caller releases.
func (o *object) queryInterface(iid *GUID) (*object, error) {
	var out *object
	if err := o.call(0, uintptr(unsafe.Pointer(iid)), uintptr(unsafe.Pointer(&out))); err != nil {
		return nil, err
	}
	return out, nil
}

// getObject calls a method that takes at most one argument and returns an
// interface pointer.
//
//go:uintptrescapes
func (o *object) getObject(slot int, args ...uintptr) (*object, error) {
	var out *object
	var err error
	switch len(args) {
	case 0:
		err = o.call(slot, uintptr(unsafe.Pointer(&out)))
	case 1:
		err = o.call(slot, args[0], uintptr(unsafe.Pointer(&out)))
	default:
		panic("winui: getObject takes at most one argument")
	}
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (o *object) getString(slot int) (string, error) {
	var h hstring
	if err := o.call(slot, uintptr(unsafe.Pointer(&h))); err != nil {
		return "", err
	}
	defer h.delete()
	return h.String(), nil
}

func (o *object) putString(slot int, s string) error {
	h, err := newHString(s)
	if err != nil {
		return err
	}
	defer h.delete()
	return o.call(slot, uintptr(h))
}

func (o *object) getBool(slot int) (bool, error) {
	var b uint8
	err := o.call(slot, uintptr(unsafe.Pointer(&b)))
	return b != 0, err
}

func (o *object) putBool(slot int, v bool) error {
	return o.call(slot, uintptr(boolean(v)))
}

func (o *object) getInt32(slot int) (int32, error) {
	var v int32
	err := o.call(slot, uintptr(unsafe.Pointer(&v)))
	return v, err
}

func (o *object) putInt32(slot int, v int32) error {
	return o.call(slot, uintptr(uint32(v)))
}

func boolean(v bool) uint8 {
	if v {
		return 1
	}
	return 0
}

// hstring is a WinRT string handle.
type hstring uintptr

func newHString(s string) (hstring, error) {
	u, err := windows.UTF16FromString(s)
	if err != nil {
		return 0, err
	}
	var h hstring
	r, _, _ := procWindowsCreateString.Call(uintptr(unsafe.Pointer(&u[0])), uintptr(len(u)-1), uintptr(unsafe.Pointer(&h)))
	runtime.KeepAlive(u)
	if err := hresult(uint32(r)).err(); err != nil {
		return 0, err
	}
	return h, nil
}

func (h hstring) delete() {
	if h != 0 {
		procWindowsDeleteString.Call(uintptr(h))
	}
}

func (h hstring) String() string {
	if h == 0 {
		return ""
	}
	var n uint32
	r, _, _ := procWindowsGetStringRawBuffer.Call(uintptr(h), uintptr(unsafe.Pointer(&n)))
	if r == 0 || n == 0 {
		return ""
	}
	p := *(**uint16)(unsafe.Pointer(&r))
	return windows.UTF16ToString(unsafe.Slice(p, n))
}

// restrictedErrorMessage returns the description WinRT recorded for the
// last failure on this thread, if any.
func restrictedErrorMessage() string {
	if procGetRestrictedErrorInfo.Find() != nil {
		return ""
	}
	var info *object
	if r, _, _ := procGetRestrictedErrorInfo.Call(uintptr(unsafe.Pointer(&info))); r != 0 || info == nil {
		return ""
	}
	defer info.release()
	var description, restricted, capability *uint16
	var code int32
	// IRestrictedErrorInfo::GetErrorDetails
	if info.hresult(3, uintptr(unsafe.Pointer(&description)), uintptr(unsafe.Pointer(&code)),
		uintptr(unsafe.Pointer(&restricted)), uintptr(unsafe.Pointer(&capability))) < 0 {
		return ""
	}
	message := windows.UTF16PtrToString(restricted)
	if message == "" {
		message = windows.UTF16PtrToString(description)
	}
	for _, s := range []*uint16{description, restricted, capability} {
		if s != nil {
			procSysFreeString.Call(uintptr(unsafe.Pointer(s)))
		}
	}
	return message
}

// activationFactory returns the factory or statics interface iid of class.
func activationFactory(class string, iid *GUID) (*object, error) {
	h, err := newHString(class)
	if err != nil {
		return nil, err
	}
	defer h.delete()
	var out *object
	r, _, _ := procRoGetActivationFactory.Call(uintptr(h), uintptr(unsafe.Pointer(iid)), uintptr(unsafe.Pointer(&out)))
	if err := hresult(uint32(r)).err(); err != nil {
		return nil, fmt.Errorf("activation factory for %s: %w", class, err)
	}
	return out, nil
}

// activate creates class with its default constructor and returns
// interface iid of the new object.
func activate(class string, iid *GUID) (*object, error) {
	h, err := newHString(class)
	if err != nil {
		return nil, err
	}
	defer h.delete()
	var inspectable *object
	r, _, _ := procRoActivateInstance.Call(uintptr(h), uintptr(unsafe.Pointer(&inspectable)))
	if err := hresult(uint32(r)).err(); err != nil {
		return nil, fmt.Errorf("activate %s: %w", class, err)
	}
	defer inspectable.release()
	return inspectable.queryInterface(iid)
}

// create calls a composable factory's CreateInstance without an outer
// object and returns the new instance.
func create(class string, factoryIID *GUID) (*object, error) {
	factory, err := activationFactory(class, factoryIID)
	if err != nil {
		return nil, err
	}
	defer factory.release()
	var inner, instance *object
	if err := factory.call(6, 0, uintptr(unsafe.Pointer(&inner)), uintptr(unsafe.Pointer(&instance))); err != nil {
		return nil, fmt.Errorf("create %s: %w", class, err)
	}
	inner.release()
	return instance, nil
}
