package winui

import (
	"log/slog"
	"runtime/debug"
	"sync"
	"sync/atomic"
	"syscall"
	"unsafe"
)

// delegate is a WinRT delegate implemented in Go, such as an event handler.
// XAML holds references to it, so live delegates are kept reachable from
// Go until their last reference is released.
type delegate struct {
	vtbl   *[4]uintptr
	iid    GUID
	refs   atomic.Int32
	invoke func(a, b *object)
}

var (
	liveDelegates sync.Map // *delegate -> struct{}

	// Invoke takes one argument for callbacks such as
	// ApplicationInitializationCallback and two for event handlers.
	delegate1Vtbl = [4]uintptr{
		syscall.NewCallback(delegateQueryInterface),
		syscall.NewCallback(delegateAddRef),
		syscall.NewCallback(delegateRelease),
		syscall.NewCallback(func(d *delegate, a *object) uintptr { return d.call(a, nil) }),
	}
	delegate2Vtbl = [4]uintptr{
		syscall.NewCallback(delegateQueryInterface),
		syscall.NewCallback(delegateAddRef),
		syscall.NewCallback(delegateRelease),
		syscall.NewCallback(func(d *delegate, a, b *object) uintptr { return d.call(a, b) }),
	}
)

// newDelegate returns a delegate with one reference, which the caller
// releases after handing it to WinRT.
func newDelegate(iid GUID, args int, invoke func(a, b *object)) *delegate {
	d := &delegate{iid: iid, invoke: invoke}
	if args == 1 {
		d.vtbl = &delegate1Vtbl
	} else {
		d.vtbl = &delegate2Vtbl
	}
	d.refs.Store(1)
	liveDelegates.Store(d, struct{}{})
	return d
}

func (d *delegate) ptr() uintptr {
	return uintptr(unsafe.Pointer(d))
}

func (d *delegate) release() {
	delegateRelease(d)
}

func (d *delegate) call(a, b *object) uintptr {
	if err := callGo(func() { d.invoke(a, b) }); err != nil {
		return uintptr(uint32(errFail))
	}
	return 0
}

func delegateQueryInterface(d *delegate, iid *GUID, out *unsafe.Pointer) uintptr {
	if out == nil {
		return uintptr(uint32(errPointer))
	}
	if *iid == d.iid || *iid == iidIUnknown || *iid == iidIAgileObject {
		d.refs.Add(1)
		*out = unsafe.Pointer(d)
		return 0
	}
	*out = nil
	return uintptr(uint32(errNoInterface))
}

func delegateAddRef(d *delegate) uintptr {
	return uintptr(d.refs.Add(1))
}

func delegateRelease(d *delegate) uintptr {
	n := d.refs.Add(-1)
	if n == 0 {
		liveDelegates.Delete(d)
	}
	return uintptr(n)
}

// callGo runs fn from a WinRT callback, logging a panic instead of letting
// it end the process.
func callGo(fn func()) (err error) {
	defer func() {
		if r := recover(); r != nil {
			slog.Error("panic in Windows UI callback", "panic", r, "stack", string(debug.Stack()))
			err = errFail.err()
		}
	}()
	fn()
	return nil
}
