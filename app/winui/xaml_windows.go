package winui

import (
	"errors"
	"fmt"
	"unsafe"

	"golang.org/x/sys/windows"
)

var (
	iidIXamlReaderStatics           = guid("82a4cd9e-435e-5aeb-8c4f-300cece45cae")
	iidIWindowFactory               = guid("f0441536-afef-5222-918f-324a9b2dec75")
	iidIWindow2                     = guid("42febaa5-1c32-522a-a591-57618c6f665d")
	iidIWindowNative                = guid("eecdbf0e-bae9-4cb6-a68e-9598e1cb57bb")
	iidIWindowEventArgs             = guid("1140827c-fe0a-5268-bc2b-f4492c2ccb49")
	iidIMicaBackdropFactory         = guid("774379ce-74bd-59d4-849d-d99c4184d838")
	iidISystemBackdrop              = guid("5aeed5c4-37ac-5852-b73f-1b76ebc3205f")
	iidIOverlappedPresenter3        = guid("55d26138-4c38-57e7-a0c1-d467b774db8c")
	iidIUIElement                   = guid("c3c01020-320c-5cf6-9d24-d396bbfa4d8b")
	iidIFrameworkElement            = guid("fe08f13d-dc6a-5495-ad44-c2d8d21863b0")
	iidIControl                     = guid("857d6e8a-d45a-5c69-a99c-bf6a5c54fb38")
	iidIContentControl              = guid("07e81761-11b2-52ae-8f8b-4d53d2b5900a")
	iidITextBlock                   = guid("1ac8d84f-392c-5c7e-83f5-a53e3bf0abb0")
	iidIToggleSwitch                = guid("1b17eeb1-74bf-5a83-8161-a86f0fdcdf24")
	iidIButtonBase                  = guid("65714269-2473-5327-a652-0ea6bce7f403")
	iidISelector                    = guid("8f7e2159-e61d-576f-8476-f83fde3d689e")
	iidIInfoBar                     = guid("c1c3a438-dd79-5d22-9e42-5a3cdf8113a9")
	iidIPersonPicture               = guid("30ec982c-0efa-5538-a356-e9ebd5aa363c")
	iidIProgressRing                = guid("2670d03f-e28c-5652-bee2-b5212ebdf7ff")
	iidIBitmapImageFactory          = guid("f037e0e9-f229-522e-95c9-da2211a14b05")
	iidIImageSource                 = guid("6c2038f6-d6d5-55e9-9b9e-082f12dbff60")
	iidIPropertyValueStatics        = guid("629bdbc8-d932-4ff4-96b9-8d96c5c1e858")
	iidIUriRuntimeClassFactory      = guid("44a9796f-723e-4fdf-a218-033e75b0c084")
	iidIReferenceInt32              = guid("548cefbd-bc8a-5fa0-8df2-957440fc8bf4")
	iidIWindowActivatedEventArgs    = guid("c723a5ea-82c4-5dd6-861b-70ef573b88d6")
	iidIComboBox                    = guid("c77da58b-4fd7-51e0-a431-f84658a83e9e")
	iidIFlyoutBase                  = guid("bb6603bf-744d-5c31-a87d-744394634d77")
	iidIToolTipServiceStatics       = guid("5aa38adc-9874-5e0a-8d8e-1574efc0b88f")
	iidIDependencyObject            = guid("e7beaee7-160e-50f7-8789-d63463f979fa")
	iidRoutedEventHandler           = guid("dae23d85-69ca-5bdf-805b-6161a3a215cc")
	iidSelectionChangedEventHandler = guid("a232390d-0e34-595e-8931-fa928a9909f4")
	iidWindowClosedEventHandler     = typedEventHandlerIID(inspectableSignature,
		runtimeClassSignature("Microsoft.UI.Xaml.WindowEventArgs", iidIWindowEventArgs))
	iidWindowActivatedEventHandler = typedEventHandlerIID(inspectableSignature,
		runtimeClassSignature("Microsoft.UI.Xaml.WindowActivatedEventArgs", iidIWindowActivatedEventArgs))

	procGetDpiForWindow     = user32.NewProc("GetDpiForWindow")
	procSetForegroundWindow = user32.NewProc("SetForegroundWindow")
	procGetCursorPos        = user32.NewProc("GetCursorPos")
	procMonitorFromPoint    = user32.NewProc("MonitorFromPoint")
	procMonitorFromWindow   = user32.NewProc("MonitorFromWindow")
	procGetMonitorInfoW     = user32.NewProc("GetMonitorInfoW")
	procGetWindowRect       = user32.NewProc("GetWindowRect")
	procSetWindowPos        = user32.NewProc("SetWindowPos")
)

// typedEventHandlerIID returns the identifier of
// TypedEventHandler<sender, args> from the type signatures of its arguments.
func typedEventHandlerIID(sender, args string) GUID {
	return parameterizedIID("pinterface({9de1c534-6ae1-11e0-84e1-18a905bcc53f};" + sender + ";" + args + ")")
}

// runtimeClassSignature is the generic-argument signature of a runtime
// class, named with its default interface.
func runtimeClassSignature(class string, defaultInterface GUID) string {
	return "rc(" + class + ";{" + defaultInterface.String() + "})"
}

// inspectableSignature is the generic-argument signature of Object.
const inspectableSignature = "cinterface(IInspectable)"

// load parses XAML markup and returns the root object.
func load(xaml string) (*object, error) {
	reader, err := activationFactory("Microsoft.UI.Xaml.Markup.XamlReader", &iidIXamlReaderStatics)
	if err != nil {
		return nil, err
	}
	defer reader.release()
	h, err := newHString(xaml)
	if err != nil {
		return nil, err
	}
	defer h.delete()
	return reader.getObject(6, uintptr(h)) // Load
}

// Window is a top-level XAML window. Use it only on the UI thread.
type Window struct {
	window  *object // IWindow
	content *object // IFrameworkElement of the window's content
}

// NewWindow creates a hidden window showing content, which is XAML markup
// whose root is a FrameworkElement.
func NewWindow(title, content string) (*Window, error) {
	window, err := create("Microsoft.UI.Xaml.Window", &iidIWindowFactory)
	if err != nil {
		return nil, err
	}
	w := &Window{window: window}
	if err := w.setContent(content); err != nil {
		w.release()
		return nil, err
	}
	if err := window.putString(15, title); err != nil { // put_Title
		w.release()
		return nil, err
	}
	return w, nil
}

func (w *Window) setContent(content string) error {
	root, err := load(content)
	if err != nil {
		return fmt.Errorf("load window content: %w", err)
	}
	defer root.release()
	element, err := root.queryInterface(&iidIUIElement)
	if err != nil {
		return err
	}
	defer element.release()
	if err := w.window.call(9, uintptr(unsafe.Pointer(element))); err != nil { // put_Content
		return err
	}
	w.content, err = root.queryInterface(&iidIFrameworkElement)
	return err
}

func (w *Window) release() {
	w.content.release()
	w.window.release()
	w.content, w.window = nil, nil
}

// ExtendIntoTitleBar draws the window's content under the title bar and
// makes the element named titleBar the area that drags the window.
func (w *Window) ExtendIntoTitleBar(titleBar string) error {
	e, err := w.Element(titleBar)
	if err != nil {
		return err
	}
	defer e.Release()
	element, err := e.obj.queryInterface(&iidIUIElement)
	if err != nil {
		return err
	}
	defer element.release()
	if err := w.window.putBool(17, true); err != nil { // put_ExtendsContentIntoTitleBar
		return err
	}
	return w.window.call(28, uintptr(unsafe.Pointer(element))) // SetTitleBar
}

// UseMica gives the window the Mica material, which tints it with the
// desktop background.
func (w *Window) UseMica() error {
	mica, err := create("Microsoft.UI.Xaml.Media.MicaBackdrop", &iidIMicaBackdropFactory)
	if err != nil {
		return err
	}
	defer mica.release()
	backdrop, err := mica.queryInterface(&iidISystemBackdrop)
	if err != nil {
		return err
	}
	defer backdrop.release()
	window2, err := w.window.queryInterface(&iidIWindow2)
	if err != nil {
		return err
	}
	defer window2.release()
	return window2.call(7, uintptr(unsafe.Pointer(backdrop))) // put_SystemBackdrop
}

func (w *Window) appWindow() (*object, error) {
	window2, err := w.window.queryInterface(&iidIWindow2)
	if err != nil {
		return nil, err
	}
	defer window2.release()
	return window2.getObject(8) // get_AppWindow
}

// hwnd returns the window's handle.
func (w *Window) hwnd() windows.HWND {
	native, err := w.window.queryInterface(&iidIWindowNative)
	if err != nil {
		return 0
	}
	defer native.release()
	var hwnd windows.HWND
	native.call(3, uintptr(unsafe.Pointer(&hwnd))) // get_WindowHandle
	return hwnd
}

// scale converts device-independent pixels to the window's pixels.
func (w *Window) scale(v int) int32 {
	dpi, _, _ := procGetDpiForWindow.Call(uintptr(w.hwnd()))
	if dpi == 0 {
		dpi = 96
	}
	return int32(v * int(dpi) / 96)
}

// Resize sets the window's size in device-independent pixels.
func (w *Window) Resize(width, height int) error {
	appWindow, err := w.appWindow()
	if err != nil {
		return err
	}
	defer appWindow.release()
	size := uint64(uint32(w.scale(width))) | uint64(uint32(w.scale(height)))<<32 // SizeInt32
	return appWindow.call(22, uintptr(size))                                     // Resize
}

// SetMinimumSize keeps people from making the window smaller than width by
// height device-independent pixels.
func (w *Window) SetMinimumSize(width, height int) error {
	appWindow, err := w.appWindow()
	if err != nil {
		return err
	}
	defer appWindow.release()
	presenter, err := appWindow.getObject(12) // get_Presenter
	if err != nil {
		return err
	}
	defer presenter.release()
	overlapped, err := presenter.queryInterface(&iidIOverlappedPresenter3)
	if err != nil {
		return err
	}
	defer overlapped.release()
	for slot, v := range map[int]int{9: width, 7: height} { // put_PreferredMinimumWidth, put_PreferredMinimumHeight
		boxed, err := boxInt32(w.scale(v))
		if err != nil {
			return err
		}
		err = overlapped.call(slot, uintptr(unsafe.Pointer(boxed)))
		boxed.release()
		if err != nil {
			return err
		}
	}
	return nil
}

// SetIcon shows the .ico file at path in the title bar and taskbar.
func (w *Window) SetIcon(path string) error {
	appWindow, err := w.appWindow()
	if err != nil {
		return err
	}
	defer appWindow.release()
	return appWindow.putString(23, path) // SetIcon
}

// Activate shows the window and brings it to the front.
func (w *Window) Activate() error {
	if err := w.window.call(26); err != nil { // Activate
		return err
	}
	procSetForegroundWindow.Call(uintptr(w.hwnd()))
	return nil
}

// OnClosed calls fn after the window closes. The window can't be used
// after that.
func (w *Window) OnClosed(fn func()) error {
	d := newDelegate(iidWindowClosedEventHandler, 2, func(_, _ *object) {
		fn()
		w.release()
	})
	defer d.release()
	var token int64
	return w.window.call(20, d.ptr(), uintptr(unsafe.Pointer(&token))) // add_Closed
}

// OnActivated calls fn when the window becomes active or inactive.
func (w *Window) OnActivated(fn func(active bool)) error {
	d := newDelegate(iidWindowActivatedEventHandler, 2, func(_, args *object) {
		e, err := args.queryInterface(&iidIWindowActivatedEventArgs)
		if err != nil {
			return
		}
		state, err := e.getInt32(8) // get_WindowActivationState
		e.release()
		if err == nil {
			fn(state != 1) // Deactivated
		}
	})
	defer d.release()
	var token int64
	return w.window.call(18, d.ptr(), uintptr(unsafe.Pointer(&token))) // add_Activated
}

type rect struct {
	left, top, right, bottom int32
}

type monitorInfo struct {
	size    uint32
	monitor rect
	work    rect
	flags   uint32
}

// Center moves the window to the middle of the display with the pointer,
// where people expect windows they open from the notification area.
func (w *Window) Center() error {
	hwnd := uintptr(w.hwnd())
	const nearest = 2 // MONITOR_DEFAULTTONEAREST
	var monitor uintptr
	var cursor struct{ x, y int32 }
	if r, _, _ := procGetCursorPos.Call(uintptr(unsafe.Pointer(&cursor))); r != 0 {
		point := uintptr(uint32(cursor.x)) | uintptr(uint32(cursor.y))<<32
		monitor, _, _ = procMonitorFromPoint.Call(point, nearest)
	} else {
		// Without a pointer, as in a disconnected remote session, use the
		// window's display.
		monitor, _, _ = procMonitorFromWindow.Call(hwnd, nearest)
	}
	info := monitorInfo{}
	info.size = uint32(unsafe.Sizeof(info))
	if r, _, err := procGetMonitorInfoW.Call(monitor, uintptr(unsafe.Pointer(&info))); r == 0 {
		return err
	}
	var window rect
	if r, _, err := procGetWindowRect.Call(hwnd, uintptr(unsafe.Pointer(&window))); r == 0 {
		return err
	}
	width, height := window.right-window.left, window.bottom-window.top
	x := info.work.left + max(0, (info.work.right-info.work.left-width)/2)
	y := info.work.top + max(0, (info.work.bottom-info.work.top-height)/2)
	const flags = 0x1 | 0x4 | 0x10 // SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE
	if r, _, err := procSetWindowPos.Call(hwnd, 0, uintptr(x), uintptr(y), 0, 0, flags); r == 0 {
		return err
	}
	return nil
}

// Element returns the element named name with x:Name in the window's
// content. Release it when done.
func (w *Window) Element(name string) (*Element, error) {
	h, err := newHString(name)
	if err != nil {
		return nil, err
	}
	defer h.delete()
	obj, err := w.content.getObject(77, uintptr(h)) // FindName
	if err != nil {
		return nil, err
	}
	if obj == nil {
		return nil, fmt.Errorf("no element named %q", name)
	}
	return &Element{obj: obj}, nil
}

// Element is a XAML element. Its methods apply to the element types that
// support them and return an error for others.
type Element struct {
	obj *object // IInspectable
}

// Release drops the reference to the element.
func (e *Element) Release() {
	if e != nil {
		e.obj.release()
		e.obj = nil
	}
}

// errNoElement is returned by methods of an element that wasn't found.
var errNoElement = errors.New("no such element")

// with calls fn with the element's interface iid.
func (e *Element) with(iid *GUID, fn func(*object) error) error {
	if e == nil || e.obj == nil {
		return errNoElement
	}
	o, err := e.obj.queryInterface(iid)
	if err != nil {
		return err
	}
	defer o.release()
	return fn(o)
}

// SetVisible shows the element or collapses it out of the layout.
func (e *Element) SetVisible(visible bool) error {
	var visibility int32 = 1 // Collapsed
	if visible {
		visibility = 0
	}
	return e.with(&iidIUIElement, func(o *object) error { return o.putInt32(24, visibility) })
}

// SetEnabled enables or disables a control.
func (e *Element) SetEnabled(enabled bool) error {
	return e.with(&iidIControl, func(o *object) error { return o.putBool(29, enabled) })
}

// SetText sets a TextBlock's text.
func (e *Element) SetText(text string) error {
	return e.with(&iidITextBlock, func(o *object) error { return o.putString(27, text) })
}

// SetContent sets a content control's content, such as a button's label,
// to text.
func (e *Element) SetContent(text string) error {
	boxed, err := boxString(text)
	if err != nil {
		return err
	}
	defer boxed.release()
	return e.with(&iidIContentControl, func(o *object) error { return o.call(7, uintptr(unsafe.Pointer(boxed))) })
}

// IsOn reports whether a ToggleSwitch is on.
func (e *Element) IsOn() bool {
	var on bool
	e.with(&iidIToggleSwitch, func(o *object) (err error) {
		on, err = o.getBool(6)
		return err
	})
	return on
}

// SetOn turns a ToggleSwitch on or off. It raises Toggled when the state
// changes.
func (e *Element) SetOn(on bool) error {
	return e.with(&iidIToggleSwitch, func(o *object) error { return o.putBool(7, on) })
}

// OnToggled calls fn when a ToggleSwitch changes state.
func (e *Element) OnToggled(fn func()) error {
	return e.addRoutedHandler(&iidIToggleSwitch, 21, fn)
}

// OnClick calls fn when a button is clicked.
func (e *Element) OnClick(fn func()) error {
	return e.addRoutedHandler(&iidIButtonBase, 14, fn)
}

func (e *Element) addRoutedHandler(iid *GUID, slot int, fn func()) error {
	return e.with(iid, func(o *object) error {
		d := newDelegate(iidRoutedEventHandler, 2, func(_, _ *object) { fn() })
		defer d.release()
		var token int64
		return o.call(slot, d.ptr(), uintptr(unsafe.Pointer(&token)))
	})
}

// SelectedIndex returns the index of a selector's selected item, or -1.
func (e *Element) SelectedIndex() int {
	index := int32(-1)
	e.with(&iidISelector, func(o *object) (err error) {
		index, err = o.getInt32(6)
		return err
	})
	return int(index)
}

// SetSelectedIndex selects an item of a selector such as a ComboBox. It
// raises SelectionChanged when the selection changes.
func (e *Element) SetSelectedIndex(index int) error {
	return e.with(&iidISelector, func(o *object) error { return o.putInt32(7, int32(index)) })
}

// OnSelectionChanged calls fn when a selector's selection changes.
func (e *Element) OnSelectionChanged(fn func()) error {
	return e.with(&iidISelector, func(o *object) error {
		d := newDelegate(iidSelectionChangedEventHandler, 2, func(_, _ *object) { fn() })
		defer d.release()
		var token int64
		return o.call(16, d.ptr(), uintptr(unsafe.Pointer(&token))) // add_SelectionChanged
	})
}

// InfoBarSeverity is the kind of message an InfoBar shows.
type InfoBarSeverity int32

const (
	InfoBarInformational InfoBarSeverity = iota
	InfoBarSuccess
	InfoBarWarning
	InfoBarError
)

// ShowInfoBar opens an InfoBar with a message.
func (e *Element) ShowInfoBar(severity InfoBarSeverity, title, message string) error {
	return e.with(&iidIInfoBar, func(o *object) error {
		if err := o.putInt32(13, int32(severity)); err != nil {
			return err
		}
		if err := o.putString(9, title); err != nil {
			return err
		}
		if err := o.putString(11, message); err != nil {
			return err
		}
		return o.putBool(7, true)
	})
}

// SetPerson sets a PersonPicture's name, used for its initials, and the
// URL of its picture, which may be empty.
func (e *Element) SetPerson(name, pictureURL string) error {
	return e.with(&iidIPersonPicture, func(o *object) error {
		if err := o.putString(19, name); err != nil { // put_DisplayName
			return err
		}
		var picture *object
		if pictureURL != "" {
			var err error
			if picture, err = bitmapImage(pictureURL); err != nil {
				return err
			}
			defer picture.release()
		}
		return o.call(25, uintptr(unsafe.Pointer(picture))) // put_ProfilePicture
	})
}

// SetActive starts or stops a ProgressRing.
func (e *Element) SetActive(active bool) error {
	return e.with(&iidIProgressRing, func(o *object) error { return o.putBool(7, active) })
}

// SetPlaceholderText sets the text a ComboBox shows when nothing is selected.
func (e *Element) SetPlaceholderText(text string) error {
	return e.with(&iidIComboBox, func(o *object) error { return o.putString(21, text) })
}

// HideFlyout closes a flyout.
func (e *Element) HideFlyout() error {
	return e.with(&iidIFlyoutBase, func(o *object) error { return o.call(40) })
}

// SetToolTip shows text when people hover over the element. Empty text
// removes the tooltip.
func (e *Element) SetToolTip(text string) error {
	service, err := activationFactory("Microsoft.UI.Xaml.Controls.ToolTipService", &iidIToolTipServiceStatics)
	if err != nil {
		return err
	}
	defer service.release()
	var boxed *object
	if text != "" {
		if boxed, err = boxString(text); err != nil {
			return err
		}
		defer boxed.release()
	}
	return e.with(&iidIDependencyObject, func(o *object) error {
		return service.call(14, uintptr(unsafe.Pointer(o)), uintptr(unsafe.Pointer(boxed))) // SetToolTip
	})
}

// bitmapImage returns an ImageSource that loads the image at url.
func bitmapImage(url string) (*object, error) {
	uriFactory, err := activationFactory("Windows.Foundation.Uri", &iidIUriRuntimeClassFactory)
	if err != nil {
		return nil, err
	}
	defer uriFactory.release()
	h, err := newHString(url)
	if err != nil {
		return nil, err
	}
	defer h.delete()
	uri, err := uriFactory.getObject(6, uintptr(h)) // CreateUri
	if err != nil {
		return nil, err
	}
	defer uri.release()
	imageFactory, err := activationFactory("Microsoft.UI.Xaml.Media.Imaging.BitmapImage", &iidIBitmapImageFactory)
	if err != nil {
		return nil, err
	}
	defer imageFactory.release()
	image, err := imageFactory.getObject(6, uintptr(unsafe.Pointer(uri))) // CreateInstanceWithUriSource
	if err != nil {
		return nil, err
	}
	defer image.release()
	return image.queryInterface(&iidIImageSource)
}

func propertyValues() (*object, error) {
	return activationFactory("Windows.Foundation.PropertyValue", &iidIPropertyValueStatics)
}

// boxString returns text as an IInspectable for properties of type Object.
func boxString(text string) (*object, error) {
	values, err := propertyValues()
	if err != nil {
		return nil, err
	}
	defer values.release()
	h, err := newHString(text)
	if err != nil {
		return nil, err
	}
	defer h.delete()
	return values.getObject(18, uintptr(h)) // CreateString
}

// boxInt32 returns v as an IReference<Int32>.
func boxInt32(v int32) (*object, error) {
	values, err := propertyValues()
	if err != nil {
		return nil, err
	}
	defer values.release()
	boxed, err := values.getObject(10, uintptr(uint32(v))) // CreateInt32
	if err != nil {
		return nil, err
	}
	defer boxed.release()
	return boxed.queryInterface(&iidIReferenceInt32)
}
