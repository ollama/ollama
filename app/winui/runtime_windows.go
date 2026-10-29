// Package winui runs a WinUI 3 user interface from Go. It loads the Windows
// App SDK runtime with the Windows 11 dynamic dependency API, or on Windows
// 10 with the Windows App SDK's bootstrapper DLL from the program's folder.
// Programs need no XAML compiler or cgo: interfaces are called through their
// vtables, and UI is described with XAML markup that is parsed at run time.
package winui

import (
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
	"unsafe"

	"golang.org/x/sys/windows"
)

const (
	// runtimeFamily is the Windows App SDK runtime this package uses.
	runtimeFamily = "Microsoft.WindowsAppRuntime.1.8_8wekyb3d8bbwe"
	// runtimeRelease is that runtime's release, 1.8, as the bootstrapper
	// takes it.
	runtimeRelease = 0x00010008
	// runtimeMinVersion accepts any 1.8 runtime, as a PACKAGE_VERSION.
	runtimeMinVersion = 8000 << 48
	// BootstrapDLL is the Windows App SDK's bootstrapper, which programs ship
	// next to their executable to run on Windows 10.
	BootstrapDLL = "Microsoft.WindowsAppRuntime.Bootstrap.dll"
)

// RuntimeDownloadURL downloads the installer of the Windows App Runtime for
// people who don't have it.
var RuntimeDownloadURL = "https://aka.ms/windowsappsdk/1.8/latest/windowsappruntimeinstall-" +
	map[string]string{"amd64": "x64", "arm64": "arm64"}[runtime.GOARCH] + ".exe"

var (
	// ErrUnsupported means this version of Windows can't load the Windows App
	// SDK runtime: it's older than Windows 11 and the bootstrapper DLL isn't
	// next to the program.
	ErrUnsupported = errors.New("this version of Windows can't load the Windows App SDK")
	// ErrRuntimeMissing means the Windows App Runtime isn't installed.
	ErrRuntimeMissing = errors.New("the Windows App Runtime 1.8 isn't installed")
)

var (
	kernel32   = windows.NewLazySystemDLL("kernel32.dll")
	kernelbase = windows.NewLazySystemDLL("kernelbase.dll")
	user32     = windows.NewLazySystemDLL("user32.dll")

	procGetProcessHeap             = kernel32.NewProc("GetProcessHeap")
	procHeapFree                   = kernel32.NewProc("HeapFree")
	procTryCreatePackageDependency = kernelbase.NewProc("TryCreatePackageDependency")
	procAddPackageDependency       = kernelbase.NewProc("AddPackageDependency")
	procRegisterClassExW           = user32.NewProc("RegisterClassExW")
	procCreateWindowExW            = user32.NewProc("CreateWindowExW")
	procDefWindowProcW             = user32.NewProc("DefWindowProcW")
	procPostMessageW               = user32.NewProc("PostMessageW")
	procSetThreadDpiAwareness      = user32.NewProc("SetThreadDpiAwarenessContext")
)

var (
	iidIApplicationStatics               = guid("4e0d09f5-4358-512c-a987-503b52848e95")
	iidIApplicationFactory               = guid("9fd96657-5294-5a65-a1db-4fea143597da")
	iidIApplicationOverrides             = guid("a33e81ef-c665-503b-8827-d27ef1720a06")
	iidIApplication3                     = guid("be941595-61fe-5b36-a3d3-962a647d7c6f")
	iidApplicationInitializationCallback = guid("d8eef1c9-1234-56f1-9963-45dd9c80a661")
	iidIXamlMetadataProvider             = guid("a96251f0-2214-5d53-8746-ce99a2593cd7")
	iidIResourceDictionary               = guid("1b690975-a710-5783-a6e1-15836f6186c2")
	iidUnhandledExceptionEventHandler    = guid("3427c1b6-5eca-5631-84b8-5bae732fb67f")
	iidIUnhandledExceptionEventArgs      = guid("59eaeba9-8f9c-5be7-9b3b-820960faa220")
)

const (
	// stateRepositoryDependencyNotResolved is returned when a framework
	// package isn't installed.
	stateRepositoryDependencyNotResolved hresult = -0x7f98ffea // 0x80670016
	// errNoMatch is the bootstrapper's error when no runtime is installed,
	// HRESULT_FROM_WIN32(ERROR_NO_MATCH).
	errNoMatch hresult = -0x7ff8fb6f // 0x80070491
)

var (
	startOnce sync.Once
	startErr  error
	// stopped is closed when the UI thread finishes.
	stopped = make(chan struct{})
)

// Start loads the Windows App SDK runtime and starts XAML on a new UI
// thread, with resources as the application's ResourceDictionary markup.
// It returns once XAML is ready; later calls return the first result.
func Start(resources string) error {
	startOnce.Do(func() {
		started := make(chan error, 1)
		go runApplication(resources, started)
		startErr = <-started
	})
	return startErr
}

// Stop shuts XAML down and waits for the UI thread to finish. Call it
// before the process exits: exiting while XAML runs can hang.
func Stop() {
	exit := func() {
		if currentApp != nil && currentApp.instance != nil {
			currentApp.instance.call(17) // Exit
		}
	}
	if !Post(exit) {
		return
	}
	select {
	case <-stopped:
	case <-time.After(5 * time.Second):
		slog.Warn("XAML didn't stop")
	}
}

// runApplication runs the XAML message loop until Stop.
func runApplication(resources string, started chan<- error) {
	runtime.LockOSThread()
	// Loading the runtime fails on Windows 10 before calling anything newer.
	if err := loadWindowsAppRuntime(); err != nil {
		started <- err
		close(stopped)
		return
	}
	// Windows on this thread scale with each display. The rest of the
	// process keeps its own DPI awareness.
	const perMonitorAwareV2 = ^uintptr(3) // DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
	procSetThreadDpiAwareness.Call(perMonitorAwareV2)
	if r, _, _ := procRoInitialize.Call(0); hresult(uint32(r)) < 0 { // RO_INIT_SINGLETHREADED
		started <- fmt.Errorf("initialize Windows Runtime: %w", hresult(uint32(r)).err())
		close(stopped)
		return
	}
	statics, err := activationFactory("Microsoft.UI.Xaml.Application", &iidIApplicationStatics)
	if err != nil {
		started <- err
		close(stopped)
		return
	}

	var once sync.Once
	report := func(err error) { once.Do(func() { started <- err }) }
	app := &application{resources: resources, launched: report}
	app.outer = appInterface{vtbl: &appOuterVtbl, app: app}
	app.overrides = appInterface{vtbl: &appOverridesVtbl, app: app}
	app.provider = appInterface{vtbl: &appProviderVtbl, app: app}
	currentApp = app

	initialize := newDelegate(iidApplicationInitializationCallback, 1, func(_, _ *object) {
		if err := app.create(); err != nil {
			report(err)
		}
	})
	if err := statics.call(7, initialize.ptr()); err != nil { // Start
		report(fmt.Errorf("run XAML: %w", err))
	}
	initialize.release()
	report(errors.New("XAML stopped"))

	// XAML's other threads use this one until the process exits, as they
	// would in an app whose main thread ran XAML, so it must not end.
	close(stopped)
	select {}
}

// loadWindowsAppRuntime adds the Windows App SDK framework package to this
// process so its WinRT classes can be activated.
func loadWindowsAppRuntime() error {
	if procTryCreatePackageDependency.Find() != nil || procAddPackageDependency.Find() != nil {
		// Windows 10 has no dynamic dependency API; the bootstrapper brings one.
		exe, err := os.Executable()
		if err != nil {
			return err
		}
		return bootstrapWindowsAppRuntime(filepath.Join(filepath.Dir(exe), BootstrapDLL))
	}
	family, err := windows.UTF16PtrFromString(runtimeFamily)
	if err != nil {
		return err
	}
	var architecture uintptr = 0x4 // PackageDependencyProcessorArchitectures_X64
	if runtime.GOARCH == "arm64" {
		architecture = 0x10 // PackageDependencyProcessorArchitectures_Arm64
	}
	var id *uint16
	r, _, _ := procTryCreatePackageDependency.Call(
		0, // the current user
		uintptr(unsafe.Pointer(family)),
		runtimeMinVersion,
		architecture,
		0, // PackageDependencyLifetimeKind_Process
		0, // no lifetime artifact
		0, // CreatePackageDependencyOptions_None
		uintptr(unsafe.Pointer(&id)),
	)
	if hr := hresult(uint32(r)); hr == stateRepositoryDependencyNotResolved {
		return ErrRuntimeMissing
	} else if err := hr.err(); err != nil {
		return fmt.Errorf("find the Windows App Runtime: %w", err)
	}
	defer heapFree(unsafe.Pointer(id))

	var context uintptr
	var fullName *uint16
	r, _, _ = procAddPackageDependency.Call(uintptr(unsafe.Pointer(id)), 0, 0,
		uintptr(unsafe.Pointer(&context)), uintptr(unsafe.Pointer(&fullName)))
	if err := hresult(uint32(r)).err(); err != nil {
		return fmt.Errorf("load the Windows App Runtime: %w", err)
	}
	slog.Debug("loaded Windows App Runtime", "package", windows.UTF16PtrToString(fullName))
	heapFree(unsafe.Pointer(fullName))
	return nil
}

// bootstrapWindowsAppRuntime loads the runtime with the bootstrapper DLL at
// path.
func bootstrapWindowsAppRuntime(path string) error {
	initialize := windows.NewLazyDLL(path).NewProc("MddBootstrapInitialize2")
	if initialize.Find() != nil {
		return ErrUnsupported
	}
	r, _, _ := initialize.Call(runtimeRelease, 0, runtimeMinVersion, 0) // no version tag, MddBootstrapInitializeOptions_None
	// Without the whole runtime, which the runtime's installer brings, the
	// bootstrapper can't resolve the framework package.
	if hr := hresult(uint32(r)); hr == errNoMatch || hr == stateRepositoryDependencyNotResolved {
		return ErrRuntimeMissing
	} else if err := hr.err(); err != nil {
		return fmt.Errorf("load the Windows App Runtime: %w", err)
	}
	slog.Debug("loaded Windows App Runtime with the bootstrapper", "path", path)
	return nil
}

func heapFree(p unsafe.Pointer) {
	if p == nil {
		return
	}
	heap, _, _ := procGetProcessHeap.Call()
	procHeapFree.Call(heap, 0, uintptr(p))
}

// application is the Go half of the XAML Application. The framework's
// Application aggregates it: QueryInterface answers for the interfaces Go
// implements and forwards the rest to the framework's inner object.
type application struct {
	outer     appInterface // IInspectable, the controlling unknown
	overrides appInterface // IApplicationOverrides
	provider  appInterface // IXamlMetadataProvider

	refs      atomic.Int32
	inner     *object // the framework's non-delegating IInspectable
	instance  *object // IApplication
	metadata  *object // IXamlMetadataProvider for the WinUI controls
	resources string
	launched  func(error)
}

// appInterface is one interface of application, as seen by COM.
type appInterface struct {
	vtbl *[9]uintptr
	app  *application
}

// currentApp keeps the application reachable from Go while XAML uses it.
var currentApp *application

var (
	appOuterVtbl = [9]uintptr{
		syscall.NewCallback(appQueryInterface),
		syscall.NewCallback(appAddRef),
		syscall.NewCallback(appRelease),
		syscall.NewCallback(appGetIids),
		syscall.NewCallback(appGetRuntimeClassName),
		syscall.NewCallback(appGetTrustLevel),
	}
	appOverridesVtbl = [9]uintptr{
		appOuterVtbl[0], appOuterVtbl[1], appOuterVtbl[2], appOuterVtbl[3], appOuterVtbl[4], appOuterVtbl[5],
		syscall.NewCallback(appOnLaunched),
	}
	appProviderVtbl = [9]uintptr{
		appOuterVtbl[0], appOuterVtbl[1], appOuterVtbl[2], appOuterVtbl[3], appOuterVtbl[4], appOuterVtbl[5],
		syscall.NewCallback(providerGetXamlType),
		syscall.NewCallback(providerGetXamlTypeByFullName),
		syscall.NewCallback(providerGetXmlnsDefinitions),
	}
)

// create makes the composed Application. XAML calls OnLaunched once Start
// is running its message loop.
func (a *application) create() error {
	factory, err := activationFactory("Microsoft.UI.Xaml.Application", &iidIApplicationFactory)
	if err != nil {
		return err
	}
	defer factory.release()
	if err := factory.call(6, uintptr(unsafe.Pointer(&a.outer)),
		uintptr(unsafe.Pointer(&a.inner)), uintptr(unsafe.Pointer(&a.instance))); err != nil {
		return fmt.Errorf("create application: %w", err)
	}
	// XAML resolves the WinUI control types, such as NavigationView, through
	// the application's metadata provider.
	a.metadata, err = activate("Microsoft.UI.Xaml.XamlTypeInfo.XamlControlsXamlMetaDataProvider", &iidIXamlMetadataProvider)
	if err != nil {
		return err
	}
	return nil
}

func (a *application) launch() error {
	unhandled := newDelegate(iidUnhandledExceptionEventHandler, 2, func(_, args *object) {
		logUnhandledException(args)
	})
	defer unhandled.release()
	var token int64
	if err := a.instance.call(15, unhandled.ptr(), uintptr(unsafe.Pointer(&token))); err != nil {
		return fmt.Errorf("handle XAML exceptions: %w", err)
	}

	// Keep XAML running while no window is open.
	app3, err := a.instance.queryInterface(&iidIApplication3)
	if err != nil {
		return err
	}
	err = app3.putInt32(7, 1) // DispatcherShutdownMode = OnExplicitShutdown
	app3.release()
	if err != nil {
		return fmt.Errorf("set shutdown mode: %w", err)
	}

	resources, err := load(a.resources)
	if err != nil {
		return fmt.Errorf("load resources: %w", err)
	}
	defer resources.release()
	dictionary, err := resources.queryInterface(&iidIResourceDictionary)
	if err != nil {
		return err
	}
	defer dictionary.release()
	if err := a.instance.call(7, uintptr(unsafe.Pointer(dictionary))); err != nil { // put_Resources
		return fmt.Errorf("set resources: %w", err)
	}
	return createDispatchWindow()
}

func logUnhandledException(args *object) {
	e, err := args.queryInterface(&iidIUnhandledExceptionEventArgs)
	if err != nil {
		return
	}
	defer e.release()
	message, _ := e.getString(7)
	slog.Error("unhandled exception in Windows UI", "error", message)
	e.putBool(9, true) // Handled
}

func appQueryInterface(this *appInterface, iid *GUID, out *unsafe.Pointer) uintptr {
	if out == nil {
		return uintptr(uint32(errPointer))
	}
	a := this.app
	var p unsafe.Pointer
	switch *iid {
	case iidIUnknown, iidIInspectable:
		p = unsafe.Pointer(&a.outer)
	case iidIApplicationOverrides:
		p = unsafe.Pointer(&a.overrides)
	case iidIXamlMetadataProvider:
		p = unsafe.Pointer(&a.provider)
	default:
		if a.inner == nil {
			*out = nil
			return uintptr(uint32(errNoInterface))
		}
		return uintptr(uint32(a.inner.hresult(0, uintptr(unsafe.Pointer(iid)), uintptr(unsafe.Pointer(out)))))
	}
	a.refs.Add(1)
	*out = p
	return 0
}

func appAddRef(this *appInterface) uintptr {
	return uintptr(this.app.refs.Add(1))
}

// appRelease never frees the application, which lives as long as XAML.
func appRelease(this *appInterface) uintptr {
	return uintptr(this.app.refs.Add(-1))
}

func appGetIids(_ *appInterface, count *uint32, iids *uintptr) uintptr {
	*count, *iids = 0, 0
	return 0
}

func appGetRuntimeClassName(this *appInterface, name *hstring) uintptr {
	if inner := this.app.inner; inner != nil {
		return uintptr(uint32(inner.hresult(4, uintptr(unsafe.Pointer(name)))))
	}
	*name = 0
	return 0
}

func appGetTrustLevel(_ *appInterface, level *int32) uintptr {
	*level = 0 // BaseTrust
	return 0
}

func appOnLaunched(this *appInterface, _ uintptr) uintptr {
	a := this.app
	err := callGo(func() {
		if err := a.launch(); err != nil {
			a.launched(err)
			return
		}
		a.launched(nil)
	})
	if err != nil {
		a.launched(err)
	}
	return 0
}

// The metadata provider forwards to the WinUI controls' provider. On amd64
// GetXamlType receives its TypeName struct by reference; on arm64 the
// struct's two fields arrive in registers.
func providerGetXamlType(this *appInterface, a, b, c uintptr) uintptr {
	m := this.app.metadata
	if m == nil {
		return uintptr(uint32(errFail))
	}
	if runtime.GOARCH == "amd64" {
		return uintptr(uint32(m.hresult(6, a, b)))
	}
	return uintptr(uint32(m.hresult(6, a, b, c)))
}

func providerGetXamlTypeByFullName(this *appInterface, name, result uintptr) uintptr {
	m := this.app.metadata
	if m == nil {
		return uintptr(uint32(errFail))
	}
	return uintptr(uint32(m.hresult(7, name, result)))
}

func providerGetXmlnsDefinitions(this *appInterface, count, definitions uintptr) uintptr {
	m := this.app.metadata
	if m == nil {
		return uintptr(uint32(errFail))
	}
	return uintptr(uint32(m.hresult(8, count, definitions)))
}

// The dispatch window is a message-only window on the UI thread that runs
// functions posted from other goroutines.
const wmRun = 0x8000 + 1 // WM_APP + 1

var (
	dispatchWindow uintptr
	dispatchMu     sync.Mutex
	dispatchQueue  []func()
)

type wndClassEx struct {
	size       uint32
	style      uint32
	wndProc    uintptr
	clsExtra   int32
	wndExtra   int32
	instance   windows.Handle
	icon       windows.Handle
	cursor     windows.Handle
	background windows.Handle
	menuName   *uint16
	className  *uint16
	iconSm     windows.Handle
}

func createDispatchWindow() error {
	var module windows.Handle
	if err := windows.GetModuleHandleEx(0, nil, &module); err != nil {
		return err
	}
	className, err := windows.UTF16PtrFromString("OllamaWinUIDispatch")
	if err != nil {
		return err
	}
	class := wndClassEx{
		wndProc:   syscall.NewCallback(dispatchWndProc),
		instance:  module,
		className: className,
	}
	class.size = uint32(unsafe.Sizeof(class))
	if r, _, err := procRegisterClassExW.Call(uintptr(unsafe.Pointer(&class))); r == 0 {
		return fmt.Errorf("register dispatch window: %w", err)
	}
	const hwndMessage = ^uintptr(2) // HWND_MESSAGE
	hwnd, _, err := procCreateWindowExW.Call(0, uintptr(unsafe.Pointer(className)), 0, 0, 0, 0, 0, 0, hwndMessage, 0, uintptr(module), 0)
	if hwnd == 0 {
		return fmt.Errorf("create dispatch window: %w", err)
	}
	dispatchMu.Lock()
	dispatchWindow = hwnd
	dispatchMu.Unlock()
	return nil
}

func dispatchWndProc(hwnd, msg, wParam, lParam uintptr) uintptr {
	if msg != wmRun {
		r, _, _ := procDefWindowProcW.Call(hwnd, msg, wParam, lParam)
		return r
	}
	dispatchMu.Lock()
	queue := dispatchQueue
	dispatchQueue = nil
	dispatchMu.Unlock()
	for _, fn := range queue {
		callGo(fn)
	}
	return 0
}

// Post runs fn on the UI thread. It reports false if the UI isn't running.
func Post(fn func()) bool {
	dispatchMu.Lock()
	defer dispatchMu.Unlock()
	if dispatchWindow == 0 {
		return false
	}
	dispatchQueue = append(dispatchQueue, fn)
	if len(dispatchQueue) == 1 {
		procPostMessageW.Call(dispatchWindow, wmRun, 0, 0)
	}
	return true
}
