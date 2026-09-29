//go:build windows || darwin

package main

// #include "menu.h"
import "C"

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/ollama/ollama/app/dialog"
	"github.com/ollama/ollama/app/store"
	"github.com/ollama/ollama/app/webview"
)

const (
	defaultWindowWidth     = 1360
	defaultWindowHeight    = 960
	onboardingWindowWidth  = 900
	onboardingWindowHeight = 660
	minimumWindowWidth     = onboardingWindowWidth
	minimumWindowHeight    = onboardingWindowHeight
)

type Webview struct {
	port       int
	token      string
	webview    webview.WebView
	mutex      sync.Mutex
	onboarding atomic.Bool

	Store *store.Store
}

// Run initializes the webview and starts its event loop.
// Note: this must be called from the primary app thread
// This returns the OS native window handle to the caller
func (w *Webview) Run(path string) unsafe.Pointer {
	var url string
	if devMode {
		// In development mode, use the local dev server
		url = fmt.Sprintf("http://localhost:5173%s", path)
	} else {
		url = fmt.Sprintf("http://127.0.0.1:%d%s", w.port, path)
	}
	w.mutex.Lock()
	defer w.mutex.Unlock()

	if w.webview == nil {
		// Note: turning on debug on macos throws errors but is marginally functional for debugging
		// TODO (jmorganca): we should pre-create the window and then provide it here to
		// webview so we can hide it from the start and make other modifications
		wv := webview.New(debug)
		// start the window hidden
		hideWindow(wv.Window())
		wv.SetTitle("Ollama")

		// TODO (jmorganca): this isn't working yet since it needs to be set
		// on the first page load, ideally in an interstitial page like `/token`
		// that exists only to set the cookie and redirect to /
		// wv.Init(fmt.Sprintf(`document.cookie = "token=%s; path=/"`, w.token))
		init := `
		// Disable reload
		document.addEventListener('keydown', function(e) {
			if ((e.ctrlKey || e.metaKey) && e.key === 'r') {
				e.preventDefault();
				return false;
			}
		});

		// Prevent back/forward navigation
		window.addEventListener('popstate', function(e) {
			e.preventDefault();
			history.pushState(null, '', window.location.pathname);
			return false;
		});

		// Clear history on load
		window.addEventListener('load', function() {
			history.pushState(null, '', window.location.pathname);
			window.history.replaceState(null, '', window.location.pathname);
		});

		// Set token cookie
		document.cookie = "token=` + w.token + `; path=/";
	`
		// Windows-specific scrollbar styling
		if runtime.GOOS == "windows" {
			init += `
				// Keep Edge WebView2 scrollbars aligned with the system theme.
				function updateScrollbarStyles() {
					const existingStyle = document.getElementById('scrollbar-style');
					if (existingStyle) existingStyle.remove();

					const style = document.createElement('style');
					style.id = 'scrollbar-style';
					style.textContent = ` + "`" + `
						::-webkit-scrollbar { width: 6px !important; height: 6px !important; }
						::-webkit-scrollbar-track { background: #f0f0f0 !important; }
						::-webkit-scrollbar-thumb { background: #c0c0c0 !important; border-radius: 6px !important; }
						::-webkit-scrollbar-thumb:hover { background: #a0a0a0 !important; }
						::-webkit-scrollbar-corner { background: #f0f0f0 !important; }
						@media (prefers-color-scheme: dark) {
							::-webkit-scrollbar-track { background: #1a1a1a !important; }
							::-webkit-scrollbar-thumb { background: #404040 !important; }
							::-webkit-scrollbar-thumb:hover { background: #505050 !important; }
							::-webkit-scrollbar-corner { background: #1a1a1a !important; }
						}
						::-webkit-scrollbar-button {
							background: transparent !important;
							border: none !important;
							width: 0px !important;
							height: 0px !important;
							margin: 0 !important;
							padding: 0 !important;
						}
					` + "`" + `;
					document.head.appendChild(style);
				}

				window.addEventListener('load', updateScrollbarStyles);
			`
		}
		init += fmt.Sprintf(`
			window.OLLAMA_PLATFORM = %q;
		`, runtime.GOOS)

		wv.Init(init)

		// Add keyboard handler for zoom
		wv.Init(`
			window.addEventListener('keydown', function(e) {
				const isZoomShortcut = (e.metaKey || e.ctrlKey) && (
					e.key === '+' || e.key === '=' || e.key === '-' ||
					e.key === '_' || e.key === '0' ||
					e.code === 'NumpadAdd' || e.code === 'NumpadSubtract'
				);

				// Keep fixed-scale onboarding and apps pages at their intended size.
				const isFixedScalePage =
					window.location.pathname === '/onboarding' ||
					window.location.pathname === '/connect';
				if (isFixedScalePage && isZoomShortcut) {
					e.preventDefault();
					e.stopImmediatePropagation();
					return false;
				}

				// CMD/Ctrl + Plus/Equals (zoom in)
				if ((e.metaKey || e.ctrlKey) && (e.key === '+' || e.key === '=')) {
					e.preventDefault();
					window.zoomIn && window.zoomIn();
					return false;
				}

				// CMD/Ctrl + Minus (zoom out)
				if ((e.metaKey || e.ctrlKey) && e.key === '-') {
					e.preventDefault();
					window.zoomOut && window.zoomOut();
					return false;
				}

				// CMD/Ctrl + 0 (reset zoom)
				if ((e.metaKey || e.ctrlKey) && e.key === '0') {
					e.preventDefault();
					window.zoomReset && window.zoomReset();
					return false;
				}
			}, true);
		`)

		wv.Bind("zoomIn", func() {
			current := wv.GetZoom()
			wv.SetZoom(current + 0.1)
		})

		wv.Bind("zoomOut", func() {
			current := wv.GetZoom()
			wv.SetZoom(current - 0.1)
		})

		wv.Bind("zoomReset", func() {
			wv.SetZoom(1.0)
		})

		wv.Bind("ready", func() {
			showWindow(wv.Window())
		})

		wv.Bind("activateOllama", func() {
			showWindow(wv.Window())
		})

		bindClaudeDesktop(wv)
		bindCodexDesktop(wv)

		wv.Bind("close", func() {
			hideWindow(wv.Window())
		})

		wv.Bind("setOnboardingWindow", func(enabled bool) {
			w.onboarding.Store(enabled)
			wv.Dispatch(func() {
				if enabled {
					wv.SetSize(onboardingWindowWidth, onboardingWindowHeight, webview.HintFixed)
					setOnboardingWindowStyle(wv.Window(), true)
					return
				}

				if runtime.GOOS == "darwin" {
					// Keep the current frame through the handoff. SetSize also
					// recenters the macOS window and would jump before Apps paints.
					setOnboardingWindowStyle(wv.Window(), false)
					return
				}

				width, height := defaultWindowWidth, defaultWindowHeight
				if w.Store != nil {
					storedWidth, storedHeight, err := w.Store.WindowSize()
					if err != nil {
						slog.Error("failed to restore window size", "error", err)
					} else if storedWidth > 0 && storedHeight > 0 {
						width, height = storedWidth, storedHeight
					}
				}

				wv.SetSize(width, height, webview.HintNone)
				wv.SetSize(minimumWindowWidth, minimumWindowHeight, webview.HintMin)
				setOnboardingWindowStyle(wv.Window(), false)
			})
		})

		// Webviews do not allow access to the file system by default, so we need to
		// bind file system operations here
		wv.Bind("selectModelsDirectory", func() {
			go func() {
				// Helper function to call the JavaScript callback with data or null
				callCallback := func(data interface{}) {
					dataJSON, _ := json.Marshal(data)
					wv.Dispatch(func() {
						wv.Eval(fmt.Sprintf("window.__selectModelsDirectoryCallback && window.__selectModelsDirectoryCallback(%s)", dataJSON))
					})
				}

				directory, err := dialog.Directory().Title("Select Model Directory").ShowHidden(true).Browse()
				if err != nil {
					slog.Debug("Directory selection cancelled or failed", "error", err)
					callCallback(nil)
					return
				}
				slog.Debug("Directory selected", "path", directory)
				callCallback(directory)
			}()
		})

		wv.Bind("drag", func() {
			wv.Dispatch(func() {
				drag(wv.Window())
			})
		})

		wv.Bind("doubleClick", func() {
			wv.Dispatch(func() {
				doubleClick(wv.Window())
			})
		})

		wv.Bind("setContextMenuItems", func(items []map[string]interface{}) error {
			menuMutex.Lock()
			defer menuMutex.Unlock()

			if len(menuItems) > 0 {
				pinner.Unpin()
			}

			menuItems = nil
			for _, item := range items {
				menuItem := C.menuItem{
					label:     C.CString(item["label"].(string)),
					enabled:   0,
					separator: 0,
				}

				if item["enabled"] != nil {
					menuItem.enabled = 1
				}

				if item["separator"] != nil {
					menuItem.separator = 1
				}
				menuItems = append(menuItems, menuItem)
			}
			return nil
		})

		// Debounce resize events
		var resizeTimer *time.Timer
		var resizeMutex sync.Mutex

		wv.Bind("resize", func(width, height int) {
			if w.Store != nil {
				resizeMutex.Lock()
				if resizeTimer != nil {
					resizeTimer.Stop()
				}
				resizeTimer = time.AfterFunc(100*time.Millisecond, func() {
					err := w.Store.SetWindowSize(width, height)
					if err != nil {
						slog.Error("failed to set window size", "error", err)
					}
				})
				resizeMutex.Unlock()
			}
		})

		// On Darwin, we can't have 2 threads both running global event loops
		// but on Windows, the event loops are tied to the window, so we're
		// able to run in both the tray and webview
		if runtime.GOOS != "darwin" {
			slog.Debug("starting webview event loop")
			go func() {
				wv.Run()
				slog.Debug("webview event loop exited")
			}()
		}

		width, height := defaultWindowWidth, defaultWindowHeight
		if w.Store != nil {
			storedWidth, storedHeight, err := w.Store.WindowSize()
			if err != nil {
				slog.Error("failed to get window size", "error", err)
			}
			if storedWidth > 0 && storedHeight > 0 {
				width, height = storedWidth, storedHeight
			}
		}
		wv.SetSize(width, height, webview.HintNone)
		wv.SetSize(minimumWindowWidth, minimumWindowHeight, webview.HintMin)

		w.webview = wv
		w.webview.Navigate(url)
	} else {
		w.webview.Eval(fmt.Sprintf(`
			history.pushState({}, '', '%s');
		`, path))
		showWindow(w.webview.Window())
	}

	return w.webview.Window()
}

// pickExportPath opens native pickers on the initialized UI thread. In
// particular, the Windows folder picker requires that thread's COM apartment.
func (w *Webview) pickExportPath(pick func() (string, error)) (string, error) {
	w.mutex.Lock()
	if w.webview == nil {
		w.mutex.Unlock()
		return "", fmt.Errorf("export is unavailable in this window")
	}
	var path string
	var err error
	done := make(chan struct{})
	w.webview.Dispatch(func() {
		path, err = pick()
		close(done)
	})
	w.mutex.Unlock()
	<-done
	return path, err
}

func (w *Webview) Terminate() {
	w.onboarding.Store(false)
	w.mutex.Lock()
	if w.webview == nil {
		w.mutex.Unlock()
		return
	}

	wv := w.webview
	w.webview = nil
	w.mutex.Unlock()
	wv.Terminate()
	wv.Destroy()
}

func (w *Webview) OnboardingActive() bool {
	return w.onboarding.Load()
}

func (w *Webview) IsRunning() bool {
	w.mutex.Lock()
	defer w.mutex.Unlock()
	return w.webview != nil
}

var (
	menuItems []C.menuItem
	menuMutex sync.RWMutex
	pinner    runtime.Pinner
)

//export menu_get_item_count
func menu_get_item_count() C.int {
	menuMutex.RLock()
	defer menuMutex.RUnlock()
	return C.int(len(menuItems))
}

//export menu_get_items
func menu_get_items() unsafe.Pointer {
	menuMutex.RLock()
	defer menuMutex.RUnlock()

	if len(menuItems) == 0 {
		return nil
	}

	// Return pointer to the slice data
	pinner.Pin(&menuItems[0])
	return unsafe.Pointer(&menuItems[0])
}

//export menu_handle_selection
func menu_handle_selection(item *C.char) {
	wv.webview.Eval(fmt.Sprintf("window.handleContextMenuResult('%s')", C.GoString(item)))
}
