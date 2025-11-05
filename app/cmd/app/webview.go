//go:build windows || darwin

package main

// #include "menu.h"
import "C"

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"
	"unsafe"

	"github.com/ollama/ollama/app/dialog"
	"github.com/ollama/ollama/app/store"
	"github.com/ollama/ollama/app/webview"
)

type Webview struct {
	port    int
	token   string
	webview webview.WebView
	mutex   sync.Mutex

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
				// Fix scrollbar styling for Edge WebView2 on Windows only
				function updateScrollbarStyles() {
					const isDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
					const existingStyle = document.getElementById('scrollbar-style');
					if (existingStyle) existingStyle.remove();
					
					const style = document.createElement('style');
					style.id = 'scrollbar-style';
					
					if (isDark) {
						style.textContent = ` + "`" + `
							::-webkit-scrollbar { width: 6px !important; height: 6px !important; }
							::-webkit-scrollbar-track { background: #1a1a1a !important; }
							::-webkit-scrollbar-thumb { background: #404040 !important; border-radius: 6px !important; }
							::-webkit-scrollbar-thumb:hover { background: #505050 !important; }
							::-webkit-scrollbar-corner { background: #1a1a1a !important; }
							::-webkit-scrollbar-button { 
								background: transparent !important;
								border: none !important;
								width: 0px !important;
								height: 0px !important;
								margin: 0 !important;
								padding: 0 !important;
							}
							::-webkit-scrollbar-button:vertical:start:decrement {
								background: transparent !important;
								height: 0px !important;
							}
							::-webkit-scrollbar-button:vertical:end:increment {
								background: transparent !important;
								height: 0px !important;
							}
							::-webkit-scrollbar-button:horizontal:start:decrement {
								background: transparent !important;
								width: 0px !important;
							}
							::-webkit-scrollbar-button:horizontal:end:increment {
								background: transparent !important;
								width: 0px !important;
							}
						` + "`" + `;
					} else {
						style.textContent = ` + "`" + `
							::-webkit-scrollbar { width: 6px !important; height: 6px !important; }
							::-webkit-scrollbar-track { background: #f0f0f0 !important; }
							::-webkit-scrollbar-thumb { background: #c0c0c0 !important; border-radius: 6px !important; }
							::-webkit-scrollbar-thumb:hover { background: #a0a0a0 !important; }
							::-webkit-scrollbar-corner { background: #f0f0f0 !important; }
							::-webkit-scrollbar-button { 
								background: transparent !important;
								border: none !important;
								width: 0px !important;
								height: 0px !important;
								margin: 0 !important;
								padding: 0 !important;
							}
							::-webkit-scrollbar-button:vertical:start:decrement {
								background: transparent !important;
								height: 0px !important;
							}
							::-webkit-scrollbar-button:vertical:end:increment {
								background: transparent !important;
								height: 0px !important;
							}
							::-webkit-scrollbar-button:horizontal:start:decrement {
								background: transparent !important;
								width: 0px !important;
							}
							::-webkit-scrollbar-button:horizontal:end:increment {
								background: transparent !important;
								width: 0px !important;
							}
						` + "`" + `;
					}
					document.head.appendChild(style);
				}
				
				window.addEventListener('load', updateScrollbarStyles);
				window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', updateScrollbarStyles);
			`
		}
		// on windows make ctrl+n open new chat
		// TODO (jmorganca): later we should use proper accelerators
		// once we introduce a native menu for the window
		// this is only used on windows since macOS uses the proper accelerators
		if runtime.GOOS == "windows" {
			init += `
				document.addEventListener('keydown', function(e) {
					if ((e.ctrlKey || e.metaKey) && e.key === 'n') {
						e.preventDefault();
						// Use the existing navigation method
						history.pushState({}, '', '/c/new');
						window.dispatchEvent(new PopStateEvent('popstate'));
						return false;
					}
				});
			`
		}

		init += `
			window.OLLAMA_WEBSEARCH = true;
		`

		wv.Init(init)

		// Add keyboard handler for zoom
		wv.Init(`
			window.addEventListener('keydown', function(e) {
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

		wv.Bind("close", func() {
			hideWindow(wv.Window())
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

		// Bind selectFiles function for selecting multiple files at once
		wv.Bind("selectFiles", func() {
			go func() {
				// Helper function to call the JavaScript callback with data or null
				callCallback := func(data interface{}) {
					dataJSON, _ := json.Marshal(data)
					wv.Dispatch(func() {
						wv.Eval(fmt.Sprintf("window.__selectFilesCallback && window.__selectFilesCallback(%s)", dataJSON))
					})
				}

				// Define allowed extensions for native dialog filtering
				textExts := []string{
					"pdf", "docx", "txt", "md", "csv", "json", "xml", "html", "htm",
					"js", "jsx", "ts", "tsx", "py", "java", "cpp", "c", "cc", "h", "cs", "php", "rb",
					"go", "rs", "swift", "kt", "scala", "sh", "bat", "yaml", "yml", "toml", "ini",
					"cfg", "conf", "log", "rtf",
				}
				imageExts := []string{"png", "jpg", "jpeg", "webp"}
				allowedExts := append(textExts, imageExts...)

				// Use native multiple file selection with extension filtering
				filenames, err := dialog.File().
					Filter("Supported Files", allowedExts...).
					Title("Select Files").
					LoadMultiple()
				if err != nil {
					slog.Debug("Multiple file selection cancelled or failed", "error", err)
					callCallback(nil)
					return
				}

				if len(filenames) == 0 {
					callCallback(nil)
					return
				}

				var files []map[string]string
				maxFileSize := int64(10 * 1024 * 1024) // 10MB

				for _, filename := range filenames {
					// Check file extension (double-check after native dialog filtering)
					ext := strings.ToLower(strings.TrimPrefix(filepath.Ext(filename), "."))
					validExt := false
					for _, allowedExt := range allowedExts {
						if ext == allowedExt {
							validExt = true
							break
						}
					}
					if !validExt {
						slog.Warn("file extension not allowed, skipping", "filename", filepath.Base(filename), "extension", ext)
						continue
					}

					// Check file size before reading (pre-filter large files)
					fileStat, err := os.Stat(filename)
					if err != nil {
						slog.Error("failed to get file info", "error", err, "filename", filename)
						continue
					}

					if fileStat.Size() > maxFileSize {
						slog.Warn("file too large, skipping", "filename", filepath.Base(filename), "size", fileStat.Size())
						continue
					}

					fileBytes, err := os.ReadFile(filename)
					if err != nil {
						slog.Error("failed to read file", "error", err, "filename", filename)
						continue
					}

					mimeType := http.DetectContentType(fileBytes)
					dataURL := fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(fileBytes))

					fileResult := map[string]string{
						"filename": filepath.Base(filename),
						"path":     filename,
						"dataURL":  dataURL,
					}

					files = append(files, fileResult)
				}

				if len(files) == 0 {
					callCallback(nil)
				} else {
					callCallback(files)
				}
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

		// Add binding for working directory selection
		wv.Bind("selectWorkingDirectory", func() {
			go func() {
				// Helper function to call the JavaScript callback with data or null
				callCallback := func(data interface{}) {
					dataJSON, _ := json.Marshal(data)
					wv.Dispatch(func() {
						wv.Eval(fmt.Sprintf("window.__selectWorkingDirectoryCallback && window.__selectWorkingDirectoryCallback(%s)", dataJSON))
					})
				}

				directory, err := dialog.Directory().Title("Select Working Directory").ShowHidden(true).Browse()
				if err != nil {
					slog.Debug("Directory selection cancelled or failed", "error", err)
					callCallback(nil)
					return
				}
				slog.Debug("Directory selected", "path", directory)
				callCallback(directory)
			}()
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

		if w.Store != nil {
			width, height, err := w.Store.WindowSize()
			if err != nil {
				slog.Error("failed to get window size", "error", err)
			}
			if width > 0 && height > 0 {
				wv.SetSize(width, height, webview.HintNone)
			} else {
				wv.SetSize(800, 600, webview.HintNone)
			}
		}
		wv.SetSize(800, 600, webview.HintMin)

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

func (w *Webview) Terminate() {
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
