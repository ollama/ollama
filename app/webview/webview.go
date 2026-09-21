//go:build windows || darwin

/*
 * MIT License
 *
 * Copyright (c) 2017 Serge Zaitsev
 * Copyright (c) 2022 Steffen André Langnes
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
package webview

/*
#cgo CFLAGS: -I${SRCDIR}/libs/webview/include
#cgo CXXFLAGS: -I${SRCDIR}/libs/webview/include -DWEBVIEW_STATIC

#cgo darwin CXXFLAGS: -DWEBVIEW_COCOA -std=c++11
#cgo darwin LDFLAGS: -framework WebKit -ldl

#cgo windows CXXFLAGS: -DWEBVIEW_EDGE -std=c++14 -I${SRCDIR}/libs/mswebview2/include
#cgo windows LDFLAGS: -static -ladvapi32 -lole32 -lshell32 -lshlwapi -luser32 -lversion

#include "webview.h"

#include <stdlib.h>
#include <stdint.h>

void CgoWebViewDispatch(webview_t w, uintptr_t arg);
void CgoWebViewBind(webview_t w, const char *name, uintptr_t index);
void CgoWebViewUnbind(webview_t w, const char *name);
void CgoWebViewReturn(webview_t w, const char *id, int status, const char *result);

void webview_set_zoom(webview_t w, double level);
double webview_get_zoom(webview_t w);
*/
import "C"

import (
	"encoding/json"
	"errors"
	"reflect"
	"runtime"
	"sync"
	"unsafe"
)

func init() {
	// Ensure that main.main is called from the main thread
	runtime.LockOSThread()
}

// Hints are used to configure window sizing and resizing
type Hint int

const (
	// Width and height are default size
	HintNone = C.WEBVIEW_HINT_NONE

	// Window size can not be changed by a user
	HintFixed = C.WEBVIEW_HINT_FIXED

	// Width and height are minimum bounds
	HintMin = C.WEBVIEW_HINT_MIN

	// Width and height are maximum bounds
	HintMax = C.WEBVIEW_HINT_MAX
)

type WebView interface {
	// Run runs the main loop until it's terminated. After this function exits -
	// you must destroy the webview.
	Run()

	// Terminate stops the main loop. It is safe to call this function from
	// a background thread.
	Terminate()

	// Dispatch posts a function to be executed on the main thread. You normally
	// do not need to call this function, unless you want to tweak the native
	// window.
	Dispatch(f func())

	// Destroy destroys a webview and closes the native window.
	Destroy()

	// Window returns a native window handle pointer. When using GTK backend the
	// pointer is GtkWindow pointer, when using Cocoa backend the pointer is
	// NSWindow pointer, when using Win32 backend the pointer is HWND pointer.
	Window() unsafe.Pointer

	// SetTitle updates the title of the native window. Must be called from the UI
	// thread.
	SetTitle(title string)

	// SetSize updates native window size. See Hint constants.
	SetSize(w int, h int, hint Hint)

	// Navigate navigates webview to the given URL. URL may be a properly encoded data.
	// URI. Examples:
	// w.Navigate("https://github.com/webview/webview")
	// w.Navigate("data:text/html,%3Ch1%3EHello%3C%2Fh1%3E")
	// w.Navigate("data:text/html;base64,PGgxPkhlbGxvPC9oMT4=")
	Navigate(url string)

	// SetHtml sets the webview HTML directly.
	// Example: w.SetHtml(w, "<h1>Hello</h1>");
	SetHtml(html string)

	// Init injects JavaScript code at the initialization of the new page. Every
	// time the webview will open a the new page - this initialization code will
	// be executed. It is guaranteed that code is executed before window.onload.
	Init(js string)

	// Eval evaluates arbitrary JavaScript code. Evaluation happens asynchronously,
	// also the result of the expression is ignored. Use RPC bindings if you want
	// to receive notifications about the results of the evaluation.
	Eval(js string)

	// Bind binds a callback function so that it will appear under the given name
	// as a global JavaScript function. Internally it uses webview_init().
	// Callback receives a request string and a user-provided argument pointer.
	// Request string is a JSON array of all the arguments passed to the
	// JavaScript function.
	//
	// f must be a function
	// f must return either value and error or just error
	Bind(name string, f interface{}) error

	// BindAsync runs the callback on a background goroutine. The callback must
	// not access native UI objects and must support concurrent calls.
	BindAsync(name string, f interface{}) error

	// Removes a callback that was previously set by Bind.
	Unbind(name string) error

	// SetZoom sets the zoom level of the webview.
	// level: 1.0 is normal size, >1.0 zooms in, <1.0 zooms out.
	SetZoom(level float64)

	// GetZoom returns the current zoom level of the webview.
	GetZoom() float64
}

type webview struct {
	w        C.webview_t
	mu       sync.Mutex
	bindings map[string]uintptr
	pending  map[uintptr]struct{}
}

type binding struct {
	call  func(id, req string) (interface{}, error)
	async bool
	owner *webview
	name  string
}

var (
	m        sync.Mutex
	index    uintptr
	dispatch = map[uintptr]func(){}
	bindings = map[uintptr]*binding{}
)

func boolToInt(b bool) C.int {
	if b {
		return 1
	}
	return 0
}

// New calls NewWindow to create a new window and a new webview instance. If debug
// is non-zero - developer tools will be enabled (if the platform supports them).
func New(debug bool) WebView { return NewWindow(debug, nil) }

// NewWindow creates a new webview instance. If debug is non-zero - developer
// tools will be enabled (if the platform supports them). Window parameter can be
// a pointer to the native window handle. If it's non-null - then child WebView is
// embedded into the given parent window. Otherwise a new window is created.
// Depending on the platform, a GtkWindow, NSWindow or HWND pointer can be passed
// here.
func NewWindow(debug bool, window unsafe.Pointer) WebView {
	w := &webview{bindings: make(map[string]uintptr), pending: make(map[uintptr]struct{})}
	w.w = C.webview_create(boolToInt(debug), window)
	return w
}

func (w *webview) Destroy() {
	w.mu.Lock()
	native := w.w
	w.w = nil
	m.Lock()
	for _, id := range w.bindings {
		delete(bindings, id)
	}
	for id := range w.pending {
		delete(dispatch, id)
	}
	m.Unlock()
	clear(w.bindings)
	clear(w.pending)
	w.mu.Unlock()
	// Native destruction may drain the dispatch queue, so do not hold mu here.
	if native != nil {
		C.webview_destroy(native)
	}
}

func (w *webview) Run() {
	C.webview_run(w.w)
}

func (w *webview) Terminate() {
	C.webview_terminate(w.w)
}

func (w *webview) Window() unsafe.Pointer {
	return C.webview_get_window(w.w)
}

func (w *webview) Navigate(url string) {
	s := C.CString(url)
	defer C.free(unsafe.Pointer(s))
	C.webview_navigate(w.w, s)
}

func (w *webview) SetHtml(html string) {
	s := C.CString(html)
	defer C.free(unsafe.Pointer(s))
	C.webview_set_html(w.w, s)
}

func (w *webview) SetTitle(title string) {
	s := C.CString(title)
	defer C.free(unsafe.Pointer(s))
	C.webview_set_title(w.w, s)
}

func (w *webview) SetSize(width int, height int, hint Hint) {
	C.webview_set_size(w.w, C.int(width), C.int(height), C.webview_hint_t(hint))
}

func (w *webview) Init(js string) {
	s := C.CString(js)
	defer C.free(unsafe.Pointer(s))
	C.webview_init(w.w, s)
}

func (w *webview) Eval(js string) {
	s := C.CString(js)
	defer C.free(unsafe.Pointer(s))
	C.webview_eval(w.w, s)
}

func (w *webview) Dispatch(f func()) {
	m.Lock()
	for ; dispatch[index] != nil; index++ {
	}
	dispatch[index] = f
	id := index
	index++
	m.Unlock()
	C.CgoWebViewDispatch(w.w, C.uintptr_t(id))
}

//export _webviewDispatchGoCallback
func _webviewDispatchGoCallback(index unsafe.Pointer) {
	m.Lock()
	f := dispatch[uintptr(index)]
	delete(dispatch, uintptr(index))
	m.Unlock()
	if f != nil {
		f()
	}
}

//export _webviewBindingGoCallback
func _webviewBindingGoCallback(w C.webview_t, id *C.char, req *C.char, index uintptr) {
	m.Lock()
	b := bindings[index]
	m.Unlock()
	if b == nil {
		return
	}
	// The C strings belong to the native callback and expire when it returns.
	requestID, request := C.GoString(id), C.GoString(req)
	b.invoke(requestID, request, func(status int, result string) {
		if b.async {
			b.owner.returnAsync(b.name, index, requestID, status, result)
			return
		}
		s := C.CString(result)
		defer C.free(unsafe.Pointer(s))
		C.webview_return(w, id, C.int(status), s)
	})
}

func (b *binding) invoke(id, req string, reply func(int, string)) {
	call := func() {
		status, result := bindingResult(b.call(id, req))
		reply(status, result)
	}
	if b.async {
		go call()
	} else {
		call()
	}
}

func bindingResult(res interface{}, err error) (int, string) {
	jsString := func(v interface{}) string { b, _ := json.Marshal(v); return string(b) }
	if err != nil {
		return -1, jsString(err.Error())
	}
	b, err := json.Marshal(res)
	if err != nil {
		return -1, jsString(err.Error())
	}
	return 0, string(b)
}

func (w *webview) returnAsync(name string, bindingID uintptr, requestID string, status int, result string) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if id, ok := w.bindings[name]; w.w == nil || !ok || id != bindingID {
		return
	}
	m.Lock()
	for ; dispatch[index] != nil; index++ {
	}
	id := index
	index++
	w.pending[id] = struct{}{}
	dispatch[id] = func() {
		w.mu.Lock()
		defer w.mu.Unlock()
		delete(w.pending, id)
		if current, ok := w.bindings[name]; w.w == nil || !ok || current != bindingID {
			return
		}
		seq, value := C.CString(requestID), C.CString(result)
		defer C.free(unsafe.Pointer(seq))
		defer C.free(unsafe.Pointer(value))
		// Already on the UI thread. Avoid a second dispatch that could outlive w.
		C.CgoWebViewReturn(w.w, seq, C.int(status), value)
	}
	m.Unlock()
	C.CgoWebViewDispatch(w.w, C.uintptr_t(id))
}

func (w *webview) Bind(name string, f interface{}) error {
	return w.bind(name, f, false)
}

func (w *webview) BindAsync(name string, f interface{}) error {
	return w.bind(name, f, true)
}

func (w *webview) bind(name string, f interface{}, async bool) error {
	v := reflect.ValueOf(f)
	// f must be a function
	if v.Kind() != reflect.Func {
		return errors.New("only functions can be bound")
	}
	// f must return either value and error or just error
	if n := v.Type().NumOut(); n > 2 {
		return errors.New("function may only return a value or a value+error")
	}

	call := func(id, req string) (interface{}, error) {
		raw := []json.RawMessage{}
		if err := json.Unmarshal([]byte(req), &raw); err != nil {
			return nil, err
		}

		isVariadic := v.Type().IsVariadic()
		numIn := v.Type().NumIn()
		if (isVariadic && len(raw) < numIn-1) || (!isVariadic && len(raw) != numIn) {
			return nil, errors.New("function arguments mismatch")
		}
		args := []reflect.Value{}
		for i := range raw {
			var arg reflect.Value
			if isVariadic && i >= numIn-1 {
				arg = reflect.New(v.Type().In(numIn - 1).Elem())
			} else {
				arg = reflect.New(v.Type().In(i))
			}
			if err := json.Unmarshal(raw[i], arg.Interface()); err != nil {
				return nil, err
			}
			args = append(args, arg.Elem())
		}
		errorType := reflect.TypeOf((*error)(nil)).Elem()
		res := v.Call(args)
		switch len(res) {
		case 0:
			// No results from the function, just return nil
			return nil, nil
		case 1:
			// One result may be a value, or an error
			if res[0].Type().Implements(errorType) {
				if res[0].Interface() != nil {
					return nil, res[0].Interface().(error)
				}
				return nil, nil
			}
			return res[0].Interface(), nil
		case 2:
			// Two results: first one is value, second is error
			if !res[1].Type().Implements(errorType) {
				return nil, errors.New("second return value must be an error")
			}
			if res[1].Interface() == nil {
				return res[0].Interface(), nil
			}
			return res[0].Interface(), res[1].Interface().(error)
		default:
			return nil, errors.New("unexpected number of return values")
		}
	}

	w.mu.Lock()
	defer w.mu.Unlock()
	if w.w == nil {
		return errors.New("webview is destroyed")
	}
	if _, ok := w.bindings[name]; ok {
		return nil
	}
	m.Lock()
	for ; bindings[index] != nil; index++ {
	}
	id := index
	index++
	bindings[id] = &binding{call: call, async: async, owner: w, name: name}
	w.bindings[name] = id
	m.Unlock()
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	C.CgoWebViewBind(w.w, cname, C.uintptr_t(id))
	return nil
}

func (w *webview) Unbind(name string) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.w == nil {
		return errors.New("webview is destroyed")
	}
	m.Lock()
	if id, ok := w.bindings[name]; ok {
		delete(bindings, id)
		delete(w.bindings, name)
	}
	m.Unlock()
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	C.CgoWebViewUnbind(w.w, cname)
	return nil
}

func (w *webview) SetZoom(level float64) {
	C.webview_set_zoom(w.w, C.double(level))
}

func (w *webview) GetZoom() float64 {
	return float64(C.webview_get_zoom(w.w))
}
