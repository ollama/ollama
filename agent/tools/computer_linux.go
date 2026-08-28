//go:build linux

package tools

/*
#cgo LDFLAGS: -lX11 -lXtst -lXfixes
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/extensions/XTest.h>
#include <X11/cursorfont.h>
#include <stdlib.h>
#include <string.h>

// captureScreen captures the entire root window and returns raw RGBA pixels.
// The caller must free the returned buffer.
static unsigned char* captureScreen(Display *dpy, int *width, int *height) {
	Window root = DefaultRootWindow(dpy);
	XWindowAttributes attr;
	XGetWindowAttributes(dpy, root, &attr);
	int w = attr.width;
	int h = attr.height;
	*width = w;
	*height = h;

	XImage *image = XGetImage(dpy, root, 0, 0, w, h, AllPlanes, ZPixmap);
	if (!image) {
		return NULL;
	}

	// Convert XImage (typically BGRA or RGBA) to RGBA bytes
	unsigned char *rgba = (unsigned char *)malloc(w * h * 4);
	if (!rgba) {
		XDestroyImage(image);
		return NULL;
	}

	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			unsigned long pixel = XGetPixel(image, x, y);
			int off = (y * w + x) * 4;
			rgba[off + 0] = (pixel >> 16) & 0xFF; // R
			rgba[off + 1] = (pixel >> 8) & 0xFF;  // G
			rgba[off + 2] = pixel & 0xFF;          // B
			rgba[off + 3] = 0xFF;                  // A
		}
	}

	XDestroyImage(image);
	return rgba;
}

// warpPointer moves the mouse pointer to (x, y).
static void warpPointer(Display *dpy, int x, int y) {
	Window root = DefaultRootWindow(dpy);
	XWarpPointer(dpy, None, root, 0, 0, 0, 0, x, y);
	XSync(dpy, False);
}

// fakeClick performs a single click at (x, y) using XTest.
static void fakeClick(Display *dpy, int x, int y) {
	Window root = DefaultRootWindow(dpy);
	XWarpPointer(dpy, None, root, 0, 0, 0, 0, x, y);
	XSync(dpy, False);

	XTestFakeButtonEvent(dpy, 1, True, CurrentTime);
	XTestFakeButtonEvent(dpy, 1, False, CurrentTime);
	XSync(dpy, False);
}

// fakeDoubleClick performs a double-click at (x, y).
static void fakeDoubleClick(Display *dpy, int x, int y) {
	Window root = DefaultRootWindow(dpy);
	XWarpPointer(dpy, None, root, 0, 0, 0, 0, x, y);
	XSync(dpy, False);

	XTestFakeButtonEvent(dpy, 1, True, CurrentTime);
	XTestFakeButtonEvent(dpy, 1, False, CurrentTime);
	XTestFakeButtonEvent(dpy, 1, True, CurrentTime);
	XTestFakeButtonEvent(dpy, 1, False, CurrentTime);
	XSync(dpy, False);
}

// fakeKey sends a key press and release for the given keycode.
static void fakeKey(Display *dpy, int keycode, int press) {
	XTestFakeKeyEvent(dpy, keycode, press, CurrentTime);
	XSync(dpy, False);
}

// fakeScroll performs a scroll by the given amounts.
// dy > 0 scrolls down (button 5), dy < 0 scrolls up (button 4).
// dx > 0 scrolls right (button 7), dx < 0 scrolls left (button 6).
static void fakeScroll(Display *dpy, int dx, int dy) {
	if (dy > 0) {
		for (int i = 0; i < dy; i++) {
			XTestFakeButtonEvent(dpy, 5, True, CurrentTime);
			XTestFakeButtonEvent(dpy, 5, False, CurrentTime);
		}
	} else if (dy < 0) {
		for (int i = 0; i < -dy; i++) {
			XTestFakeButtonEvent(dpy, 4, True, CurrentTime);
			XTestFakeButtonEvent(dpy, 4, False, CurrentTime);
		}
	}
	if (dx > 0) {
		for (int i = 0; i < dx; i++) {
			XTestFakeButtonEvent(dpy, 7, True, CurrentTime);
			XTestFakeButtonEvent(dpy, 7, False, CurrentTime);
		}
	} else if (dx < 0) {
		for (int i = 0; i < -dx; i++) {
			XTestFakeButtonEvent(dpy, 6, True, CurrentTime);
			XTestFakeButtonEvent(dpy, 6, False, CurrentTime);
		}
	}
	XSync(dpy, False);
}

// openDisplay opens the X11 display.
static Display* openDisplay() {
	return XOpenDisplay(NULL);
}

// closeDisplay closes the X11 display.
static void closeDisplay(Display *dpy) {
	XCloseDisplay(dpy);
}
*/
import "C"

import (
	"bytes"
	"context"
	"fmt"
	"image"
	"image/png"
	"strings"
	"unicode"
	"image/color"
	"unsafe"
)

type linuxPlatform struct {
	display *C.Display
}

func newPlatform() computerPlatform {
	dpy := C.openDisplay()
	if dpy == nil {
		return nil
	}
	return &linuxPlatform{display: dpy}
}

func (l *linuxPlatform) Screenshot(ctx context.Context) (*ScreenImage, error) {
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	var w, h C.int
	rgba := C.captureScreen(l.display, &w, &h)
	if rgba == nil {
		return nil, fmt.Errorf("screen capture failed — check X11 display")
	}
	defer C.free(unsafe.Pointer(rgba))

	width, height := int(w), int(h)
	rawLen := width * height * 4
	rawBytes := C.GoBytes(unsafe.Pointer(rgba), C.CFIndex(rawLen))

	// Encode to PNG
	img := image.NewRGBA(image.Rect(0, 0, width, height))
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			srcOff := (y*width + x) * 4
			img.SetRGBA(x, y, pixelFromRGBA(
				rawBytes[srcOff+0],
				rawBytes[srcOff+1],
				rawBytes[srcOff+2],
				rawBytes[srcOff+3],
			))
		}
	}

	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		return nil, fmt.Errorf("failed to encode screenshot: %w", err)
	}

	return &ScreenImage{
		Pixels: buf.Bytes(),
		Width:  width,
		Height: height,
	}, nil
}

func (l *linuxPlatform) Click(ctx context.Context, x, y int) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}
	C.fakeClick(l.display, C.int(x), C.int(y))
	return nil
}

func (l *linuxPlatform) DoubleClick(ctx context.Context, x, y int) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}
	C.fakeDoubleClick(l.display, C.int(x), C.int(y))
	return nil
}

func (l *linuxPlatform) Move(ctx context.Context, x, y int) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}
	C.warpPointer(l.display, C.int(x), C.int(y))
	return nil
}

func (l *linuxPlatform) Type(ctx context.Context, text string) error {
	for _, r := range text {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
		if err := l.typeChar(r); err != nil {
			return err
		}
	}
	return nil
}

func (l *linuxPlatform) Key(ctx context.Context, key string) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	vk, ok := linuxKeyMap(strings.ToUpper(strings.TrimSpace(key)))
	if !ok {
		return fmt.Errorf("unknown key name: %s", key)
	}

	C.fakeKey(l.display, C.int(vk), 1) // press
	C.fakeKey(l.display, C.int(vk), 0) // release
	return nil
}

func (l *linuxPlatform) Scroll(ctx context.Context, dx, dy int) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}
	C.fakeScroll(l.display, C.int(dx), C.int(dy))
	return nil
}

func (l *linuxPlatform) typeChar(r rune) error {
	// Use XStringToKeysym and XkbKeycodeToKeysym approach
	// For printable ASCII, we can use a simple approach
	if r >= 32 && r <= 126 {
		keycode := asciiToXKeycode(r)
		if keycode != 0 {
			C.fakeKey(l.display, C.int(keycode), 1)
			C.fakeKey(l.display, C.int(keycode), 0)
			return nil
		}
	}
	// For non-ASCII characters, we'd need XIM which is complex.
	// For now, skip unsupported characters.
	if unicode.IsPrint(r) {
		return fmt.Errorf("non-ASCII character input not yet supported: %c", r)
	}
	return nil
}

func pixelFromRGBA(r, g, b, a byte) color.RGBA {
	return color.RGBA{R: r, G: g, B: b, A: a}
}

// Ensure compile-time interface compliance
var _ computerPlatform = (*linuxPlatform)(nil)

func asciiToXKeysym(ch rune) uint32 {
	if ch >= 'A' && ch <= 'Z' {
		return uint32(ch)
	}
	if ch >= 'a' && ch <= 'z' {
		return uint32(ch - 32) // uppercase
	}
	if ch >= '0' && ch <= '9' {
		return uint32(ch)
	}
	switch ch {
	case ' ':
		return 0x20
	case '\n':
		return 0xff0d
	case '\t':
		return 0xff09
	case '.':
		return 0x2e
	case ',':
		return 0x2c
	case '/':
		return 0x2f
	case '\\':
		return 0x5c
	case ';':
		return 0x3b
	case '\'':
		return 0x27
	case '[':
		return 0x5b
	case ']':
		return 0x5d
	case '-':
		return 0x2d
	case '=':
		return 0x3d
	case '`':
		return 0x60
	}
	return 0
}

func asciiToXKeycode(ch rune) uint32 {
	// X11 keycodes are typically offset by 8 from the key position
	// This is a simplified mapping; real apps use XKeysymToKeycode
	switch {
	case ch >= 'a' && ch <= 'z':
		// a=38, b=56, c=54, etc. (standard X11 keycodes)
	 keycodeMap := map[rune]uint32{
		'a': 38, 'b': 56, 'c': 54, 'd': 40, 'e': 26,
		'f': 41, 'g': 42, 'h': 43, 'i': 46, 'j': 44,
		'k': 45, 'l': 46, 'm': 58, 'n': 57, 'o': 32,
		'p': 33, 'q': 24, 'r': 27, 's': 39, 't': 28,
		'u': 30, 'v': 55, 'w': 25, 'x': 53, 'y': 29,
		'z': 52,
	 }
		if kc, ok := keycodeMap[ch]; ok {
			return kc
		}
		return 0
	case ch >= 'A' && ch <= 'Z':
		return asciiToXKeycode(ch + 32)
	case ch >= '0' && ch <= '9':
		keycodeMap := map[rune]uint32{
			'0': 19, '1': 10, '2': 11, '3': 12, '4': 13,
			'5': 14, '6': 15, '7': 16, '8': 17, '9': 18,
		}
		if kc, ok := keycodeMap[ch]; ok {
			return kc
		}
		return 0
	case ch == ' ':
		return 65
	case ch == '\n':
		return 36
	case ch == '\t':
		return 23
	}
	return 0
}

func linuxKeyMap(name string) (uint32, bool) {
	// Handle modifier combos like "CTRL+C" — extract the main key
	parts := strings.Split(name, "+")
	if len(parts) > 1 {
		name = strings.TrimSpace(parts[len(parts)-1])
	}

	switch name {
	case "A":
		return 38, true
	case "B":
		return 56, true
	case "C":
		return 54, true
	case "D":
		return 40, true
	case "E":
		return 26, true
	case "F":
		return 41, true
	case "G":
		return 42, true
	case "H":
		return 43, true
	case "I":
		return 46, true
	case "J":
		return 44, true
	case "K":
		return 45, true
	case "L":
		return 46, true
	case "M":
		return 58, true
	case "N":
		return 57, true
	case "O":
		return 32, true
	case "P":
		return 33, true
	case "Q":
		return 24, true
	case "R":
		return 27, true
	case "S":
		return 39, true
	case "T":
		return 28, true
	case "U":
		return 30, true
	case "V":
		return 55, true
	case "W":
		return 25, true
	case "X":
		return 53, true
	case "Y":
		return 29, true
	case "Z":
		return 52, true
	case "0":
		return 19, true
	case "1":
		return 10, true
	case "2":
		return 11, true
	case "3":
		return 12, true
	case "4":
		return 13, true
	case "5":
		return 14, true
	case "6":
		return 15, true
	case "7":
		return 16, true
	case "8":
		return 17, true
	case "9":
		return 18, true
	case "ENTER", "RETURN":
		return 36, true
	case "TAB":
		return 23, true
	case "SPACE":
		return 65, true
	case "BACKSPACE", "BACK":
		return 22, true
	case "DELETE", "DEL":
		return 119, true
	case "ESCAPE", "ESC":
		return 9, true
	case "SHIFT":
		return 50, true
	case "CTRL", "CONTROL":
		return 37, true
	case "ALT":
		return 64, true
	case "CAPSLOCK":
		return 66, true
	case "NUMLOCK":
		return 77, true
	case "LEFT":
		return 113, true
	case "RIGHT":
		return 114, true
	case "DOWN":
		return 116, true
	case "UP":
		return 111, true
	case "PAGEUP":
		return 112, true
	case "PAGEDOWN":
		return 117, true
	case "HOME":
		return 110, true
	case "END":
		return 115, true
	case "INSERT":
		return 106, true
	case "F1":
		return 67, true
	case "F2":
		return 68, true
	case "F3":
		return 69, true
	case "F4":
		return 70, true
	case "F5":
		return 71, true
	case "F6":
		return 72, true
	case "F7":
		return 73, true
	case "F8":
		return 74, true
	case "F9":
		return 75, true
	case "F10":
		return 76, true
	case "F11":
		return 95, true
	case "F12":
		return 96, true
	}
	return 0, false
}
