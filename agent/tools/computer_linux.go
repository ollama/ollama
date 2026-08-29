//go:build linux

package tools

/*
#cgo LDFLAGS: -lX11 -lXtst -lXfixes
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>
#include <X11/extensions/XTest.h>
#include <X11/cursorfont.h>
#include <stdlib.h>
#include <string.h>

// captureScreen captures the entire root window and returns raw RGBA pixels.
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

// fakeKey sends a key press or release for the given keycode.
static void fakeKey(Display *dpy, int keycode, int press) {
	XTestFakeKeyEvent(dpy, keycode, press, CurrentTime);
	XSync(dpy, False);
}

// fakeScroll performs a scroll by the given amounts.
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

// typeUnicodeKey types a single Unicode character using XkbKeycodeToKeysym.
// Returns 0 on success, -1 if the keysym has no keycode mapping.
static int typeUnicodeKey(Display *dpy, KeySym keysym) {
	KeyCode keycode = XKeysymToKeycode(dpy, keysym);
	if (keycode == 0) {
		return -1;
	}

	// Check if this keysym needs Shift (uppercase letters, symbols)
	int needsShift = 0;
	KeySym lower, upper;
	XConvertCase(keysym, &lower, &upper);
	if (upper != lower) {
		// This keysym has shifted/unshifted variants
		KeyCode shiftedKeycode = XKeysymToKeycode(dpy, upper);
		KeyCode unshiftedKeycode = XKeysymToKeycode(dpy, lower);
		if (shiftedKeycode == keycode && unshiftedKeycode != keycode) {
			needsShift = 1;
		}
	}

	// For common symbols, check if the unshifted keysym differs
	if (!needsShift) {
		// Check if keysym is in the unshifted position for this keycode
		KeySym ks = XkbKeycodeToKeysym(dpy, keycode, 0, 0);
		if (ks != keysym) {
			// keysym is not in the unshifted position, try shifted
			ks = XkbKeycodeToKeysym(dpy, keycode, 0, 1);
			if (ks == keysym) {
				needsShift = 1;
			}
		}
	}

	if (needsShift) {
		KeyCode shiftKeycode = XKeysymToKeycode(dpy, XK_Shift_L);
		XTestFakeKeyEvent(dpy, shiftKeycode, True, CurrentTime);
		XTestFakeKeyEvent(dpy, keycode, True, CurrentTime);
		XTestFakeKeyEvent(dpy, keycode, False, CurrentTime);
		XTestFakeKeyEvent(dpy, shiftKeycode, False, CurrentTime);
	} else {
		XTestFakeKeyEvent(dpy, keycode, True, CurrentTime);
		XTestFakeKeyEvent(dpy, keycode, False, CurrentTime);
	}
	XSync(dpy, False);
	return 0;
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
	"image/color"
	"image/png"
	"strings"
	"unicode"

	"github.com/ollama/ollama/agent"
)

type linuxComputerBackend struct {
	display *C.Display
}

// NewComputerBackend returns a platform-specific computer backend for the
// local machine. Returns nil if X11 display is not available.
func NewComputerBackend() agent.ComputerBackend {
	dpy := C.openDisplay()
	if dpy == nil {
		return nil
	}
	return &linuxComputerBackend{display: dpy}
}

func (l *linuxComputerBackend) Screenshot(ctx context.Context) ([]byte, int, int, error) {
	select {
	case <-ctx.Done():
		return nil, 0, 0, ctx.Err()
	default:
	}

	var w, h C.int
	rgba := C.captureScreen(l.display, &w, &h)
	if rgba == nil {
		return nil, 0, 0, fmt.Errorf("screen capture failed — check X11 display")
	}
	defer C.free(unsafe.Pointer(rgba))

	width, height := int(w), int(h)
	rawLen := width * height * 4
	rawBytes := C.GoBytes(unsafe.Pointer(rgba), C.CFIndex(rawLen))

	img := image.NewRGBA(image.Rect(0, 0, width, height))
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			srcOff := (y*width + x) * 4
			img.SetRGBA(x, y, color.RGBA{
				R: rawBytes[srcOff+0],
				G: rawBytes[srcOff+1],
				B: rawBytes[srcOff+2],
				A: rawBytes[srcOff+3],
			})
		}
	}

	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		return nil, 0, 0, fmt.Errorf("failed to encode screenshot: %w", err)
	}

	return buf.Bytes(), width, height, nil
}

func (l *linuxComputerBackend) Click(ctx context.Context, x, y int) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}
	C.fakeClick(l.display, C.int(x), C.int(y))
	return nil
}

func (l *linuxComputerBackend) DoubleClick(ctx context.Context, x, y int) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}
	C.fakeDoubleClick(l.display, C.int(x), C.int(y))
	return nil
}

func (l *linuxComputerBackend) Move(ctx context.Context, x, y int) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}
	C.warpPointer(l.display, C.int(x), C.int(y))
	return nil
}

func (l *linuxComputerBackend) Type(ctx context.Context, text string) error {
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

func (l *linuxComputerBackend) typeChar(r rune) error {
	if r == '\n' {
		C.fakeKey(l.display, C.int(36), 1) // Enter keycode
		C.fakeKey(l.display, C.int(36), 0)
		return nil
	}
	if r == '\t' {
		C.fakeKey(l.display, C.int(23), 1) // Tab keycode
		C.fakeKey(l.display, C.int(23), 0)
		return nil
	}
	if r == ' ' {
		C.fakeKey(l.display, C.int(65), 1) // Space keycode
		C.fakeKey(l.display, C.int(65), 0)
		return nil
	}

	// For all other characters, use X11's keysym/keycode mapping.
	// This correctly handles: letters, digits, punctuation, symbols.
	var keysym C.KeySym
	if r <= 0x7F {
		// ASCII range: use the character value directly as keysym
		keysym = C.KeySym(r)
	} else if unicode.IsPrint(r) {
		// Non-ASCII printable: use Unicode keysym convention
		keysym = C.KeySym(0x01000000 + r)
	} else {
		return fmt.Errorf("unsupported character: %c", r)
	}

	ret := C.typeUnicodeKey(l.display, keysym)
	if ret != 0 {
		return fmt.Errorf("no keycode mapping for character: %c (keysym 0x%x)", r, keysym)
	}
	return nil
}

func (l *linuxComputerBackend) Key(ctx context.Context, key string) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	upper := strings.ToUpper(strings.TrimSpace(key))
	parts := strings.Split(upper, "+")

	// Identify modifiers
	var modifiers []uint32
	for _, part := range parts[:len(parts)-1] {
		trimmed := strings.TrimSpace(part)
		vk, ok := linuxKeyMapSingle(trimmed)
		if !ok {
			return fmt.Errorf("unknown modifier: %s", trimmed)
		}
		modifiers = append(modifiers, vk)
	}

	// Press modifiers
	for _, m := range modifiers {
		C.fakeKey(l.display, C.int(m), 1)
	}

	// Press and release the main key
	mainKey := strings.TrimSpace(parts[len(parts)-1])
	vk, ok := linuxKeyMapSingle(mainKey)
	if !ok {
		// Release modifiers on error
		for i := len(modifiers) - 1; i >= 0; i-- {
			C.fakeKey(l.display, C.int(modifiers[i]), 0)
		}
		return fmt.Errorf("unknown key name: %s", mainKey)
	}
	C.fakeKey(l.display, C.int(vk), 1)
	C.fakeKey(l.display, C.int(vk), 0)

	// Release modifiers in reverse order
	for i := len(modifiers) - 1; i >= 0; i-- {
		C.fakeKey(l.display, C.int(modifiers[i]), 0)
	}
	return nil
}

func (l *linuxComputerBackend) Scroll(ctx context.Context, dx, dy int) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}
	C.fakeScroll(l.display, C.int(dx), C.int(dy))
	return nil
}

// linuxKeyMapSingle maps a single key name (no modifiers) to its X11 keycode.
func linuxKeyMapSingle(name string) (uint32, bool) {
	switch name {
	case "A": return 38, true
	case "B": return 56, true
	case "C": return 54, true
	case "D": return 40, true
	case "E": return 26, true
	case "F": return 41, true
	case "G": return 42, true
	case "H": return 43, true
	case "I": return 46, true
	case "J": return 44, true
	case "K": return 45, true
	case "L": return 46, true
	case "M": return 58, true
	case "N": return 57, true
	case "O": return 32, true
	case "P": return 33, true
	case "Q": return 24, true
	case "R": return 27, true
	case "S": return 39, true
	case "T": return 28, true
	case "U": return 30, true
	case "V": return 55, true
	case "W": return 25, true
	case "X": return 53, true
	case "Y": return 29, true
	case "Z": return 52, true
	case "0": return 19, true
	case "1": return 10, true
	case "2": return 11, true
	case "3": return 12, true
	case "4": return 13, true
	case "5": return 14, true
	case "6": return 15, true
	case "7": return 16, true
	case "8": return 17, true
	case "9": return 18, true
	case "ENTER", "RETURN": return 36, true
	case "TAB": return 23, true
	case "SPACE": return 65, true
	case "BACKSPACE", "BACK": return 22, true
	case "DELETE", "DEL": return 119, true
	case "ESCAPE", "ESC": return 9, true
	case "SHIFT": return 50, true
	case "CTRL", "CONTROL": return 37, true
	case "ALT": return 64, true
	case "CAPSLOCK": return 66, true
	case "NUMLOCK": return 77, true
	case "LEFT": return 113, true
	case "RIGHT": return 114, true
	case "DOWN": return 116, true
	case "UP": return 111, true
	case "PAGEUP": return 112, true
	case "PAGEDOWN": return 117, true
	case "HOME": return 110, true
	case "END": return 115, true
	case "INSERT": return 106, true
	case "F1": return 67, true
	case "F2": return 68, true
	case "F3": return 69, true
	case "F4": return 70, true
	case "F5": return 71, true
	case "F6": return 72, true
	case "F7": return 73, true
	case "F8": return 74, true
	case "F9": return 75, true
	case "F10": return 76, true
	case "F11": return 95, true
	case "F12": return 96, true
	}
	return 0, false
}

// Ensure compile-time interface compliance.
var _ agent.ComputerBackend = (*linuxComputerBackend)(nil)
