//go:build darwin

package tools

/*
#cgo LDFLAGS: -framework CoreGraphics -framework CoreFoundation -framework ApplicationServices
#include <CoreGraphics/CoreGraphics.h>
#include <CoreFoundation/CoreFoundation.h>
#include <stdlib.h>

// captureScreenshot captures the main display as a PNG.
static CFDataRef captureScreenshot(int *width, int *height) {
	CGDirectDisplayID mainDisplay = CGMainDisplayID();
	size_t w = CGDisplayPixelsWide(mainDisplay);
	size_t h = CGDisplayPixelsHigh(mainDisplay);
	CGImageRef image = CGDisplayCreateImage(mainDisplay);
	if (!image) {
		return NULL;
	}
	*width = (int)w;
	*height = (int)h;

	// Create a bitmap context and render the image into it
	CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
	uint8_t *bitmapData = (uint8_t *)calloc(w * h * 4, 1);
	CGContextRef context = CGBitmapContextCreate(
		bitmapData, w, h, 8, w * 4, colorSpace,
		kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
	if (!context) {
		free(bitmapData);
		CGColorSpaceRelease(colorSpace);
		CGImageRelease(image);
		return NULL;
	}
	CGContextDrawImage(context, CGRectMake(0, 0, w, h), image);
	CGContextRelease(context);

	// Encode to PNG using Apple's ImageIO (via CGImageDestination)
	// We return raw RGBA and let Go encode to PNG
	CGImageRelease(image);
	CGColorSpaceRelease(colorSpace);

	// Create CFData from raw pixels
	CFDataRef data = CFDataCreate(NULL, bitmapData, (CFIndex)(w * h * 4));
	free(bitmapData);
	return data;
}

// moveMouse moves the cursor to (x, y) on the main display.
static void moveMouse(int x, int y) {
	CGEventRef event = CGEventCreateMouseEvent(NULL, kCGEventMouseMoved, CGPointMake(x, y), kCGMouseButtonLeft);
	CGEventPost(kCGHIDEventTap, event);
	CFRelease(event);
}

// clickMouse performs a single click at (x, y).
static void clickMouse(int x, int y) {
	CGEventRef down = CGEventCreateMouseEvent(NULL, kCGEventLeftMouseDown, CGPointMake(x, y), kCGMouseButtonLeft);
	CGEventPost(kCGHIDEventTap, down);
	CFRelease(down);

	CGEventRef up = CGEventCreateMouseEvent(NULL, kCGEventLeftMouseUp, CGPointMake(x, y), kCGMouseButtonLeft);
	CGEventPost(kCGHIDEventTap, up);
	CFRelease(up);
}

// doubleClickMouse performs a double-click at (x, y).
static void doubleClickMouse(int x, int y) {
	CGEventRef down = CGEventCreateMouseEvent(NULL, kCGEventLeftMouseDown, CGPointMake(x, y), kCGMouseButtonLeft);
	CGEventSetIntegerValueField(down, kCGMouseEventClickState, 2);
	CGEventPost(kCGHIDEventTap, down);
	CFRelease(down);

	CGEventRef up = CGEventCreateMouseEvent(NULL, kCGEventLeftMouseUp, CGPointMake(x, y), kCGMouseButtonLeft);
	CGEventSetIntegerValueField(up, kCGMouseEventClickState, 2);
	CGEventPost(kCGHIDEventTap, up);
	CFRelease(up);
}

// typeUnicodeString types a string character by character.
static void typeUnicodeString(const uint32_t *chars, int len) {
	for (int i = 0; i < len; i++) {
		CGEventRef down = CGEventCreateKeyboardEvent(NULL, 0, true);
		CGEventKeyboardSetUnicodeString(down, 1, &chars[i]);
		CGEventPost(kCGHIDEventTap, down);
		CFRelease(down);

		CGEventRef up = CGEventCreateKeyboardEvent(NULL, 0, false);
		CGEventKeyboardSetUnicodeString(up, 1, &chars[i]);
		CGEventPost(kCGHIDEventTap, up);
		CFRelease(up);
	}
}

// scrollMouse performs a scroll at the current cursor position.
static void scrollMouse(int dx, int dy) {
	if (dy != 0) {
		CGEventRef event = CGEventCreateScrollWheelEvent(NULL, kCGScrollEventUnitPixel, 1, dy);
		CGEventPost(kCGHIDEventTap, event);
		CFRelease(event);
	}
	if (dx != 0) {
		CGEventRef event = CGEventCreateScrollWheelEvent(NULL, kCGScrollEventUnitPixel, 2, 0, dx);
		CGEventPost(kCGHIDEventTap, event);
		CFRelease(event);
	}
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
	"unsafe"
)

type darwinPlatform struct{}

func newPlatform() computerPlatform {
	return &darwinPlatform{}
}

func (d *darwinPlatform) Screenshot(ctx context.Context) (*ScreenImage, error) {
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	var w, h C.int
	cfData := C.captureScreenshot(&w, &h)
	if cfData == nil {
		return nil, fmt.Errorf("screen capture failed — check screen recording permissions")
	}
	defer C.CFRelease(C.CFTypeRef(cfData))

	width, height := int(w), int(h)
	rawLen := width * height * 4

	// Copy raw RGBA data from CFData
	rawBytes := C.GoBytes(unsafe.Pointer(C.CFDataGetBytePtr(cfData)), C.CFIndex(rawLen))

	// Encode as PNG
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
		return nil, fmt.Errorf("failed to encode screenshot: %w", err)
	}

	return &ScreenImage{
		Pixels: buf.Bytes(),
		Width:  width,
		Height: height,
	}, nil
}

func (d *darwinPlatform) Click(ctx context.Context, x, y int) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}
	C.clickMouse(C.int(x), C.int(y))
	return nil
}

func (d *darwinPlatform) DoubleClick(ctx context.Context, x, y int) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}
	C.doubleClickMouse(C.int(x), C.int(y))
	return nil
}

func (d *darwinPlatform) Move(ctx context.Context, x, y int) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}
	C.moveMouse(C.int(x), C.int(y))
	return nil
}

func (d *darwinPlatform) Type(ctx context.Context, text string) error {
	for _, r := range text {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
		char := C.uint32_t(r)
		C.typeUnicodeString(&char, 1)
	}
	return nil
}

func (d *darwinPlatform) Key(ctx context.Context, key string) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	vk, ok := darwinKeyMap(strings.ToUpper(strings.TrimSpace(key)))
	if !ok {
		return fmt.Errorf("unknown key name: %s", key)
	}

	CGKeyCodeDown := C.CGEventCreateKeyboardEvent(nil, C.CGKeyCode(vk), true)
	defer C.CFRelease(C.CFTypeRef(CGKeyCodeDown))
	C.CGEventPost(C.kCGHIDEventTap, CGKeyCodeDown)

	CGKeyCodeUp := C.CGEventCreateKeyboardEvent(nil, C.CGKeyCode(vk), false)
	defer C.CFRelease(C.CFTypeRef(CGKeyCodeUp))
	C.CGEventPost(C.kCGHIDEventTap, CGKeyCodeUp)

	return nil
}

func (d *darwinPlatform) Scroll(ctx context.Context, dx, dy int) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}
	C.scrollMouse(C.int(dx), C.int(dy))
	return nil
}

// darwinKeyMap maps key name strings to macOS virtual key codes.
func darwinKeyMap(name string) (int, bool) {
	// Handle modifier combos like "CTRL+C"
	parts := strings.Split(name, "+")
	if len(parts) > 1 {
		// For now, just handle the last key in the combo
		name = strings.TrimSpace(parts[len(parts)-1])
	}

	switch name {
	case "A":
		return 0x00, true
	case "S":
		return 0x01, true
	case "D":
		return 0x02, true
	case "F":
		return 0x03, true
	case "H":
		return 0x04, true
	case "G":
		return 0x05, true
	case "Z":
		return 0x06, true
	case "X":
		return 0x07, true
	case "C":
		return 0x08, true
	case "V":
		return 0x09, true
	case "B":
		return 0x0B, true
	case "Q":
		return 0x0C, true
	case "W":
		return 0x0D, true
	case "E":
		return 0x0E, true
	case "R":
		return 0x0F, true
	case "Y":
		return 0x10, true
	case "T":
		return 0x11, true
	case "1":
		return 0x12, true
	case "2":
		return 0x13, true
	case "3":
		return 0x14, true
	case "4":
		return 0x15, true
	case "6":
		return 0x16, true
	case "5":
		return 0x17, true
	case "7":
		return 0x1A, true
	case "8":
		return 0x1C, true
	case "9":
		return 0x19, true
	case "0":
		return 0x1D, true
	case "RETURN", "ENTER":
		return 0x24, true
	case "TAB":
		return 0x30, true
	case "SPACE":
		return 0x31, true
	case "DELETE", "BACKSPACE":
		return 0x33, true
	case "ESCAPE", "ESC":
		return 0x35, true
	case "COMMAND", "CMD":
		return 0x37, true
	case "SHIFT":
		return 0x38, true
	case "OPTION", "ALT":
		return 0x3A, true
	case "CONTROL", "CTRL":
		return 0x3B, true
	case "CAPSLOCK":
		return 0x39, true
	case "F1":
		return 0x7A, true
	case "F2":
		return 0x78, true
	case "F3":
		return 0x63, true
	case "F4":
		return 0x76, true
	case "F5":
		return 0x60, true
	case "F6":
		return 0x61, true
	case "F7":
		return 0x62, true
	case "F8":
		return 0x64, true
	case "F9":
		return 0x65, true
	case "F10":
		return 0x6D, true
	case "F11":
		return 0x67, true
	case "F12":
		return 0x6F, true
	case "LEFT":
		return 0x7B, true
	case "RIGHT":
		return 0x7C, true
	case "DOWN":
		return 0x7D, true
	case "UP":
		return 0x7E, true
	case "PAGEUP":
		return 0x74, true
	case "PAGEDOWN":
		return 0x79, true
	case "HOME":
		return 0x73, true
	case "END":
		return 0x77, true
	case "INSERT":
		return 0x72, true
	case "DELETE":
		return 0x75, true
	case "PRINTSCREEN", "PRTSC":
		return 0x69, true
	case "NUMLOCK":
		return 0x47, true
	default:
		return 0, false
	}
}

// Ensure compliance with interface at compile time
var _ computerPlatform = (*darwinPlatform)(nil)

