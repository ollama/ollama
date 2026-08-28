//go:build windows

package tools

import (
	"bytes"
	"context"
	"fmt"
	"image"
	"image/png"
	"math"
	"strings"
	"unicode/utf8"
	"unsafe"

	"golang.org/x/sys/windows"
)

var (
	user32 = windows.NewLazySystemDLL("user32.dll")
	gdi32  = windows.NewLazySystemDLL("gdi32.dll")
	pSendInput          = user32.NewProc("SendInput")
	pGetSystemMetrics   = user32.NewProc("GetSystemMetrics")
	pGetDesktopWindow   = user32.NewProc("GetDesktopWindow")
	pGetDC              = user32.NewProc("GetDC")
	pReleaseDC          = user32.NewProc("ReleaseDC")
	pCreateCompatibleDC = gdi32.NewProc("CreateCompatibleDC")
	pCreateCompatibleBitmap = gdi32.NewProc("CreateCompatibleBitmap")
	pSelectObject       = gdi32.NewProc("SelectObject")
	pBitBlt             = gdi32.NewProc("BitBlt")
	pDeleteObject       = gdi32.NewProc("DeleteObject")
	pDeleteDC           = gdi32.NewProc("DeleteDC")
	pGetDIBits          = gdi32.NewProc("GetDIBits")
)

const (
	SM_CXSCREEN = 0
	SM_CYSCREEN = 1
	SRCCOPY     = 0x00CC0020

	INPUT_MOUSE    = 0
	INPUT_KEYBOARD = 1

	MOUSEEVENTF_MOVE      = 0x0001
	MOUSEEVENTF_LEFTDOWN  = 0x0002
	MOUSEEVENTF_LEFTUP    = 0x0004
	MOUSEEVENTF_ABSOLUTE  = 0x8000
	MOUSEEVENTF_WHEEL     = 0x0800
	MOUSEEVENTF_HWHEEL    = 0x1000

	KEYEVENTF_KEYUP    = 0x0002
	KEYEVENTF_EXTENDEDKEY = 0x0001

	VK_SHIFT    = 0x10
	VK_CONTROL  = 0x11
	VK_MENU     = 0x12
	VK_LSHIFT   = 0xA0
	VK_RSHIFT   = 0xA1
	VK_LCONTROL = 0xA2
	VK_RCONTROL = 0xA3
	VK_LMENU    = 0xA4
	VK_RMENU    = 0xA5
)

type mouseInput struct {
	Type       uint32
	Mi         mouseInputData
	_          [8]byte // padding for alignment on 64-bit
}

type mouseInputData struct {
	Dx          int32
	Dy          int32
	MouseData   uint32
	DwFlags     uint32
	Time        uint32
	DwExtraInfo uintptr
}

type keyboardInput struct {
	Type uint32
	Ki   keyboardInputData
	_    [8]byte // padding for alignment on 64-bit
}

type keyboardInputData struct {
	WVk         uint16
	_           uint16 // padding
	DwScanCode  uint32
	DwFlags     uint32
	Time        uint32
	DwExtraInfo uintptr
}

type bitmapInfoHeader struct {
	BiSize          uint32
	BiWidth         int32
	BiHeight        int32
	BiPlanes        uint16
	BiBitCount      uint16
	BiCompression   uint32
	BiSizeImage     uint32
	BiXPelsPerMeter int32
	BiYPelsPerMeter int32
	BiClrUsed       uint32
	BiClrImportant  uint32
}

type fakeComputerPlatform struct{}

func newPlatform() computerPlatform {
	return &fakeComputerPlatform{}
}

func (f *fakeComputerPlatform) Screenshot(ctx context.Context) (*ScreenImage, error) {
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}

	screenW, _, _ := pGetSystemMetrics.Call(SM_CXSCREEN)
	screenH, _, _ := pGetSystemMetrics.Call(SM_CYSCREEN)
	if screenW == 0 || screenH == 0 {
		return nil, fmt.Errorf("could not determine screen dimensions")
	}
	w, h := int(screenW), int(screenH)

	hWnd, _, _ := pGetDesktopWindow.Call()
	hDC, _, _ := pGetDC.Call(hWnd)
	if hDC == 0 {
		return nil, fmt.Errorf("failed to get desktop device context")
	}
	defer pReleaseDC.Call(hWnd, hDC)

	memDC, _, _ := pCreateCompatibleDC.Call(hDC)
	if memDC == 0 {
		return nil, fmt.Errorf("failed to create compatible device context")
	}
	defer pDeleteDC.Call(memDC)

	hBitmap, _, _ := pCreateCompatibleBitmap.Call(hDC, uintptr(w), uintptr(h))
	if hBitmap == 0 {
		return nil, fmt.Errorf("failed to create compatible bitmap")
	}
	defer pDeleteObject.Call(hBitmap)

	oldBmp, _, _ := pSelectObject.Call(memDC, hBitmap)
	defer pSelectObject.Call(memDC, oldBmp)

	ret, _, _ := pBitBlt.Call(memDC, 0, 0, uintptr(w), uintptr(h), hDC, 0, 0, SRCCOPY)
	if ret == 0 {
		return nil, fmt.Errorf("BitBlt failed")
	}

	bmi := bitmapInfoHeader{
		BiSize:        uint32(unsafe.Sizeof(bitmapInfoHeader{})),
		BiWidth:       int32(w),
		BiHeight:      -int32(h), // negative = top-down
		BiPlanes:      1,
		BiBitCount:    32,
		BiCompression: 0,
	}
	bufSize := w * h * 4
	pixels := make([]byte, bufSize)
	n, _, _ := pGetDIBits.Call(memDC, hBitmap, 0, uintptr(h), uintptr(unsafe.Pointer(&pixels[0])), uintptr(unsafe.Pointer(&bmi)), 0)
	if n == 0 {
		return nil, fmt.Errorf("GetDIBits failed")
	}

	// Convert BGRA to RGBA and flip bottom-up to top-down.
	rgba := make([]byte, bufSize)
	for y := 0; y < h; y++ {
		srcRow := pixels[y*w*4 : (y+1)*w*4]
		dstRow := rgba[y*w*4 : (y+1)*w*4]
		for x := 0; x < w; x++ {
			srcOff := x * 4
			dstOff := x * 4
			dstRow[dstOff+0] = srcRow[srcOff+2] // R <- B
			dstRow[dstOff+1] = srcRow[srcOff+1] // G <- G
			dstRow[dstOff+2] = srcRow[srcOff+0] // B <- R
			dstRow[dstOff+3] = 0xFF              // A
		}
	}

	// Encode to PNG.
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	copy(img.Pix, rgba)

	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		return nil, fmt.Errorf("failed to encode screenshot: %w", err)
	}

	return &ScreenImage{
		Pixels: buf.Bytes(),
		Width:  w,
		Height: h,
	}, nil
}

func (f *fakeComputerPlatform) Click(ctx context.Context, x, y int) error {
	return f.mouseClick(ctx, x, y, 1)
}

func (f *fakeComputerPlatform) DoubleClick(ctx context.Context, x, y int) error {
	return f.mouseClick(ctx, x, y, 2)
}

func (f *fakeComputerPlatform) mouseClick(ctx context.Context, x, y int, count int) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	screenW, _, _ := pGetSystemMetrics.Call(SM_CXSCREEN)
	screenH, _, _ := pGetSystemMetrics.Call(SM_CYSCREEN)

	absX, absY := toAbsoluteCoords(x, y, int(screenW), int(screenH))

	for i := 0; i < count; i++ {
		down := mouseInput{
			Type: INPUT_MOUSE,
			Mi: mouseInputData{
				Dx:      absX,
				Dy:      absY,
				DwFlags: MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_LEFTDOWN,
			},
		}
		up := mouseInput{
			Type: INPUT_MOUSE,
			Mi: mouseInputData{
				Dx:      absX,
				Dy:      absY,
				DwFlags: MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_LEFTUP,
			},
		}
		pSendInput.Call(1, uintptr(unsafe.Pointer(&down)), unsafe.Sizeof(down))
		pSendInput.Call(1, uintptr(unsafe.Pointer(&up)), unsafe.Sizeof(up))
	}
	return nil
}

func (f *fakeComputerPlatform) Move(ctx context.Context, x, y int) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	screenW, _, _ := pGetSystemMetrics.Call(SM_CXSCREEN)
	screenH, _, _ := pGetSystemMetrics.Call(SM_CYSCREEN)

	absX, absY := toAbsoluteCoords(x, y, int(screenW), int(screenH))

	mi := mouseInput{
		Type: INPUT_MOUSE,
		Mi: mouseInputData{
			Dx:      absX,
			Dy:      absY,
			DwFlags: MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE,
		},
	}
	pSendInput.Call(1, uintptr(unsafe.Pointer(&mi)), unsafe.Sizeof(mi))
	return nil
}

func (f *fakeComputerPlatform) Type(ctx context.Context, text string) error {
	for _, r := range text {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
		if err := typeRune(r); err != nil {
			return err
		}
	}
	return nil
}

func (f *fakeComputerPlatform) Key(ctx context.Context, key string) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	upper := strings.ToUpper(strings.TrimSpace(key))
	parts := strings.Split(upper, "+")
	if len(parts) == 0 {
		return fmt.Errorf("invalid key specification: %s", key)
	}

	// Identify modifiers
	var mods []uint16
	for _, part := range parts[:len(parts)-1] {
		switch strings.TrimSpace(part) {
		case "CTRL", "CONTROL":
			mods = append(mods, VK_LCONTROL)
		case "ALT":
			mods = append(mods, VK_LMENU)
		case "SHIFT":
			mods = append(mods, VK_LSHIFT)
		default:
			return fmt.Errorf("unknown modifier: %s", part)
		}
	}

	// Press modifiers
	for _, m := range mods {
		pressKey(m, 0)
	}

	// Press and release the main key
	mainKey := strings.TrimSpace(parts[len(parts)-1])
	vk, ext, err := mapKeyName(mainKey)
	if err != nil {
		// Release modifiers on error
		for i := len(mods) - 1; i >= 0; i-- {
			pressKey(mods[i], KEYEVENTF_KEYUP)
		}
		return err
	}
	pressKey(vk, ext)

	// Release modifiers in reverse order
	for i := len(mods) - 1; i >= 0; i-- {
		pressKey(mods[i], KEYEVENTF_KEYUP)
	}
	return nil
}

func (f *fakeComputerPlatform) Scroll(ctx context.Context, dx, dy int) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	// Vertical scroll: WHEEL_DELTA = 120
	if dy != 0 {
		steps := int32(float64(dy) * 120.0 / 3.0)
		mi := mouseInput{
			Type: INPUT_MOUSE,
			Mi: mouseInputData{
				DwFlags: MOUSEEVENTF_WHEEL,
				MouseData: uint32(steps),
			},
		}
		pSendInput.Call(1, uintptr(unsafe.Pointer(&mi)), unsafe.Sizeof(mi))
	}
	// Horizontal scroll
	if dx != 0 {
		steps := int32(float64(dx) * 120.0 / 3.0)
		mi := mouseInput{
			Type: INPUT_MOUSE,
			Mi: mouseInputData{
				DwFlags: MOUSEEVENTF_HWHEEL,
				MouseData: uint32(steps),
			},
		}
		pSendInput.Call(1, uintptr(unsafe.Pointer(&mi)), unsafe.Sizeof(mi))
	}
	return nil
}

// --- helpers ---

// toAbsoluteCoords converts screenshot-space coordinates to the 0..65535
// absolute coordinate space used by SendInput.
func toAbsoluteCoords(x, y, screenW, screenH int) (int32, int32) {
	if screenW <= 1 {
		screenW = 1
	}
	if screenH <= 1 {
		screenH = 1
	}
	absX := int32(math.Round(float64(x) * 65535.0 / float64(screenW-1)))
	absY := int32(math.Round(float64(y) * 65535.0 / float64(screenH-1)))
	if absX < 0 {
		absX = 0
	}
	if absX > 65535 {
		absX = 65535
	}
	if absY < 0 {
		absY = 0
	}
	if absY > 65535 {
		absY = 65535
	}
	return absX, absY
}

// typeRune types a single Unicode rune using key-down/key-up events and
// UnicodePacket for characters outside the basic ASCII range.
func typeRune(r rune) error {
	if r <= 0x7F {
		// Map ASCII to virtual key codes.
		vk := asciiToVK(byte(r))
		if vk == 0 {
			return fmt.Errorf("unsupported ASCII character: %c", r)
		}
		return pressKey(vk, 0)
	}
	// For non-ASCII characters, use Unicode input.
	return typeUnicode(r)
}

func asciiToVK(ch byte) uint16 {
	switch {
	case ch >= 'a' && ch <= 'z':
		return uint16(ch - 'a' + 'A')
	case ch >= 'A' && ch <= 'Z':
		return uint16(ch)
	case ch >= '0' && ch <= '9':
		return uint16(ch)
	case ch == ' ':
		return 0x20 // VK_SPACE
	case ch == '\t':
		return 0x09 // VK_TAB
	case ch == '\n':
		return 0x0D // VK_RETURN
	case ch == '\r':
		return 0
	case ch == '.':
		return 0xBE
	case ch == ',':
		return 0xBC
	case ch == '/':
		return 0xBF
	case ch == '\\':
		return 0xDC
	case ch == ';':
		return 0xBA
	case ch == '\'':
		return 0xDE
	case ch == '[':
		return 0xDB
	case ch == ']':
		return 0xDD
	case ch == '-':
		return 0xBD
	case ch == '=':
		return 0xBB
	case ch == '`':
		return 0xC0
	default:
		return 0
	}
}

func typeUnicode(r rune) error {
	var buf [4]byte
	n := utf8.EncodeRune(buf[:], r)

	for i := 0; i < n; i++ {
		k := keyboardInput{
			Type: INPUT_KEYBOARD,
			Ki: keyboardInputData{
				DwScanCode: uint32(buf[i]),
				DwFlags:    KEYEVENTF_EXTENDEDKEY,
			},
		}
		pSendInput.Call(1, uintptr(unsafe.Pointer(&k)), unsafe.Sizeof(k))

		kUp := keyboardInput{
			Type: INPUT_KEYBOARD,
			Ki: keyboardInputData{
				DwScanCode: uint32(buf[i]),
				DwFlags:    KEYEVENTF_KEYUP | KEYEVENTF_EXTENDEDKEY,
			},
		}
		pSendInput.Call(1, uintptr(unsafe.Pointer(&kUp)), unsafe.Sizeof(kUp))
	}
	return nil
}

func pressKey(vk uint16, flags uint32) error {
	kDown := keyboardInput{
		Type: INPUT_KEYBOARD,
		Ki: keyboardInputData{
			WVk:     vk,
			DwFlags: flags,
		},
	}
	kUp := keyboardInput{
		Type: INPUT_KEYBOARD,
		Ki: keyboardInputData{
			WVk:     vk,
			DwFlags: KEYEVENTF_KEYUP | flags,
		},
	}
	pSendInput.Call(1, uintptr(unsafe.Pointer(&kDown)), unsafe.Sizeof(kDown))
	pSendInput.Call(1, uintptr(unsafe.Pointer(&kUp)), unsafe.Sizeof(kUp))
	return nil
}

// parseKey parses a key specification string like "CTRL+C", "ENTER",
// "ALT+TAB", "SHIFT+A" into a virtual key code. Returns the VK and
// whether it's an extended key.
// parseKey extracts the main key name from a key specification.
// Modifier handling is done in the Key() method.
func parseKey(key string) (mainKeyName string, err error) {
	upper := strings.ToUpper(strings.TrimSpace(key))
	parts := strings.Split(upper, "+")
	if len(parts) == 0 {
		return "", fmt.Errorf("invalid key specification: %s", key)
	}
	return strings.TrimSpace(parts[len(parts)-1]), nil
}



func mapKeyName(name string) (vk uint16, extended bool, err error) {
	switch name {
	case "ENTER", "RETURN":
		return 0x0D, false, nil
	case "TAB":
		return 0x09, false, nil
	case "ESCAPE", "ESC":
		return 0x1B, false, nil
	case "BACKSPACE", "BACK":
		return 0x08, false, nil
	case "DELETE", "DEL":
		return 0x2E, true, nil
	case "INSERT", "INS":
		return 0x2D, true, nil
	case "HOME":
		return 0x24, true, nil
	case "END":
		return 0x23, true, nil
	case "PAGEUP", "PAGE_UP":
		return 0x21, true, nil
	case "PAGEDOWN", "PAGE_DOWN":
		return 0x22, true, nil
	case "UP":
		return 0x26, true, nil
	case "DOWN":
		return 0x28, true, nil
	case "LEFT":
		return 0x25, true, nil
	case "RIGHT":
		return 0x27, true, nil
	case "SPACE":
		return 0x20, false, nil
	case "CAPSLOCK":
		return 0x14, false, nil
	case "NUMLOCK":
		return 0x90, false, nil
	case "F1":
		return 0x70, false, nil
	case "F2":
		return 0x71, false, nil
	case "F3":
		return 0x72, false, nil
	case "F4":
		return 0x73, false, nil
	case "F5":
		return 0x74, false, nil
	case "F6":
		return 0x75, false, nil
	case "F7":
		return 0x76, false, nil
	case "F8":
		return 0x77, false, nil
	case "F9":
		return 0x78, false, nil
	case "F10":
		return 0x79, false, nil
	case "F11":
		return 0x7A, false, nil
	case "F12":
		return 0x7B, false, nil
	case "SHIFT":
		return VK_LSHIFT, false, nil
	case "CTRL", "CONTROL":
		return VK_LCONTROL, false, nil
	case "ALT":
		return VK_LMENU, false, nil
	case "PRINTSCREEN", "PRTSC":
		return 0x2C, true, nil
	case "WINDOWS", "WIN":
		return 0x5B, true, nil
	}

	// Single character keys
	if len(name) == 1 {
		ch := name[0]
		if ch >= 'A' && ch <= 'Z' {
			return uint16(ch), false, nil
		}
		if ch >= '0' && ch <= '9' {
			return uint16(ch), false, nil
		}
	}

	// Numpad keys
	if strings.HasPrefix(name, "NUMPAD") {
		numpad := strings.TrimPrefix(name, "NUMPAD")
		if len(numpad) == 1 && numpad[0] >= '0' && numpad[0] <= '9' {
			return uint16(0x60 + numpad[0] - '0'), false, nil
		}
	}

	return 0, false, fmt.Errorf("unknown key name: %s", name)
}
