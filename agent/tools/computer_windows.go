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

	"github.com/ollama/ollama/agent"
	"golang.org/x/sys/windows"
)

var (
	user32 = windows.NewLazySystemDLL("user32.dll")
	gdi32  = windows.NewLazySystemDLL("gdi32.dll")

	pSendInput             = user32.NewProc("SendInput")
	pGetSystemMetrics      = user32.NewProc("GetSystemMetrics")
	pGetDesktopWindow      = user32.NewProc("GetDesktopWindow")
	pGetDC                 = user32.NewProc("GetDC")
	pReleaseDC             = user32.NewProc("ReleaseDC")
	pCreateCompatibleDC    = gdi32.NewProc("CreateCompatibleDC")
	pCreateCompatibleBitmap = gdi32.NewProc("CreateCompatibleBitmap")
	pSelectObject          = gdi32.NewProc("SelectObject")
	pBitBlt                = gdi32.NewProc("BitBlt")
	pDeleteObject          = gdi32.NewProc("DeleteObject")
	pDeleteDC              = gdi32.NewProc("DeleteDC")
	pGetDIBits             = gdi32.NewProc("GetDIBits")
)

const (
	SM_CXSCREEN = 0
	SM_CYSCREEN = 1
	SRCCOPY     = 0x00CC0020

	INPUT_MOUSE    = 0
	INPUT_KEYBOARD = 1

	MOUSEEVENTF_MOVE     = 0x0001
	MOUSEEVENTF_LEFTDOWN = 0x0002
	MOUSEEVENTF_LEFTUP   = 0x0004
	MOUSEEVENTF_ABSOLUTE = 0x8000
	MOUSEEVENTF_WHEEL    = 0x0800
	MOUSEEVENTF_HWHEEL   = 0x1000

	KEYEVENTF_KEYUP      = 0x0002
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

// INPUT structure layout for 64-bit Windows:
//
//	typedef struct tagINPUT {
//	    DWORD type;           // 4 bytes at offset 0
//	    DWORD padding;        // 4 bytes implicit (union aligns to 8)
//	    union {
//	        MOUSEINPUT mi;    // 32 bytes at offset 8
//	        KEYBDINPUT ki;
//	        HARDWAREINPUT hi;
//	    };
//	} INPUT;                  // total: 40 bytes
//
// On 64-bit: unsafe.Sizeof(input) == 40.
// On 32-bit: unsafe.Sizeof(input) == 28.
//
// The structures below match this layout exactly.

// mouseInputData matches Windows MOUSEINPUT exactly.
//
//	typedef struct tagMOUSEINPUT {
//	    LONG dx;
//	    LONG dy;
//	    DWORD mouseData;
//	    DWORD dwFlags;
//	    DWORD time;
//	    ULONG_PTR dwExtraInfo;
//	} MOUSEINPUT;
type mouseInputData struct {
	Dx          int32
	Dy          int32
	MouseData   uint32
	DwFlags     uint32
	Time        uint32
	DwExtraInfo uintptr // 4 bytes on 32-bit, 8 bytes on 64-bit
}

// mouseInput matches the Windows INPUT structure for mouse events.
//
//	Member layout: [Type:4][pad:4][Mi:sizeof(MOUSEINPUT)]
//	On 64-bit: 4 + 4 + 32 = 40 bytes.
type mouseInput struct {
	Type uint32
	_pad [4]byte // align Mi to 8-byte boundary (matches Windows implicit padding)
	Mi   mouseInputData
}

// keyboardInputData matches Windows KEYBDINPUT exactly.
//
//	typedef struct tagKEYBDINPUT {
//	    WORD wVk;
//	    WORD wScan;
//	    DWORD dwFlags;
//	    DWORD time;
//	    ULONG_PTR dwExtraInfo;
//	} KEYBDINPUT;
type keyboardInputData struct {
	WVk         uint16
	WScan       uint16
	DwFlags     uint32
	Time        uint32
	DwExtraInfo uintptr
}

// keyboardInput matches the Windows INPUT structure for keyboard events.
type keyboardInput struct {
	Type uint32
	_pad [4]byte
	Ki   keyboardInputData
}

// Verify struct sizes at init time.
func init() {
	// Windows INPUT is 40 bytes on 64-bit, 28 bytes on 32-bit.
	// pointer.Size is 8 on 64-bit, 4 on 32-bit.
	expectedMouse := uintptr(40)
	expectedKey := uintptr(40)
	if unsafe.Sizeof(uintptr(0)) == 4 {
		expectedMouse = 28
		expectedKey = 28
	}
	if unsafe.Sizeof(mouseInput{}) != expectedMouse {
		panic(fmt.Sprintf("computer_windows: mouseInput size = %d, want %d", unsafe.Sizeof(mouseInput{}), expectedMouse))
	}
	if unsafe.Sizeof(keyboardInput{}) != expectedKey {
		panic(fmt.Sprintf("computer_windows: keyboardInput size = %d, want %d", unsafe.Sizeof(keyboardInput{}), expectedKey))
	}
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

type windowsComputerBackend struct{}

// NewComputerBackend returns a platform-specific computer backend for the
// local machine. Returns nil if the platform is not supported.
func NewComputerBackend() agent.ComputerBackend {
	return &windowsComputerBackend{}
}

func (w *windowsComputerBackend) Screenshot(ctx context.Context) ([]byte, int, int, error) {
	select {
	case <-ctx.Done():
		return nil, 0, 0, ctx.Err()
	default:
	}

	screenW, _, _ := pGetSystemMetrics.Call(SM_CXSCREEN)
	screenH, _, _ := pGetSystemMetrics.Call(SM_CYSCREEN)
	if screenW == 0 || screenH == 0 {
		return nil, 0, 0, fmt.Errorf("could not determine screen dimensions")
	}
	w, h := int(screenW), int(screenH)

	hWnd, _, _ := pGetDesktopWindow.Call()
	hDC, _, _ := pGetDC.Call(hWnd)
	if hDC == 0 {
		return nil, 0, 0, fmt.Errorf("failed to get desktop device context")
	}
	defer pReleaseDC.Call(hWnd, hDC)

	memDC, _, _ := pCreateCompatibleDC.Call(hDC)
	if memDC == 0 {
		return nil, 0, 0, fmt.Errorf("failed to create compatible device context")
	}
	defer pDeleteDC.Call(memDC)

	hBitmap, _, _ := pCreateCompatibleBitmap.Call(hDC, uintptr(w), uintptr(h))
	if hBitmap == 0 {
		return nil, 0, 0, fmt.Errorf("failed to create compatible bitmap")
	}
	defer pDeleteObject.Call(hBitmap)

	// Select bitmap into DC for capture
	oldBmp, _, _ := pSelectObject.Call(memDC, hBitmap)

	ret, _, _ := pBitBlt.Call(memDC, 0, 0, uintptr(w), uintptr(h), hDC, 0, 0, SRCCOPY)
	if ret == 0 {
		pSelectObject.Call(memDC, oldBmp)
		return nil, 0, 0, fmt.Errorf("BitBlt failed")
	}

	// CRITICAL: Restore old bitmap BEFORE calling GetDIBits.
	// GetDIBits requires the bitmap to NOT be selected into a DC.
	pSelectObject.Call(memDC, oldBmp)

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
		return nil, 0, 0, fmt.Errorf("GetDIBits failed")
	}

	// Convert BGRA to RGBA
	rgba := make([]byte, bufSize)
	for y := 0; y < h; y++ {
		srcRow := pixels[y*w*4 : (y+1)*w*4]
		dstRow := rgba[y*w*4 : (y+1)*w*4]
		for x := 0; x < w; x++ {
			so := x * 4
			do := x * 4
			dstRow[do+0] = srcRow[so+2] // R <- B
			dstRow[do+1] = srcRow[so+1] // G <- G
			dstRow[do+2] = srcRow[so+0] // B <- R
			dstRow[do+3] = 0xFF         // A
		}
	}

	// Encode to PNG
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	copy(img.Pix, rgba)

	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		return nil, 0, 0, fmt.Errorf("failed to encode screenshot: %w", err)
	}

	return buf.Bytes(), w, h, nil
}

func (w *windowsComputerBackend) Click(ctx context.Context, x, y int) error {
	return w.mouseClick(ctx, x, y, 1)
}

func (w *windowsComputerBackend) DoubleClick(ctx context.Context, x, y int) error {
	return w.mouseClick(ctx, x, y, 2)
}

func (w *windowsComputerBackend) mouseClick(ctx context.Context, x, y int, count int) error {
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

func (w *windowsComputerBackend) Move(ctx context.Context, x, y int) error {
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

func (w *windowsComputerBackend) Type(ctx context.Context, text string) error {
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

func (w *windowsComputerBackend) Key(ctx context.Context, key string) error {
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

func (w *windowsComputerBackend) Scroll(ctx context.Context, dx, dy int) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	// Vertical scroll: WHEEL_DELTA = 120
	if dy != 0 {
		steps := int32(math.Round(float64(dy) * 120.0 / 3.0))
		mi := mouseInput{
			Type: INPUT_MOUSE,
			Mi: mouseInputData{
				DwFlags:  MOUSEEVENTF_WHEEL,
				MouseData: uint32(steps),
			},
		}
		pSendInput.Call(1, uintptr(unsafe.Pointer(&mi)), unsafe.Sizeof(mi))
	}
	// Horizontal scroll
	if dx != 0 {
		steps := int32(math.Round(float64(dx) * 120.0 / 3.0))
		mi := mouseInput{
			Type: INPUT_MOUSE,
			Mi: mouseInputData{
				DwFlags:  MOUSEEVENTF_HWHEEL,
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

func typeRune(r rune) error {
	if r <= 0x7F {
		vk := asciiToVK(byte(r))
		if vk != 0 {
			return pressKey(vk, 0)
		}
		// Fall back to Unicode input for unmapped ASCII characters
		// (e.g. !, @, #, $, %, ^, &, *, (, ), <, >, ?, {, }, |, \", ~)
	}
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
		return 0x20
	case ch == '\t':
		return 0x09
	case ch == '\n':
		return 0x0D
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
				WScan: uint16(buf[i]),
				DwFlags: KEYEVENTF_EXTENDEDKEY,
			},
		}
		pSendInput.Call(1, uintptr(unsafe.Pointer(&k)), unsafe.Sizeof(k))

		kUp := keyboardInput{
			Type: INPUT_KEYBOARD,
			Ki: keyboardInputData{
				WScan: uint16(buf[i]),
				DwFlags: KEYEVENTF_KEYUP | KEYEVENTF_EXTENDEDKEY,
			},
		}
		pSendInput.Call(1, uintptr(unsafe.Pointer(&kUp)), unsafe.Sizeof(kUp))
	}
	return nil
}

func pressKey(vk uint16, flags uint32) error {
	// Determine if this is a key-up-only call (used for modifier release).
	releaseOnly := (flags & KEYEVENTF_KEYUP) != 0
	extraFlags := flags & ^uint32(KEYEVENTF_KEYUP)

	if !releaseOnly {
		kDown := keyboardInput{
			Type: INPUT_KEYBOARD,
			Ki: keyboardInputData{
				WVk:     vk,
				DwFlags: extraFlags,
			},
		}
		pSendInput.Call(1, uintptr(unsafe.Pointer(&kDown)), unsafe.Sizeof(kDown))
	}

	kUp := keyboardInput{
		Type: INPUT_KEYBOARD,
		Ki: keyboardInputData{
			WVk:     vk,
			DwFlags: KEYEVENTF_KEYUP | extraFlags,
		},
	}
	pSendInput.Call(1, uintptr(unsafe.Pointer(&kUp)), unsafe.Sizeof(kUp))
	return nil
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

	if len(name) == 1 {
		ch := name[0]
		if ch >= 'A' && ch <= 'Z' {
			return uint16(ch), false, nil
		}
		if ch >= '0' && ch <= '9' {
			return uint16(ch), false, nil
		}
	}

	if strings.HasPrefix(name, "NUMPAD") {
		numpad := strings.TrimPrefix(name, "NUMPAD")
		if len(numpad) == 1 && numpad[0] >= '0' && numpad[0] <= '9' {
			return uint16(0x60 + numpad[0] - '0'), false, nil
		}
	}

	return 0, false, fmt.Errorf("unknown key name: %s", name)
}

// Ensure compile-time interface compliance.
var _ agent.ComputerBackend = (*windowsComputerBackend)(nil)
