package readline

import "strconv"

const (
	CharNull      = 0
	CharLineStart = 1
	CharBackward  = 2
	CharInterrupt = 3
	CharDelete    = 4
	CharLineEnd   = 5
	CharForward   = 6
	CharBell      = 7
	CharCtrlH     = 8
	CharTab       = 9
	CharCtrlJ     = 10
	CharKill      = 11
	CharCtrlL     = 12
	CharEnter     = 13
	CharNext      = 14
	CharPrev      = 16
	CharBckSearch = 18
	CharFwdSearch = 19
	CharTranspose = 20
	CharCtrlU     = 21
	CharCtrlW     = 23
	CharCtrlY     = 25
	CharCtrlZ     = 26
	CharEsc       = 27
	CharSpace     = 32
	CharEscapeEx  = 91
	CharBackspace = 127
)

const (
	KeyDel    = 51
	KeyUp     = 65
	KeyDown   = 66
	KeyRight  = 67
	KeyLeft   = 68
	MetaEnd   = 70
	MetaStart = 72
)

const (
	Esc = "\x1b"

	CursorSave    = Esc + "[s"
	CursorRestore = Esc + "[u"

	CursorEOL  = Esc + "[E"
	CursorBOL  = Esc + "[1G"
	CursorHide = Esc + "[?25l"
	CursorShow = Esc + "[?25h"

	ClearToEOL  = Esc + "[K"
	ClearLine   = Esc + "[2K"
	ClearScreen = Esc + "[2J"
	CursorReset = Esc + "[0;0f"

	ColorGrey    = Esc + "[38;5;245m"
	ColorDefault = Esc + "[0m"

	ColorBold = Esc + "[1m"

	StartBracketedPaste = Esc + "[?2004h"
	EndBracketedPaste   = Esc + "[?2004l"
)

func CursorUpN(n int) string {
	return Esc + "[" + strconv.Itoa(n) + "A"
}

func CursorDownN(n int) string {
	return Esc + "[" + strconv.Itoa(n) + "B"
}

func CursorRightN(n int) string {
	return Esc + "[" + strconv.Itoa(n) + "C"
}

func CursorLeftN(n int) string {
	return Esc + "[" + strconv.Itoa(n) + "D"
}

var (
	CursorUp    = CursorUpN(1)
	CursorDown  = CursorDownN(1)
	CursorRight = CursorRightN(1)
	CursorLeft  = CursorLeftN(1)
)

const (
	CharBracketedPaste      = 50
	CharBracketedPasteStart = "00~"
	CharBracketedPasteEnd   = "01~"
)
