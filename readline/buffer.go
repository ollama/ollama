package readline

import (
	"fmt"
	"os"

	"github.com/emirpasic/gods/lists/arraylist"
	"github.com/mattn/go-runewidth"
	"golang.org/x/term"
)

type Buffer struct {
	DisplayPos int
	Pos        int
	Buf        *arraylist.List
	LineFlags  *arraylist.List
	Prompt     *Prompt
	LineWidth  int
	Width      int
	Height     int
}

func NewBuffer(prompt *Prompt) (*Buffer, error) {
	fd := int(os.Stdout.Fd())
	width, height := 80, 24
	if termWidth, termHeight, err := term.GetSize(fd); err == nil {
		width, height = termWidth, termHeight
	}

	lwidth := width - len(prompt.prompt())

	b := &Buffer{
		DisplayPos: 0,
		Pos:        0,
		Buf:        arraylist.New(),
		LineFlags:  arraylist.New(),
		Prompt:     prompt,
		Width:      width,
		Height:     height,
		LineWidth:  lwidth,
	}

	return b, nil
}

func (b *Buffer) MoveLeft() {
	if b.Pos > 0 {
		if e, ok := b.Buf.Get(b.Pos - 1); ok {
			if r, ok := e.(rune); ok {
				rLength := runewidth.RuneWidth(r)

				if b.DisplayPos%b.LineWidth == 0 {
					fmt.Printf(CursorUp + CursorBOL + cursorRightN(b.Width))
					if rLength == 2 {
						fmt.Print(CursorLeft)
					}

					line := b.DisplayPos/b.LineWidth - 1
					cmp, _ := b.LineFlags.Get(line)
					if cmp.(bool) {
						b.DisplayPos -= 1
						fmt.Print(CursorLeft)
					}
				} else {
					fmt.Print(cursorLeftN(rLength))
				}

				b.Pos -= 1
				b.DisplayPos -= rLength
			}
		}
	}
}

func (b *Buffer) MoveLeftWord() {
	if b.Pos > 0 {
		var foundNonspace bool
		for {
			v, _ := b.Buf.Get(b.Pos - 1)
			if v == ' ' {
				if foundNonspace {
					break
				}
			} else {
				foundNonspace = true
			}
			b.MoveLeft()

			if b.Pos == 0 {
				break
			}
		}
	}
}

func (b *Buffer) MoveRight() {
	if b.Pos < b.Buf.Size() {
		if e, ok := b.Buf.Get(b.Pos); ok {
			if r, ok := e.(rune); ok {
				rLength := runewidth.RuneWidth(r)
				b.Pos += 1
				cmp, _ := b.LineFlags.Get(b.DisplayPos / b.LineWidth)
				b.DisplayPos += rLength

				if b.DisplayPos%b.LineWidth == 0 {
					fmt.Printf(CursorDown + CursorBOL + cursorRightN(len(b.Prompt.prompt())))

				} else if (b.DisplayPos-rLength)%b.LineWidth == b.LineWidth-1 && cmp != nil && cmp.(bool) {
					fmt.Printf(CursorDown + CursorBOL + cursorRightN(len(b.Prompt.prompt())+rLength))
					b.DisplayPos += 1

				} else if b.LineFlags.Size() > 0 && b.DisplayPos%b.LineWidth == b.LineWidth-1 && cmp != nil && cmp.(bool) {
					fmt.Printf(CursorDown + CursorBOL + cursorRightN(len(b.Prompt.prompt())))
					b.DisplayPos += 1

				} else {
					fmt.Print(cursorRightN(rLength))
				}
			}
		}
	}
}

func (b *Buffer) MoveRightWord() {
	if b.Pos < b.Buf.Size() {
		for {
			b.MoveRight()
			v, _ := b.Buf.Get(b.Pos)
			if v == ' ' {
				break
			}

			if b.Pos == b.Buf.Size() {
				break
			}
		}
	}
}

func (b *Buffer) MoveToStart() {
	if b.Pos > 0 {
		currLine := b.DisplayPos / b.LineWidth
		if currLine > 0 {
			for cnt := 0; cnt < currLine; cnt++ {
				fmt.Print(CursorUp)
			}
		}
		fmt.Printf(CursorBOL + cursorRightN(len(b.Prompt.prompt())))
		b.Pos = 0
		b.DisplayPos = 0
	}
}

func (b *Buffer) MoveToEnd() {
	if b.Pos < b.Buf.Size() {
		currLine := b.DisplayPos / b.LineWidth
		totalLines := b.Size() / b.LineWidth
		if currLine < totalLines {
			for cnt := 0; cnt < totalLines-currLine; cnt++ {
				fmt.Print(CursorDown)
			}
			remainder := b.Size() % b.LineWidth
			fmt.Printf(CursorBOL + cursorRightN(len(b.Prompt.prompt())+remainder))
		} else {
			fmt.Print(cursorRightN(b.Size() - b.DisplayPos))
		}

		b.Pos = b.Buf.Size()
		b.DisplayPos = b.Size()
	}
}

func (b *Buffer) Size() int {
	sum := 0
	for i := 0; i < b.Buf.Size(); i++ {
		if e, ok := b.Buf.Get(i); ok {
			if r, ok := e.(rune); ok {
				sum += runewidth.RuneWidth(r)
			}
		}
	}

	return sum
}

func (b *Buffer) Add(r rune) {
	rLength := runewidth.RuneWidth(r)

	if b.Pos == b.Buf.Size() {

		b.DisplayPos += rLength

		if b.Pos > 0 {

			if b.DisplayPos%b.LineWidth == 0 {
				fmt.Printf("%c", r)
				fmt.Printf("\n%s", b.Prompt.AltPrompt)
				b.LineFlags.Add(false)

			} else if b.DisplayPos%b.LineWidth < (b.DisplayPos-rLength)%b.LineWidth {
				fmt.Printf("\n%s", b.Prompt.AltPrompt)
				b.DisplayPos += 1
				fmt.Printf("%c", r)
				b.LineFlags.Add(true)

			} else {
				fmt.Printf("%c", r)
			}
		} else {
			fmt.Printf("%c", r)
		}

		b.Buf.Add(r)
		b.Pos += 1

		/*fmt.Printf("%c", r)
		b.Buf.Add(r)
		b.Pos += 1
		b.DisplayPos += rLength
		if b.Pos > 0 && (b.DisplayPos%b.LineWidth < (b.DisplayPos-rLength)%b.LineWidth || b.DisplayPos%b.LineWidth == 0) {
			fmt.Printf("\n%s", b.Prompt.AltPrompt)
			if b.DisplayPos%b.LineWidth == 1 {
				b.DisplayPos -= 1
			}
		}*/

		//INSERTING
	} else {
		b.DisplayPos += rLength
		if b.Pos > 0 {

			if b.DisplayPos%b.LineWidth == 0 {
				fmt.Printf("%c", r)
				fmt.Printf("\n%s", b.Prompt.AltPrompt)
				b.LineFlags.Set(b.DisplayPos/b.LineWidth-1, false)

			} else if b.DisplayPos%b.LineWidth < (b.DisplayPos-rLength)%b.LineWidth {
				fmt.Print(ClearToEOL)
				fmt.Printf("\n%s", b.Prompt.AltPrompt)
				b.DisplayPos += 1
				fmt.Printf("%c", r)
				b.LineFlags.Set(b.DisplayPos/b.LineWidth-1, true)

			} else {
				fmt.Printf("%c", r)
			}
		} else {
			fmt.Printf("%c", r)
		}
		b.Buf.Insert(b.Pos, r)
		b.Pos += 1

		/*fmt.Printf("%c", r)
		b.Buf.Insert(b.Pos, r)
		b.Pos += 1
		b.DisplayPos += rLength

		if b.Pos > 0 && b.DisplayPos%b.LineWidth < (b.DisplayPos-rLength)%b.LineWidth || b.DisplayPos%b.LineWidth == 0 {
			fmt.Printf("\n%s", b.Prompt.AltPrompt)
			if b.DisplayPos%b.LineWidth == 1 {
				b.DisplayPos -= 1
			}
		}*/

		b.drawRemaining()
	}
}

func (b *Buffer) countCurrLength(place int) int {
	var sum int
	var counter int = -1
	var prevLen int

	for place <= b.LineWidth {

		counter += 1
		sum += prevLen
		if e, ok := b.Buf.Get(b.Pos + counter); ok {
			if r, ok := e.(rune); ok {
				place += runewidth.RuneWidth(r)
				prevLen = len(string(r))
			}
		} else {
			break
		}

		if b.Pos+counter == b.Buf.Size() {
			sum += prevLen
			break
		}
	}

	return sum
}

func (b *Buffer) drawRemaining() {
	var place int
	remainingText := b.StringN(b.Pos)
	if b.Pos > 0 {
		place = b.DisplayPos % b.LineWidth
	}
	fmt.Print(CursorHide)

	// render the rest of the current line
	// issue with (b.LineWidth - place): doesn't count multi-byte characters correctly
	// solution may be to iterate to end of line and sum bytes

	currLineLength := b.countCurrLength(place)

	currLine := remainingText[:min(currLineLength, len(remainingText))]
	currLineSpace := runewidth.StringWidth(currLine)
	remLength := runewidth.StringWidth(remainingText)

	if len(currLine) > 0 {
		fmt.Printf(ClearToEOL + currLine)
		fmt.Print(cursorLeftN(currLineSpace))
	} else {
		fmt.Print(ClearToEOL)
	}

	if currLineSpace != b.LineWidth-place && currLineSpace != remLength {
		b.LineFlags.Set(b.DisplayPos/b.LineWidth, true)
	} else if currLineSpace != b.LineWidth-place {
		b.LineFlags.Remove(b.DisplayPos / b.LineWidth)
	} else {
		b.LineFlags.Set(b.DisplayPos/b.LineWidth, false)
	}

	if (b.DisplayPos+currLineSpace)%b.LineWidth == 0 && currLine == remainingText {
		fmt.Print(cursorRightN(currLineSpace))
		fmt.Printf("\n%s", b.Prompt.AltPrompt)
		fmt.Printf(CursorUp + CursorBOL + cursorRightN(b.Width-currLineSpace))
	}

	// render the other lines
	if remLength > currLineSpace {
		remaining := (remainingText[len(currLine):])
		var totalLines int
		var displayLength int
		var lineLength int = currLineSpace

		for _, c := range remaining {
			if displayLength == 0 || (displayLength+runewidth.RuneWidth(c))%b.LineWidth < displayLength%b.LineWidth {
				fmt.Printf("\n%s", b.Prompt.AltPrompt)
				totalLines += 1

				if displayLength != 0 {
					if lineLength == b.LineWidth {
						b.LineFlags.Set(b.DisplayPos/b.LineWidth+totalLines-1, false)
					} else {
						b.LineFlags.Set(b.DisplayPos/b.LineWidth+totalLines-1, true)
					}
				}

				lineLength = 0
			}

			displayLength += runewidth.RuneWidth(c)
			lineLength += runewidth.RuneWidth(c)
			fmt.Printf("%c", c)
		}
		fmt.Print(ClearToEOL)
		fmt.Print(cursorUpN(totalLines))
		fmt.Printf(CursorBOL + cursorRightN(b.Width-currLineSpace))

		if cmp, _ := b.LineFlags.Get(b.DisplayPos / b.LineWidth); cmp != nil && cmp.(bool) && b.DisplayPos%b.LineWidth != b.LineWidth-1 {
			fmt.Print(CursorLeft)
		}
	}

	fmt.Print(CursorShow)
}

func (b *Buffer) Remove() {
	if b.Buf.Size() > 0 && b.Pos > 0 {

		if e, ok := b.Buf.Get(b.Pos - 1); ok {
			if r, ok := e.(rune); ok {
				rLength := runewidth.RuneWidth(r)
				cmp, _ := b.LineFlags.Get(b.DisplayPos/b.LineWidth - 1)

				if b.DisplayPos%b.LineWidth == 0 {
					// if the user backspaces over the word boundary, do this magic to clear the line
					// and move to the end of the previous line
					fmt.Printf(CursorBOL + ClearToEOL)
					fmt.Printf(CursorUp + CursorBOL + cursorRightN(b.Width))

					if b.Size()%b.LineWidth < (b.Size()-rLength)%b.LineWidth {
						b.LineFlags.Remove(b.DisplayPos/b.LineWidth - 1)
					}

					if cmp != nil && cmp.(bool) {
						b.DisplayPos -= 1
						fmt.Print(CursorLeft)
					}

					if rLength == 2 {
						fmt.Print(CursorLeft + "  " + cursorLeftN(2))
					} else {
						fmt.Print(" " + CursorLeft)
					}

				} else if (b.DisplayPos-rLength)%b.LineWidth == 0 && cmp != nil && cmp.(bool) {
					fmt.Printf(CursorBOL + ClearToEOL)
					fmt.Printf(CursorUp + CursorBOL + cursorRightN(b.Width))

					if b.Pos == b.Buf.Size() {
						b.LineFlags.Remove(b.DisplayPos/b.LineWidth - 1)
					}
					b.DisplayPos -= 1

				} else {
					fmt.Print(cursorLeftN(rLength))
					for i := 0; i < rLength; i++ {
						fmt.Print(" ")
					}
					fmt.Print(cursorLeftN(rLength))
				}

				var eraseExtraLine bool
				if (b.Size()-1)%b.LineWidth == 0 || (rLength == 2 && ((b.Size()-2)%b.LineWidth == 0)) || b.Size()%b.LineWidth == 0 {
					eraseExtraLine = true
				}

				b.Pos -= 1
				b.DisplayPos -= rLength
				b.Buf.Remove(b.Pos)

				if b.Pos < b.Buf.Size() {
					b.drawRemaining()
					// this erases a line which is left over when backspacing in the middle of a line and there
					// are trailing characters which go over the line width boundary
					if eraseExtraLine {
						remainingLines := (b.Size() - b.DisplayPos) / b.LineWidth
						fmt.Printf(cursorDownN(remainingLines+1) + CursorBOL + ClearToEOL)
						place := b.DisplayPos % b.LineWidth
						fmt.Printf(cursorUpN(remainingLines+1) + cursorRightN(place+len(b.Prompt.prompt())))
					}
				}
			}
		}
	}
}

func (b *Buffer) Delete() {
	if b.Buf.Size() > 0 && b.Pos < b.Buf.Size() {
		b.Buf.Remove(b.Pos)
		b.drawRemaining()
		if b.Size()%b.LineWidth == 0 {
			if b.DisplayPos != b.Size() {
				remainingLines := (b.Size() - b.DisplayPos) / b.LineWidth
				fmt.Printf(cursorDownN(remainingLines) + CursorBOL + ClearToEOL)
				place := b.DisplayPos % b.LineWidth
				fmt.Printf(cursorUpN(remainingLines) + cursorRightN(place+len(b.Prompt.prompt())))
			}
		}
	}
}

func (b *Buffer) DeleteBefore() {
	if b.Pos > 0 {
		for cnt := b.Pos - 1; cnt >= 0; cnt-- {
			b.Remove()
		}
	}
}

func (b *Buffer) DeleteRemaining() {
	if b.Size() > 0 && b.Pos < b.Size() {
		charsToDel := b.Buf.Size() - b.Pos
		for cnt := 0; cnt < charsToDel; cnt++ {
			b.Delete()
		}
	}
}

func (b *Buffer) DeleteWord() {
	if b.Buf.Size() > 0 && b.Pos > 0 {
		var foundNonspace bool
		for {
			v, _ := b.Buf.Get(b.Pos - 1)
			if v == ' ' {
				if !foundNonspace {
					b.Remove()
				} else {
					break
				}
			} else {
				foundNonspace = true
				b.Remove()
			}

			if b.Pos == 0 {
				break
			}
		}
	}
}

func (b *Buffer) ClearScreen() {
	fmt.Printf(ClearScreen + CursorReset + b.Prompt.prompt())
	if b.IsEmpty() {
		ph := b.Prompt.placeholder()
		fmt.Printf(ColorGrey + ph + cursorLeftN(len(ph)) + ColorDefault)
	} else {
		currPos := b.DisplayPos
		currIndex := b.Pos
		b.Pos = 0
		b.DisplayPos = 0
		b.drawRemaining()
		fmt.Printf(CursorReset + cursorRightN(len(b.Prompt.prompt())))
		if currPos > 0 {
			targetLine := currPos / b.LineWidth
			if targetLine > 0 {
				for cnt := 0; cnt < targetLine; cnt++ {
					fmt.Print(CursorDown)
				}
			}
			remainder := currPos % b.LineWidth
			if remainder > 0 {
				fmt.Print(cursorRightN(remainder))
			}
			if currPos%b.LineWidth == 0 {
				fmt.Printf(CursorBOL + b.Prompt.AltPrompt)
			}
		}
		b.Pos = currIndex
		b.DisplayPos = currPos
	}
}

func (b *Buffer) IsEmpty() bool {
	return b.Buf.Empty()
}

func (b *Buffer) Replace(r []rune) {
	b.DisplayPos = 0
	b.Pos = 0
	lineNums := b.Size() / b.LineWidth

	b.Buf.Clear()

	fmt.Printf(CursorBOL + ClearToEOL)

	for i := 0; i < lineNums; i++ {
		fmt.Print(CursorUp + CursorBOL + ClearToEOL)
	}

	fmt.Printf(CursorBOL + b.Prompt.prompt())

	for _, c := range r {
		b.Add(c)
	}
}

func (b *Buffer) String() string {
	return b.StringN(0)
}

func (b *Buffer) StringN(n int) string {
	return b.StringNM(n, 0)
}

func (b *Buffer) StringNM(n, m int) string {
	var s string
	if m == 0 {
		m = b.Buf.Size()
	}
	for cnt := n; cnt < m; cnt++ {
		c, _ := b.Buf.Get(cnt)
		s += string(c.(rune))
	}
	return s
}

func cursorLeftN(n int) string {
	return fmt.Sprintf(CursorLeftN, n)
}

func cursorRightN(n int) string {
	return fmt.Sprintf(CursorRightN, n)
}

func cursorUpN(n int) string {
	return fmt.Sprintf(CursorUpN, n)
}

func cursorDownN(n int) string {
	return fmt.Sprintf(CursorDownN, n)
}
