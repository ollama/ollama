package readline

import (
	"fmt"
	"os"

	"github.com/emirpasic/gods/v2/lists/arraylist"
	"github.com/mattn/go-runewidth"
	"golang.org/x/term"
)

type Buffer struct {
	DisplayPos int
	Pos        int
	Buf        *arraylist.List[rune]
	// LineHasSpace is an arraylist of bools to keep track of whether a line has a space at the end
	LineHasSpace *arraylist.List[bool]
	Prompt       *Prompt
	LineWidth    int
	Width        int
	Height       int
}

func NewBuffer(prompt *Prompt) (*Buffer, error) {
	fd := int(os.Stdout.Fd())
	width, height := 80, 24
	if termWidth, termHeight, err := term.GetSize(fd); err == nil {
		width, height = termWidth, termHeight
	}

	lwidth := width - len(prompt.prompt())

	b := &Buffer{
		DisplayPos:   0,
		Pos:          0,
		Buf:          arraylist.New[rune](),
		LineHasSpace: arraylist.New[bool](),
		Prompt:       prompt,
		Width:        width,
		Height:       height,
		LineWidth:    lwidth,
	}

	return b, nil
}

func (b *Buffer) GetLineSpacing(line int) bool {
	hasSpace, _ := b.LineHasSpace.Get(line)
	return hasSpace
}

func (b *Buffer) MoveLeft() {
	if b.Pos > 0 {
		// asserts that we retrieve a rune
		if r, ok := b.Buf.Get(b.Pos - 1); ok {
			rLength := runewidth.RuneWidth(r)

			if b.DisplayPos%b.LineWidth == 0 {
				fmt.Print(CursorUp + CursorBOL + CursorRightN(b.Width))
				if rLength == 2 {
					fmt.Print(CursorLeft)
				}

				line := b.DisplayPos/b.LineWidth - 1
				hasSpace := b.GetLineSpacing(line)
				if hasSpace {
					b.DisplayPos -= 1
					fmt.Print(CursorLeft)
				}
			} else {
				fmt.Print(CursorLeftN(rLength))
			}

			b.Pos -= 1
			b.DisplayPos -= rLength
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
		if r, ok := b.Buf.Get(b.Pos); ok {
			rLength := runewidth.RuneWidth(r)
			b.Pos += 1
			hasSpace := b.GetLineSpacing(b.DisplayPos / b.LineWidth)
			b.DisplayPos += rLength

			if b.DisplayPos%b.LineWidth == 0 {
				fmt.Print(CursorDown + CursorBOL + CursorRightN(len(b.Prompt.prompt())))
			} else if (b.DisplayPos-rLength)%b.LineWidth == b.LineWidth-1 && hasSpace {
				fmt.Print(CursorDown + CursorBOL + CursorRightN(len(b.Prompt.prompt())+rLength))
				b.DisplayPos += 1
			} else if b.LineHasSpace.Size() > 0 && b.DisplayPos%b.LineWidth == b.LineWidth-1 && hasSpace {
				fmt.Print(CursorDown + CursorBOL + CursorRightN(len(b.Prompt.prompt())))
				b.DisplayPos += 1
			} else {
				fmt.Print(CursorRightN(rLength))
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
			for range currLine {
				fmt.Print(CursorUp)
			}
		}
		fmt.Print(CursorBOL + CursorRightN(len(b.Prompt.prompt())))
		b.Pos = 0
		b.DisplayPos = 0
	}
}

func (b *Buffer) MoveToEnd() {
	if b.Pos < b.Buf.Size() {
		currLine := b.DisplayPos / b.LineWidth
		totalLines := b.DisplaySize() / b.LineWidth
		if currLine < totalLines {
			for range totalLines - currLine {
				fmt.Print(CursorDown)
			}
			remainder := b.DisplaySize() % b.LineWidth
			fmt.Print(CursorBOL + CursorRightN(len(b.Prompt.prompt())+remainder))
		} else {
			fmt.Print(CursorRightN(b.DisplaySize() - b.DisplayPos))
		}

		b.Pos = b.Buf.Size()
		b.DisplayPos = b.DisplaySize()
	}
}

func (b *Buffer) DisplaySize() int {
	sum := 0
	for i := range b.Buf.Size() {
		if r, ok := b.Buf.Get(i); ok {
			sum += runewidth.RuneWidth(r)
		}
	}

	return sum
}

func (b *Buffer) Add(r rune) {
	if b.Pos == b.Buf.Size() {
		b.AddChar(r, false)
	} else {
		b.AddChar(r, true)
	}
}

func (b *Buffer) AddChar(r rune, insert bool) {
	rLength := runewidth.RuneWidth(r)
	b.DisplayPos += rLength

	if b.Pos > 0 {
		if b.DisplayPos%b.LineWidth == 0 {
			fmt.Printf("%c", r)
			fmt.Printf("\n%s", b.Prompt.AltPrompt)

			if insert {
				b.LineHasSpace.Set(b.DisplayPos/b.LineWidth-1, false)
			} else {
				b.LineHasSpace.Add(false)
			}

			// this case occurs when a double-width rune crosses the line boundary
		} else if b.DisplayPos%b.LineWidth < (b.DisplayPos-rLength)%b.LineWidth {
			if insert {
				fmt.Print(ClearToEOL)
			}
			fmt.Printf("\n%s", b.Prompt.AltPrompt)
			b.DisplayPos += 1
			fmt.Printf("%c", r)

			if insert {
				b.LineHasSpace.Set(b.DisplayPos/b.LineWidth-1, true)
			} else {
				b.LineHasSpace.Add(true)
			}
		} else {
			fmt.Printf("%c", r)
		}
	} else {
		fmt.Printf("%c", r)
	}

	if insert {
		b.Buf.Insert(b.Pos, r)
	} else {
		b.Buf.Add(r)
	}

	b.Pos += 1

	if insert {
		b.drawRemaining()
	}
}

func (b *Buffer) countRemainingLineWidth(place int) int {
	var sum int
	counter := -1
	var prevLen int

	for place <= b.LineWidth {
		counter += 1
		sum += prevLen
		if r, ok := b.Buf.Get(b.Pos + counter); ok {
			place += runewidth.RuneWidth(r)
			prevLen = len(string(r))
		} else {
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
	currLineLength := b.countRemainingLineWidth(place)

	currLine := remainingText[:min(currLineLength, len(remainingText))]
	currLineSpace := runewidth.StringWidth(currLine)
	remLength := runewidth.StringWidth(remainingText)

	if len(currLine) > 0 {
		fmt.Print(ClearToEOL + currLine + CursorLeftN(currLineSpace))
	} else {
		fmt.Print(ClearToEOL)
	}

	if currLineSpace != b.LineWidth-place && currLineSpace != remLength {
		b.LineHasSpace.Set(b.DisplayPos/b.LineWidth, true)
	} else if currLineSpace != b.LineWidth-place {
		b.LineHasSpace.Remove(b.DisplayPos / b.LineWidth)
	} else {
		b.LineHasSpace.Set(b.DisplayPos/b.LineWidth, false)
	}

	if (b.DisplayPos+currLineSpace)%b.LineWidth == 0 && currLine == remainingText {
		fmt.Print(CursorRightN(currLineSpace))
		fmt.Printf("\n%s", b.Prompt.AltPrompt)
		fmt.Print(CursorUp + CursorBOL + CursorRightN(b.Width-currLineSpace))
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
						b.LineHasSpace.Set(b.DisplayPos/b.LineWidth+totalLines-1, false)
					} else {
						b.LineHasSpace.Set(b.DisplayPos/b.LineWidth+totalLines-1, true)
					}
				}

				lineLength = 0
			}

			displayLength += runewidth.RuneWidth(c)
			lineLength += runewidth.RuneWidth(c)
			fmt.Printf("%c", c)
		}
		fmt.Print(ClearToEOL + CursorUpN(totalLines) + CursorBOL + CursorRightN(b.Width-currLineSpace))

		hasSpace := b.GetLineSpacing(b.DisplayPos / b.LineWidth)

		if hasSpace && b.DisplayPos%b.LineWidth != b.LineWidth-1 {
			fmt.Print(CursorLeft)
		}
	}

	fmt.Print(CursorShow)
}

func (b *Buffer) Remove() {
	if b.Buf.Size() > 0 && b.Pos > 0 {
		if r, ok := b.Buf.Get(b.Pos - 1); ok {
			rLength := runewidth.RuneWidth(r)
			hasSpace := b.GetLineSpacing(b.DisplayPos/b.LineWidth - 1)

			if b.DisplayPos%b.LineWidth == 0 {
				// if the user backspaces over the word boundary, do this magic to clear the line
				// and move to the end of the previous line
				fmt.Print(CursorBOL + ClearToEOL + CursorUp + CursorBOL + CursorRightN(b.Width))

				if b.DisplaySize()%b.LineWidth < (b.DisplaySize()-rLength)%b.LineWidth {
					b.LineHasSpace.Remove(b.DisplayPos/b.LineWidth - 1)
				}

				if hasSpace {
					b.DisplayPos -= 1
					fmt.Print(CursorLeft)
				}

				if rLength == 2 {
					fmt.Print(CursorLeft + "  " + CursorLeftN(2))
				} else {
					fmt.Print(" " + CursorLeft)
				}
			} else if (b.DisplayPos-rLength)%b.LineWidth == 0 && hasSpace {
				fmt.Print(CursorBOL + ClearToEOL + CursorUp + CursorBOL + CursorRightN(b.Width))

				if b.Pos == b.Buf.Size() {
					b.LineHasSpace.Remove(b.DisplayPos/b.LineWidth - 1)
				}
				b.DisplayPos -= 1
			} else {
				fmt.Print(CursorLeftN(rLength))
				for range rLength {
					fmt.Print(" ")
				}
				fmt.Print(CursorLeftN(rLength))
			}

			var eraseExtraLine bool
			if (b.DisplaySize()-1)%b.LineWidth == 0 || (rLength == 2 && ((b.DisplaySize()-2)%b.LineWidth == 0)) || b.DisplaySize()%b.LineWidth == 0 {
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
					remainingLines := (b.DisplaySize() - b.DisplayPos) / b.LineWidth
					fmt.Print(CursorDownN(remainingLines+1) + CursorBOL + ClearToEOL)
					place := b.DisplayPos % b.LineWidth
					fmt.Print(CursorUpN(remainingLines+1) + CursorRightN(place+len(b.Prompt.prompt())))
				}
			}
		}
	}
}

func (b *Buffer) Delete() {
	if b.Buf.Size() > 0 && b.Pos < b.Buf.Size() {
		b.Buf.Remove(b.Pos)
		b.drawRemaining()
		if b.DisplaySize()%b.LineWidth == 0 {
			if b.DisplayPos != b.DisplaySize() {
				remainingLines := (b.DisplaySize() - b.DisplayPos) / b.LineWidth
				fmt.Print(CursorDownN(remainingLines) + CursorBOL + ClearToEOL)
				place := b.DisplayPos % b.LineWidth
				fmt.Print(CursorUpN(remainingLines) + CursorRightN(place+len(b.Prompt.prompt())))
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
	if b.DisplaySize() > 0 && b.Pos < b.DisplaySize() {
		charsToDel := b.Buf.Size() - b.Pos
		for range charsToDel {
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
	fmt.Print(ClearScreen + CursorReset + b.Prompt.prompt())
	if b.IsEmpty() {
		ph := b.Prompt.placeholder()
		fmt.Print(ColorGrey + ph + CursorLeftN(len(ph)) + ColorDefault)
	} else {
		currPos := b.DisplayPos
		currIndex := b.Pos
		b.Pos = 0
		b.DisplayPos = 0
		b.drawRemaining()
		fmt.Print(CursorReset + CursorRightN(len(b.Prompt.prompt())))
		if currPos > 0 {
			targetLine := currPos / b.LineWidth
			if targetLine > 0 {
				for range targetLine {
					fmt.Print(CursorDown)
				}
			}
			remainder := currPos % b.LineWidth
			if remainder > 0 {
				fmt.Print(CursorRightN(remainder))
			}
			if currPos%b.LineWidth == 0 {
				fmt.Print(CursorBOL + b.Prompt.AltPrompt)
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
	lineNums := b.DisplaySize() / b.LineWidth

	b.Buf.Clear()

	fmt.Print(CursorBOL + ClearToEOL)

	for range lineNums {
		fmt.Print(CursorUp + CursorBOL + ClearToEOL)
	}

	fmt.Print(CursorBOL + b.Prompt.prompt())

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
		s += string(c)
	}
	return s
}
