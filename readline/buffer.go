package readline

import (
	"fmt"
	"os"

	"github.com/emirpasic/gods/v2/lists/arraylist"
	"github.com/mattn/go-runewidth"
	"golang.org/x/term"
)

type Buffer struct {
	Prompt    *Prompt
	LineWidth int
	Width     int
	Height    int

	line       *arraylist.List[rune]
	spaceMask  *arraylist.List[bool]
	pos        int
	displayPos int
}

func NewBuffer(prompt *Prompt) (*Buffer, error) {
	fd := int(os.Stdout.Fd())
	width, height := 80, 24
	if termWidth, termHeight, err := term.GetSize(fd); err == nil {
		width, height = termWidth, termHeight
	}

	lwidth := width - len(prompt.prompt())

	return &Buffer{
		displayPos: 0,
		pos:        0,
		line:       arraylist.New[rune](),
		spaceMask:  arraylist.New[bool](),
		Prompt:     prompt,
		Width:      width,
		Height:     height,
		LineWidth:  lwidth,
	}, nil
}

func (b *Buffer) GetLineSpacing(line int) bool {
	hasSpace, _ := b.spaceMask.Get(line)
	return hasSpace
}

func (b *Buffer) MoveLeft() {
	if b.pos > 0 {
		r, _ := b.line.Get(b.pos - 1)
		rLength := runewidth.RuneWidth(r)

		if b.displayPos%b.LineWidth == 0 {
			fmt.Print(CursorUp + CursorBOL + CursorRightN(b.Width))
			if rLength == 2 {
				fmt.Print(CursorLeft)
			}

			line := b.displayPos/b.LineWidth - 1
			hasSpace := b.GetLineSpacing(line)
			if hasSpace {
				b.displayPos -= 1
				fmt.Print(CursorLeft)
			}
		} else {
			fmt.Print(CursorLeftN(rLength))
		}

		b.pos -= 1
		b.displayPos -= rLength
	}
}

func (b *Buffer) MoveLeftWord() {
	var foundNonspace bool
	for b.pos > 0 {
		v, _ := b.line.Get(b.pos - 1)
		if v == ' ' {
			if foundNonspace {
				break
			}
		} else {
			foundNonspace = true
		}
		b.MoveLeft()
	}
}

func (b *Buffer) MoveRight() {
	if b.pos < b.line.Size() {
		r, _ := b.line.Get(b.pos)
		rLength := runewidth.RuneWidth(r)
		b.pos += 1
		hasSpace := b.GetLineSpacing(b.displayPos / b.LineWidth)
		b.displayPos += rLength

		if b.displayPos%b.LineWidth == 0 {
			fmt.Print(CursorDown + CursorBOL + CursorRightN(len(b.Prompt.prompt())))
		} else if (b.displayPos-rLength)%b.LineWidth == b.LineWidth-1 && hasSpace {
			fmt.Print(CursorDown + CursorBOL + CursorRightN(len(b.Prompt.prompt())+rLength))
			b.displayPos += 1
		} else if b.spaceMask.Size() > 0 && b.displayPos%b.LineWidth == b.LineWidth-1 && hasSpace {
			fmt.Print(CursorDown + CursorBOL + CursorRightN(len(b.Prompt.prompt())))
			b.displayPos += 1
		} else {
			fmt.Print(CursorRightN(rLength))
		}
	}
}

func (b *Buffer) MoveRightWord() {
	for b.pos < b.line.Size() {
		b.MoveRight()
		v, _ := b.line.Get(b.pos)
		if v == ' ' {
			break
		}
	}
}

func (b *Buffer) MoveToStart() {
	if b.pos > 0 {
		currLine := b.displayPos / b.LineWidth
		if currLine > 0 {
			for range currLine {
				fmt.Print(CursorUp)
			}
		}
		fmt.Print(CursorBOL + CursorRightN(len(b.Prompt.prompt())))
		b.pos = 0
		b.displayPos = 0
	}
}

func (b *Buffer) MoveToEnd() {
	if b.pos < b.line.Size() {
		currLine := b.displayPos / b.LineWidth
		totalLines := b.DisplaySize() / b.LineWidth
		if currLine < totalLines {
			for range totalLines - currLine {
				fmt.Print(CursorDown)
			}
			remainder := b.DisplaySize() % b.LineWidth
			fmt.Print(CursorBOL + CursorRightN(len(b.Prompt.prompt())+remainder))
		} else {
			fmt.Print(CursorRightN(b.DisplaySize() - b.displayPos))
		}

		b.pos = b.line.Size()
		b.displayPos = b.DisplaySize()
	}
}

func (b *Buffer) DisplaySize() int {
	sum := 0
	for i := range b.line.Size() {
		if r, ok := b.line.Get(i); ok {
			sum += runewidth.RuneWidth(r)
		}
	}

	return sum
}

func (b *Buffer) Add(r rune) {
	if b.pos == b.line.Size() {
		b.AddChar(r, false)
	} else {
		b.AddChar(r, true)
	}
}

func (b *Buffer) AddChar(r rune, insert bool) {
	rLength := runewidth.RuneWidth(r)
	b.displayPos += rLength

	if b.pos > 0 {
		if b.displayPos%b.LineWidth == 0 {
			fmt.Printf("%c", r)
			fmt.Printf("\n%s", b.Prompt.AltPrompt)

			if insert {
				b.spaceMask.Set(b.displayPos/b.LineWidth-1, false)
			} else {
				b.spaceMask.Add(false)
			}

			// this case occurs when a double-width rune crosses the line boundary
		} else if b.displayPos%b.LineWidth < (b.displayPos-rLength)%b.LineWidth {
			if insert {
				fmt.Print(ClearToEOL)
			}
			fmt.Printf("\n%s", b.Prompt.AltPrompt)
			b.displayPos += 1
			fmt.Printf("%c", r)

			if insert {
				b.spaceMask.Set(b.displayPos/b.LineWidth-1, true)
			} else {
				b.spaceMask.Add(true)
			}
		} else {
			fmt.Printf("%c", r)
		}
	} else {
		fmt.Printf("%c", r)
	}

	if insert {
		b.line.Insert(b.pos, r)
	} else {
		b.line.Add(r)
	}

	b.pos += 1

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
		if r, ok := b.line.Get(b.pos + counter); ok {
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
	remainingText := b.StringN(b.pos)
	if b.pos > 0 {
		place = b.displayPos % b.LineWidth
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
		b.spaceMask.Set(b.displayPos/b.LineWidth, true)
	} else if currLineSpace != b.LineWidth-place {
		b.spaceMask.Remove(b.displayPos / b.LineWidth)
	} else {
		b.spaceMask.Set(b.displayPos/b.LineWidth, false)
	}

	if (b.displayPos+currLineSpace)%b.LineWidth == 0 && currLine == remainingText {
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
						b.spaceMask.Set(b.displayPos/b.LineWidth+totalLines-1, false)
					} else {
						b.spaceMask.Set(b.displayPos/b.LineWidth+totalLines-1, true)
					}
				}

				lineLength = 0
			}

			displayLength += runewidth.RuneWidth(c)
			lineLength += runewidth.RuneWidth(c)
			fmt.Printf("%c", c)
		}
		fmt.Print(ClearToEOL + CursorUpN(totalLines) + CursorBOL + CursorRightN(b.Width-currLineSpace))

		hasSpace := b.GetLineSpacing(b.displayPos / b.LineWidth)

		if hasSpace && b.displayPos%b.LineWidth != b.LineWidth-1 {
			fmt.Print(CursorLeft)
		}
	}

	fmt.Print(CursorShow)
}

func (b *Buffer) Remove() {
	if b.line.Size() > 0 && b.pos > 0 {
		if r, ok := b.line.Get(b.pos - 1); ok {
			rLength := runewidth.RuneWidth(r)
			hasSpace := b.GetLineSpacing(b.displayPos/b.LineWidth - 1)

			if b.displayPos%b.LineWidth == 0 {
				// if the user backspaces over the word boundary, do this magic to clear the line
				// and move to the end of the previous line
				fmt.Print(CursorBOL + ClearToEOL + CursorUp + CursorBOL + CursorRightN(b.Width))

				if b.DisplaySize()%b.LineWidth < (b.DisplaySize()-rLength)%b.LineWidth {
					b.spaceMask.Remove(b.displayPos/b.LineWidth - 1)
				}

				if hasSpace {
					b.displayPos -= 1
					fmt.Print(CursorLeft)
				}

				if rLength == 2 {
					fmt.Print(CursorLeft + "  " + CursorLeftN(2))
				} else {
					fmt.Print(" " + CursorLeft)
				}
			} else if (b.displayPos-rLength)%b.LineWidth == 0 && hasSpace {
				fmt.Print(CursorBOL + ClearToEOL + CursorUp + CursorBOL + CursorRightN(b.Width))

				if b.pos == b.line.Size() {
					b.spaceMask.Remove(b.displayPos/b.LineWidth - 1)
				}
				b.displayPos -= 1
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

			b.pos -= 1
			b.displayPos -= rLength
			b.line.Remove(b.pos)

			if b.pos < b.line.Size() {
				b.drawRemaining()
				// this erases a line which is left over when backspacing in the middle of a line and there
				// are trailing characters which go over the line width boundary
				if eraseExtraLine {
					remainingLines := (b.DisplaySize() - b.displayPos) / b.LineWidth
					fmt.Print(CursorDownN(remainingLines+1) + CursorBOL + ClearToEOL)
					place := b.displayPos % b.LineWidth
					fmt.Print(CursorUpN(remainingLines+1) + CursorRightN(place+len(b.Prompt.prompt())))
				}
			}
		}
	}
}

func (b *Buffer) Delete() {
	if b.line.Size() > 0 && b.pos < b.line.Size() {
		b.line.Remove(b.pos)
		b.drawRemaining()
		if b.DisplaySize()%b.LineWidth == 0 {
			if b.displayPos != b.DisplaySize() {
				remainingLines := (b.DisplaySize() - b.displayPos) / b.LineWidth
				fmt.Print(CursorDownN(remainingLines) + CursorBOL + ClearToEOL)
				place := b.displayPos % b.LineWidth
				fmt.Print(CursorUpN(remainingLines) + CursorRightN(place+len(b.Prompt.prompt())))
			}
		}
	}
}

func (b *Buffer) DeleteBefore() {
	if b.pos > 0 {
		for cnt := b.pos - 1; cnt >= 0; cnt-- {
			b.Remove()
		}
	}
}

func (b *Buffer) DeleteRemaining() {
	if b.DisplaySize() > 0 && b.pos < b.DisplaySize() {
		charsToDel := b.line.Size() - b.pos
		for range charsToDel {
			b.Delete()
		}
	}
}

func (b *Buffer) DeleteWord() {
	if b.line.Size() > 0 {
		var foundNonspace bool
		for b.pos > 0 {
			v, _ := b.line.Get(b.pos - 1)
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
		}
	}
}

func (b *Buffer) ClearScreen() {
	fmt.Print(ClearScreen + CursorReset + b.Prompt.prompt())
	if b.IsEmpty() {
		ph := b.Prompt.placeholder()
		fmt.Print(ColorGrey + ph + CursorLeftN(len(ph)) + ColorDefault)
	} else {
		currPos := b.displayPos
		currIndex := b.pos
		b.pos = 0
		b.displayPos = 0
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
		b.pos = currIndex
		b.displayPos = currPos
	}
}

func (b *Buffer) IsEmpty() bool {
	return b.line.Empty()
}

func (b *Buffer) Replace(r []rune) {
	b.displayPos = 0
	b.pos = 0
	lineNums := b.DisplaySize() / b.LineWidth

	b.line.Clear()

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
		m = b.line.Size()
	}
	for cnt := n; cnt < m; cnt++ {
		c, _ := b.line.Get(cnt)
		s += string(c)
	}
	return s
}
