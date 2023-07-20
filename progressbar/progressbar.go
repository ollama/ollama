package progressbar

import (
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/mattn/go-runewidth"
	"github.com/mitchellh/colorstring"
	"golang.org/x/term"
)

// ProgressBar is a thread-safe, simple
// progress bar
type ProgressBar struct {
	state  state
	config config
	lock   sync.Mutex
}

// State is the basic properties of the bar
type State struct {
	CurrentPercent float64
	CurrentBytes   float64
	SecondsSince   float64
	SecondsLeft    float64
	KBsPerSecond   float64
}

type state struct {
	currentNum        int64
	currentPercent    int
	lastPercent       int
	currentSaucerSize int
	isAltSaucerHead   bool

	lastShown time.Time
	startTime time.Time

	counterTime         time.Time
	counterNumSinceLast int64
	counterLastTenRates []float64

	maxLineWidth int
	currentBytes float64
	finished     bool
	exit         bool // Progress bar exit halfway

	rendered string
}

type config struct {
	max                  int64 // max number of the counter
	maxHumanized         string
	maxHumanizedSuffix   string
	width                int
	writer               io.Writer
	theme                Theme
	renderWithBlankState bool
	description          string
	iterationString      string
	ignoreLength         bool // ignoreLength if max bytes not known

	// whether the output is expected to contain color codes
	colorCodes bool

	// show rate of change in kB/sec or MB/sec
	showBytes bool
	// show the iterations per second
	showIterationsPerSecond bool
	showIterationsCount     bool

	// whether the progress bar should show elapsed time.
	// always enabled if predictTime is true.
	elapsedTime bool

	showElapsedTimeOnFinish bool

	// whether the progress bar should attempt to predict the finishing
	// time of the progress based on the start time and the average
	// number of seconds between  increments.
	predictTime bool

	// minimum time to wait in between updates
	throttleDuration time.Duration

	// clear bar once finished
	clearOnFinish bool

	// spinnerType should be a number between 0-75
	spinnerType int

	// spinnerTypeOptionUsed remembers if the spinnerType was changed manually
	spinnerTypeOptionUsed bool

	// spinner represents the spinner as a slice of string
	spinner []string

	// fullWidth specifies whether to measure and set the bar to a specific width
	fullWidth bool

	// invisible doesn't render the bar at all, useful for debugging
	invisible bool

	onCompletion func()

	// whether the render function should make use of ANSI codes to reduce console I/O
	useANSICodes bool

	// showDescriptionAtLineEnd specifies whether description should be written at line end instead of line start
	showDescriptionAtLineEnd bool
}

// Theme defines the elements of the bar
type Theme struct {
	Saucer        string
	AltSaucerHead string
	SaucerHead    string
	SaucerPadding string
	BarStart      string
	BarEnd        string
}

// Option is the type all options need to adhere to
type Option func(p *ProgressBar)

// OptionSetWidth sets the width of the bar
func OptionSetWidth(s int) Option {
	return func(p *ProgressBar) {
		p.config.width = s
	}
}

// OptionSpinnerType sets the type of spinner used for indeterminate bars
func OptionSpinnerType(spinnerType int) Option {
	return func(p *ProgressBar) {
		p.config.spinnerTypeOptionUsed = true
		p.config.spinnerType = spinnerType
	}
}

// OptionSpinnerCustom sets the spinner used for indeterminate bars to the passed
// slice of string
func OptionSpinnerCustom(spinner []string) Option {
	return func(p *ProgressBar) {
		p.config.spinner = spinner
	}
}

// OptionSetTheme sets the elements the bar is constructed of
func OptionSetTheme(t Theme) Option {
	return func(p *ProgressBar) {
		p.config.theme = t
	}
}

// OptionSetVisibility sets the visibility
func OptionSetVisibility(visibility bool) Option {
	return func(p *ProgressBar) {
		p.config.invisible = !visibility
	}
}

// OptionFullWidth sets the bar to be full width
func OptionFullWidth() Option {
	return func(p *ProgressBar) {
		p.config.fullWidth = true
	}
}

// OptionSetWriter sets the output writer (defaults to os.StdOut)
func OptionSetWriter(w io.Writer) Option {
	return func(p *ProgressBar) {
		p.config.writer = w
	}
}

// OptionSetRenderBlankState sets whether or not to render a 0% bar on construction
func OptionSetRenderBlankState(r bool) Option {
	return func(p *ProgressBar) {
		p.config.renderWithBlankState = r
	}
}

// OptionSetDescription sets the description of the bar to render in front of it
func OptionSetDescription(description string) Option {
	return func(p *ProgressBar) {
		p.config.description = description
	}
}

// OptionEnableColorCodes enables or disables support for color codes
// using mitchellh/colorstring
func OptionEnableColorCodes(colorCodes bool) Option {
	return func(p *ProgressBar) {
		p.config.colorCodes = colorCodes
	}
}

// OptionSetElapsedTime will enable elapsed time. Always enabled if OptionSetPredictTime is true.
func OptionSetElapsedTime(elapsedTime bool) Option {
	return func(p *ProgressBar) {
		p.config.elapsedTime = elapsedTime
	}
}

// OptionSetPredictTime will also attempt to predict the time remaining.
func OptionSetPredictTime(predictTime bool) Option {
	return func(p *ProgressBar) {
		p.config.predictTime = predictTime
	}
}

// OptionShowCount will also print current count out of total
func OptionShowCount() Option {
	return func(p *ProgressBar) {
		p.config.showIterationsCount = true
	}
}

// OptionShowIts will also print the iterations/second
func OptionShowIts() Option {
	return func(p *ProgressBar) {
		p.config.showIterationsPerSecond = true
	}
}

// OptionShowElapsedOnFinish will keep the display of elapsed time on finish
func OptionShowElapsedTimeOnFinish() Option {
	return func(p *ProgressBar) {
		p.config.showElapsedTimeOnFinish = true
	}
}

// OptionSetItsString sets what's displayed for iterations a second. The default is "it" which would display: "it/s"
func OptionSetItsString(iterationString string) Option {
	return func(p *ProgressBar) {
		p.config.iterationString = iterationString
	}
}

// OptionThrottle will wait the specified duration before updating again. The default
// duration is 0 seconds.
func OptionThrottle(duration time.Duration) Option {
	return func(p *ProgressBar) {
		p.config.throttleDuration = duration
	}
}

// OptionClearOnFinish will clear the bar once its finished
func OptionClearOnFinish() Option {
	return func(p *ProgressBar) {
		p.config.clearOnFinish = true
	}
}

// OptionOnCompletion will invoke cmpl function once its finished
func OptionOnCompletion(cmpl func()) Option {
	return func(p *ProgressBar) {
		p.config.onCompletion = cmpl
	}
}

// OptionShowBytes will update the progress bar
// configuration settings to display/hide kBytes/Sec
func OptionShowBytes(val bool) Option {
	return func(p *ProgressBar) {
		p.config.showBytes = val
	}
}

// OptionUseANSICodes will use more optimized terminal i/o.
//
// Only useful in environments with support for ANSI escape sequences.
func OptionUseANSICodes(val bool) Option {
	return func(p *ProgressBar) {
		p.config.useANSICodes = val
	}
}

// OptionShowDescriptionAtLineEnd defines whether description should be written at line end instead of line start
func OptionShowDescriptionAtLineEnd() Option {
	return func(p *ProgressBar) {
		p.config.showDescriptionAtLineEnd = true
	}
}

var defaultTheme = Theme{Saucer: "█", SaucerPadding: " ", BarStart: "|", BarEnd: "|"}

// NewOptions constructs a new instance of ProgressBar, with any options you specify
func NewOptions(max int, options ...Option) *ProgressBar {
	return NewOptions64(int64(max), options...)
}

// NewOptions64 constructs a new instance of ProgressBar, with any options you specify
func NewOptions64(max int64, options ...Option) *ProgressBar {
	b := ProgressBar{
		state: getBasicState(),
		config: config{
			writer:           os.Stdout,
			theme:            defaultTheme,
			iterationString:  "it",
			width:            40,
			max:              max,
			throttleDuration: 0 * time.Nanosecond,
			elapsedTime:      true,
			predictTime:      true,
			spinnerType:      9,
			invisible:        false,
		},
	}

	for _, o := range options {
		o(&b)
	}

	if b.config.spinnerType < 0 || b.config.spinnerType > 75 {
		panic("invalid spinner type, must be between 0 and 75")
	}

	// ignoreLength if max bytes not known
	if b.config.max == -1 {
		b.config.ignoreLength = true
		b.config.max = int64(b.config.width)
		b.config.predictTime = false
	}

	b.config.maxHumanized, b.config.maxHumanizedSuffix = humanizeBytes(float64(b.config.max))

	if b.config.renderWithBlankState {
		b.RenderBlank()
	}

	return &b
}

func getBasicState() state {
	now := time.Now()
	return state{
		startTime:   now,
		lastShown:   now,
		counterTime: now,
	}
}

// New returns a new ProgressBar
// with the specified maximum
func New(max int) *ProgressBar {
	return NewOptions(max)
}

// DefaultBytes provides a progressbar to measure byte
// throughput with recommended defaults.
// Set maxBytes to -1 to use as a spinner.
func DefaultBytes(maxBytes int64, description ...string) *ProgressBar {
	desc := ""
	if len(description) > 0 {
		desc = description[0]
	}
	return NewOptions64(
		maxBytes,
		OptionSetDescription(desc),
		OptionSetWriter(os.Stderr),
		OptionShowBytes(true),
		OptionSetWidth(10),
		OptionThrottle(65*time.Millisecond),
		OptionShowCount(),
		OptionOnCompletion(func() {
			fmt.Fprint(os.Stderr, "\n")
		}),
		OptionSpinnerType(14),
		OptionFullWidth(),
		OptionSetRenderBlankState(true),
	)
}

// DefaultBytesSilent is the same as DefaultBytes, but does not output anywhere.
// String() can be used to get the output instead.
func DefaultBytesSilent(maxBytes int64, description ...string) *ProgressBar {
	// Mostly the same bar as DefaultBytes

	desc := ""
	if len(description) > 0 {
		desc = description[0]
	}
	return NewOptions64(
		maxBytes,
		OptionSetDescription(desc),
		OptionSetWriter(io.Discard),
		OptionShowBytes(true),
		OptionSetWidth(10),
		OptionThrottle(65*time.Millisecond),
		OptionShowCount(),
		OptionSpinnerType(14),
		OptionFullWidth(),
	)
}

// Default provides a progressbar with recommended defaults.
// Set max to -1 to use as a spinner.
func Default(max int64, description ...string) *ProgressBar {
	desc := ""
	if len(description) > 0 {
		desc = description[0]
	}
	return NewOptions64(
		max,
		OptionSetDescription(desc),
		OptionSetWriter(os.Stderr),
		OptionSetWidth(10),
		OptionThrottle(65*time.Millisecond),
		OptionShowCount(),
		OptionShowIts(),
		OptionOnCompletion(func() {
			fmt.Fprint(os.Stderr, "\n")
		}),
		OptionSpinnerType(14),
		OptionFullWidth(),
		OptionSetRenderBlankState(true),
	)
}

// DefaultSilent is the same as Default, but does not output anywhere.
// String() can be used to get the output instead.
func DefaultSilent(max int64, description ...string) *ProgressBar {
	// Mostly the same bar as Default

	desc := ""
	if len(description) > 0 {
		desc = description[0]
	}
	return NewOptions64(
		max,
		OptionSetDescription(desc),
		OptionSetWriter(io.Discard),
		OptionSetWidth(10),
		OptionThrottle(65*time.Millisecond),
		OptionShowCount(),
		OptionShowIts(),
		OptionSpinnerType(14),
		OptionFullWidth(),
	)
}

// String returns the current rendered version of the progress bar.
// It will never return an empty string while the progress bar is running.
func (p *ProgressBar) String() string {
	return p.state.rendered
}

// RenderBlank renders the current bar state, you can use this to render a 0% state
func (p *ProgressBar) RenderBlank() error {
	if p.config.invisible {
		return nil
	}
	if p.state.currentNum == 0 {
		p.state.lastShown = time.Time{}
	}
	return p.render()
}

// Reset will reset the clock that is used
// to calculate current time and the time left.
func (p *ProgressBar) Reset() {
	p.lock.Lock()
	defer p.lock.Unlock()

	p.state = getBasicState()
}

// Finish will fill the bar to full
func (p *ProgressBar) Finish() error {
	p.lock.Lock()
	p.state.currentNum = p.config.max
	p.lock.Unlock()
	return p.Add(0)
}

// Exit will exit the bar to keep current state
func (p *ProgressBar) Exit() error {
	p.lock.Lock()
	defer p.lock.Unlock()

	p.state.exit = true
	if p.config.onCompletion != nil {
		p.config.onCompletion()
	}
	return nil
}

// Add will add the specified amount to the progressbar
func (p *ProgressBar) Add(num int) error {
	return p.Add64(int64(num))
}

// Set will set the bar to a current number
func (p *ProgressBar) Set(num int) error {
	return p.Set64(int64(num))
}

// Set64 will set the bar to a current number
func (p *ProgressBar) Set64(num int64) error {
	p.lock.Lock()
	toAdd := num - int64(p.state.currentBytes)
	p.lock.Unlock()
	return p.Add64(toAdd)
}

// Add64 will add the specified amount to the progressbar
func (p *ProgressBar) Add64(num int64) error {
	if p.config.invisible {
		return nil
	}
	p.lock.Lock()
	defer p.lock.Unlock()

	if p.state.exit {
		return nil
	}

	// error out since OptionSpinnerCustom will always override a manually set spinnerType
	if p.config.spinnerTypeOptionUsed && len(p.config.spinner) > 0 {
		return errors.New("OptionSpinnerType and OptionSpinnerCustom cannot be used together")
	}

	if p.config.max == 0 {
		return errors.New("max must be greater than 0")
	}

	if p.state.currentNum < p.config.max {
		if p.config.ignoreLength {
			p.state.currentNum = (p.state.currentNum + num) % p.config.max
		} else {
			p.state.currentNum += num
		}
	}

	p.state.currentBytes += float64(num)

	// reset the countdown timer every second to take rolling average
	p.state.counterNumSinceLast += num
	if time.Since(p.state.counterTime).Seconds() > 0.5 {
		p.state.counterLastTenRates = append(p.state.counterLastTenRates, float64(p.state.counterNumSinceLast)/time.Since(p.state.counterTime).Seconds())
		if len(p.state.counterLastTenRates) > 10 {
			p.state.counterLastTenRates = p.state.counterLastTenRates[1:]
		}
		p.state.counterTime = time.Now()
		p.state.counterNumSinceLast = 0
	}

	percent := float64(p.state.currentNum) / float64(p.config.max)
	p.state.currentSaucerSize = int(percent * float64(p.config.width))
	p.state.currentPercent = int(percent * 100)
	updateBar := p.state.currentPercent != p.state.lastPercent && p.state.currentPercent > 0

	p.state.lastPercent = p.state.currentPercent
	if p.state.currentNum > p.config.max {
		return errors.New("current number exceeds max")
	}

	// always update if show bytes/second or its/second
	if updateBar || p.config.showIterationsPerSecond || p.config.showIterationsCount {
		return p.render()
	}

	return nil
}

// Clear erases the progress bar from the current line
func (p *ProgressBar) Clear() error {
	return clearProgressBar(p.config, p.state)
}

// Describe will change the description shown before the progress, which
// can be changed on the fly (as for a slow running process).
func (p *ProgressBar) Describe(description string) {
	p.lock.Lock()
	defer p.lock.Unlock()
	p.config.description = description
	if p.config.invisible {
		return
	}
	p.render()
}

// New64 returns a new ProgressBar
// with the specified maximum
func New64(max int64) *ProgressBar {
	return NewOptions64(max)
}

// GetMax returns the max of a bar
func (p *ProgressBar) GetMax() int {
	return int(p.config.max)
}

// GetMax64 returns the current max
func (p *ProgressBar) GetMax64() int64 {
	return p.config.max
}

// ChangeMax takes in a int
// and changes the max value
// of the progress bar
func (p *ProgressBar) ChangeMax(newMax int) {
	p.ChangeMax64(int64(newMax))
}

// ChangeMax64 is basically
// the same as ChangeMax,
// but takes in a int64
// to avoid casting
func (p *ProgressBar) ChangeMax64(newMax int64) {
	p.config.max = newMax

	if p.config.showBytes {
		p.config.maxHumanized, p.config.maxHumanizedSuffix = humanizeBytes(float64(p.config.max))
	}

	p.Add(0) // re-render
}

// IsFinished returns true if progress bar is completed
func (p *ProgressBar) IsFinished() bool {
	return p.state.finished
}

// render renders the progress bar, updating the maximum
// rendered line width. this function is not thread-safe,
// so it must be called with an acquired lock.
func (p *ProgressBar) render() error {
	// make sure that the rendering is not happening too quickly
	// but always show if the currentNum reaches the max
	if time.Since(p.state.lastShown).Nanoseconds() < p.config.throttleDuration.Nanoseconds() &&
		p.state.currentNum < p.config.max {
		return nil
	}

	if !p.config.useANSICodes {
		// first, clear the existing progress bar
		err := clearProgressBar(p.config, p.state)
		if err != nil {
			return err
		}
	}

	// check if the progress bar is finished
	if !p.state.finished && p.state.currentNum >= p.config.max {
		p.state.finished = true
		if !p.config.clearOnFinish {
			renderProgressBar(p.config, &p.state)
		}
		if p.config.onCompletion != nil {
			p.config.onCompletion()
		}
	}
	if p.state.finished {
		// when using ANSI codes we don't pre-clean the current line
		if p.config.useANSICodes && p.config.clearOnFinish {
			err := clearProgressBar(p.config, p.state)
			if err != nil {
				return err
			}
		}
		return nil
	}

	// then, re-render the current progress bar
	w, err := renderProgressBar(p.config, &p.state)
	if err != nil {
		return err
	}

	if w > p.state.maxLineWidth {
		p.state.maxLineWidth = w
	}

	p.state.lastShown = time.Now()

	return nil
}

// State returns the current state
func (p *ProgressBar) State() State {
	p.lock.Lock()
	defer p.lock.Unlock()
	s := State{}
	s.CurrentPercent = float64(p.state.currentNum) / float64(p.config.max)
	s.CurrentBytes = p.state.currentBytes
	s.SecondsSince = time.Since(p.state.startTime).Seconds()
	if p.state.currentNum > 0 {
		s.SecondsLeft = s.SecondsSince / float64(p.state.currentNum) * (float64(p.config.max) - float64(p.state.currentNum))
	}
	s.KBsPerSecond = float64(p.state.currentBytes) / 1000.0 / s.SecondsSince
	return s
}

// regex matching ansi escape codes
var ansiRegex = regexp.MustCompile(`\x1b\[[0-9;]*[a-zA-Z]`)

func getStringWidth(c config, str string, colorize bool) int {
	if c.colorCodes {
		// convert any color codes in the progress bar into the respective ANSI codes
		str = colorstring.Color(str)
	}

	// the width of the string, if printed to the console
	// does not include the carriage return character
	cleanString := strings.Replace(str, "\r", "", -1)

	if c.colorCodes {
		// the ANSI codes for the colors do not take up space in the console output,
		// so they do not count towards the output string width
		cleanString = ansiRegex.ReplaceAllString(cleanString, "")
	}

	// get the amount of runes in the string instead of the
	// character count of the string, as some runes span multiple characters.
	// see https://stackoverflow.com/a/12668840/2733724
	stringWidth := runewidth.StringWidth(cleanString)
	return stringWidth
}

func renderProgressBar(c config, s *state) (int, error) {
	var sb strings.Builder

	averageRate := average(s.counterLastTenRates)
	if len(s.counterLastTenRates) == 0 || s.finished {
		// if no average samples, or if finished,
		// then average rate should be the total rate
		if t := time.Since(s.startTime).Seconds(); t > 0 {
			averageRate = s.currentBytes / t
		} else {
			averageRate = 0
		}
	}

	// show iteration count in "current/total" iterations format
	if c.showIterationsCount {
		if sb.Len() == 0 {
			sb.WriteString("(")
		} else {
			sb.WriteString(", ")
		}
		if !c.ignoreLength {
			if c.showBytes {
				currentHumanize, currentSuffix := humanizeBytes(s.currentBytes)
				if currentSuffix == c.maxHumanizedSuffix {
					sb.WriteString(fmt.Sprintf("%s/%s%s",
						currentHumanize, c.maxHumanized, c.maxHumanizedSuffix))
				} else {
					sb.WriteString(fmt.Sprintf("%s%s/%s%s",
						currentHumanize, currentSuffix, c.maxHumanized, c.maxHumanizedSuffix))
				}
			} else {
				sb.WriteString(fmt.Sprintf("%.0f/%d", s.currentBytes, c.max))
			}
		} else {
			if c.showBytes {
				currentHumanize, currentSuffix := humanizeBytes(s.currentBytes)
				sb.WriteString(fmt.Sprintf("%s%s", currentHumanize, currentSuffix))
			} else {
				sb.WriteString(fmt.Sprintf("%.0f/%s", s.currentBytes, "-"))
			}
		}
	}

	// show rolling average rate
	if c.showBytes && averageRate > 0 && !math.IsInf(averageRate, 1) {
		if sb.Len() == 0 {
			sb.WriteString("(")
		} else {
			sb.WriteString(", ")
		}
		currentHumanize, currentSuffix := humanizeBytes(averageRate)
		sb.WriteString(fmt.Sprintf("%s%s/s", currentHumanize, currentSuffix))
	}

	// show iterations rate
	if c.showIterationsPerSecond {
		if sb.Len() == 0 {
			sb.WriteString("(")
		} else {
			sb.WriteString(", ")
		}
		if averageRate > 1 {
			sb.WriteString(fmt.Sprintf("%0.0f %s/s", averageRate, c.iterationString))
		} else if averageRate*60 > 1 {
			sb.WriteString(fmt.Sprintf("%0.0f %s/min", 60*averageRate, c.iterationString))
		} else {
			sb.WriteString(fmt.Sprintf("%0.0f %s/hr", 3600*averageRate, c.iterationString))
		}
	}
	if sb.Len() > 0 {
		sb.WriteString(")")
	}

	leftBrac, rightBrac, saucer, saucerHead := "", "", "", ""

	// show time prediction in "current/total" seconds format
	switch {
	case c.predictTime:
		rightBracNum := (time.Duration((1/averageRate)*(float64(c.max)-float64(s.currentNum))) * time.Second)
		if rightBracNum.Seconds() < 0 {
			rightBracNum = 0 * time.Second
		}
		rightBrac = rightBracNum.String()
		fallthrough
	case c.elapsedTime:
		leftBrac = (time.Duration(time.Since(s.startTime).Seconds()) * time.Second).String()
	}

	if c.fullWidth && !c.ignoreLength {
		width, err := termWidth()
		if err != nil {
			width = 80
		}

		amend := 1 // an extra space at eol
		switch {
		case leftBrac != "" && rightBrac != "":
			amend = 4 // space, square brackets and colon
		case leftBrac != "" && rightBrac == "":
			amend = 4 // space and square brackets and another space
		case leftBrac == "" && rightBrac != "":
			amend = 3 // space and square brackets
		}
		if c.showDescriptionAtLineEnd {
			amend += 1 // another space
		}

		c.width = width - getStringWidth(c, c.description, true) - 10 - amend - sb.Len() - len(leftBrac) - len(rightBrac)
		s.currentSaucerSize = int(float64(s.currentPercent) / 100.0 * float64(c.width))
	}
	if s.currentSaucerSize > 0 {
		if c.ignoreLength {
			saucer = strings.Repeat(c.theme.SaucerPadding, s.currentSaucerSize-1)
		} else {
			saucer = strings.Repeat(c.theme.Saucer, s.currentSaucerSize-1)
		}

		// Check if an alternate saucer head is set for animation
		if c.theme.AltSaucerHead != "" && s.isAltSaucerHead {
			saucerHead = c.theme.AltSaucerHead
			s.isAltSaucerHead = false
		} else if c.theme.SaucerHead == "" || s.currentSaucerSize == c.width {
			// use the saucer for the saucer head if it hasn't been set
			// to preserve backwards compatibility
			saucerHead = c.theme.Saucer
		} else {
			saucerHead = c.theme.SaucerHead
			s.isAltSaucerHead = true
		}
	}

	/*
		Progress Bar format
		Description % |------        |  (kb/s) (iteration count) (iteration rate) (predict time)

		or if showDescriptionAtLineEnd is enabled
		% |------        |  (kb/s) (iteration count) (iteration rate) (predict time) Description
	*/

	repeatAmount := c.width - s.currentSaucerSize
	if repeatAmount < 0 {
		repeatAmount = 0
	}

	str := ""

	if c.ignoreLength {
		selectedSpinner := spinners[c.spinnerType]
		if len(c.spinner) > 0 {
			selectedSpinner = c.spinner
		}
		spinner := selectedSpinner[int(math.Round(math.Mod(float64(time.Since(s.startTime).Milliseconds()/100), float64(len(selectedSpinner)))))]
		if c.elapsedTime {
			if c.showDescriptionAtLineEnd {
				str = fmt.Sprintf("\r%s %s [%s] %s ",
					spinner,
					sb.String(),
					leftBrac,
					c.description)
			} else {
				str = fmt.Sprintf("\r%s %s %s [%s] ",
					spinner,
					c.description,
					sb.String(),
					leftBrac)
			}
		} else {
			if c.showDescriptionAtLineEnd {
				str = fmt.Sprintf("\r%s %s %s ",
					spinner,
					sb.String(),
					c.description)
			} else {
				str = fmt.Sprintf("\r%s %s %s ",
					spinner,
					c.description,
					sb.String())
			}
		}
	} else if rightBrac == "" {
		str = fmt.Sprintf("%4d%% %s%s%s%s%s %s",
			s.currentPercent,
			c.theme.BarStart,
			saucer,
			saucerHead,
			strings.Repeat(c.theme.SaucerPadding, repeatAmount),
			c.theme.BarEnd,
			sb.String())

		if s.currentPercent == 100 && c.showElapsedTimeOnFinish {
			str = fmt.Sprintf("%s [%s]", str, leftBrac)
		}

		if c.showDescriptionAtLineEnd {
			str = fmt.Sprintf("\r%s %s ", str, c.description)
		} else {
			str = fmt.Sprintf("\r%s%s ", c.description, str)
		}
	} else {
		if s.currentPercent == 100 {
			str = fmt.Sprintf("%4d%% %s%s%s%s%s %s",
				s.currentPercent,
				c.theme.BarStart,
				saucer,
				saucerHead,
				strings.Repeat(c.theme.SaucerPadding, repeatAmount),
				c.theme.BarEnd,
				sb.String())

			if c.showElapsedTimeOnFinish {
				str = fmt.Sprintf("%s [%s]", str, leftBrac)
			}

			if c.showDescriptionAtLineEnd {
				str = fmt.Sprintf("\r%s %s", str, c.description)
			} else {
				str = fmt.Sprintf("\r%s%s", c.description, str)
			}
		} else {
			str = fmt.Sprintf("%4d%% %s%s%s%s%s %s [%s:%s]",
				s.currentPercent,
				c.theme.BarStart,
				saucer,
				saucerHead,
				strings.Repeat(c.theme.SaucerPadding, repeatAmount),
				c.theme.BarEnd,
				sb.String(),
				leftBrac,
				rightBrac)

			if c.showDescriptionAtLineEnd {
				str = fmt.Sprintf("\r%s %s", str, c.description)
			} else {
				str = fmt.Sprintf("\r%s%s", c.description, str)
			}
		}
	}

	if c.colorCodes {
		// convert any color codes in the progress bar into the respective ANSI codes
		str = colorstring.Color(str)
	}

	s.rendered = str

	return getStringWidth(c, str, false), writeString(c, str)
}

func clearProgressBar(c config, s state) error {
	if s.maxLineWidth == 0 {
		return nil
	}
	if c.useANSICodes {
		// write the "clear current line" ANSI escape sequence
		return writeString(c, "\033[2K\r")
	}
	// fill the empty content
	// to overwrite the progress bar and jump
	// back to the beginning of the line
	str := fmt.Sprintf("\r%s\r", strings.Repeat(" ", s.maxLineWidth))
	return writeString(c, str)
	// the following does not show correctly if the previous line is longer than subsequent line
	// return writeString(c, "\r")
}

func writeString(c config, str string) error {
	if _, err := io.WriteString(c.writer, str); err != nil {
		return err
	}

	if f, ok := c.writer.(*os.File); ok {
		// ignore any errors in Sync(), as stdout
		// can't be synced on some operating systems
		// like Debian 9 (Stretch)
		f.Sync()
	}

	return nil
}

// Reader is the progressbar io.Reader struct
type Reader struct {
	io.Reader
	bar *ProgressBar
}

// NewReader return a new Reader with a given progress bar.
func NewReader(r io.Reader, bar *ProgressBar) Reader {
	return Reader{
		Reader: r,
		bar:    bar,
	}
}

// Read will read the data and add the number of bytes to the progressbar
func (r *Reader) Read(p []byte) (n int, err error) {
	n, err = r.Reader.Read(p)
	r.bar.Add(n)
	return
}

// Close the reader when it implements io.Closer
func (r *Reader) Close() (err error) {
	if closer, ok := r.Reader.(io.Closer); ok {
		return closer.Close()
	}
	r.bar.Finish()
	return
}

// Write implement io.Writer
func (p *ProgressBar) Write(b []byte) (n int, err error) {
	n = len(b)
	p.Add(n)
	return
}

// Read implement io.Reader
func (p *ProgressBar) Read(b []byte) (n int, err error) {
	n = len(b)
	p.Add(n)
	return
}

func (p *ProgressBar) Close() (err error) {
	p.Finish()
	return
}

func average(xs []float64) float64 {
	total := 0.0
	for _, v := range xs {
		total += v
	}
	return total / float64(len(xs))
}

func humanizeBytes(s float64) (string, string) {
	sizes := []string{" B", " kB", " MB", " GB", " TB", " PB", " EB"}
	base := 1000.0
	if s < 10 {
		return fmt.Sprintf("%2.0f", s), sizes[0]
	}
	e := math.Floor(logn(float64(s), base))
	suffix := sizes[int(e)]
	val := math.Floor(float64(s)/math.Pow(base, e)*10+0.5) / 10
	f := "%.0f"
	if val < 10 {
		f = "%.1f"
	}

	return fmt.Sprintf(f, val), suffix
}

func logn(n, b float64) float64 {
	return math.Log(n) / math.Log(b)
}

// termWidth function returns the visible width of the current terminal
// and can be redefined for testing
var termWidth = func() (width int, err error) {
	width, _, err = term.GetSize(int(os.Stdout.Fd()))
	if err == nil {
		return width, nil
	}

	return 0, err
}
