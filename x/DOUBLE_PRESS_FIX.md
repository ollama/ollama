# Tool approval double-press fix

## What happened
The approval popup reads directly from `os.Stdin` in raw mode. Meanwhile the interactive prompt uses the `readline` package, which spun up a background goroutine (`Terminal.ioloop`) that continuously read from stdin and queued runes on a channel.

When the popup appeared, the first keypress was often consumed by the background goroutine and held in its channel, so the popup never saw it. That forced a second keypress, and the first key would later show up at the next prompt (e.g., a stray `1`).

## Fix
`readline` now reads from stdin synchronously instead of via a background goroutine. The terminal is not left in raw mode while idle, and `Terminal.Read()` pulls directly from a buffered reader when `Readline()` is active.

## Files changed
- `readline/readline.go`
