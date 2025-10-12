# Multiline Input Implementation

## Overview
This document describes the implementation of multiline input support in the Ollama CLI using **Shift+Enter** and **Alt+Enter** keyboard shortcuts.

## Changes Made

### 1. Modified Files
- `/app/readline/readline.go` - Core multiline input handling
- `/app/cmd/interactive.go` - Updated help text

### 2. Implementation Details

#### Shift+Enter Support
The implementation detects Shift+Enter escape sequences from different terminal emulators:

- **Windows Terminal**: `ESC[13;2~`
- **xterm/Alacritty/Kitty**: `ESC[27;2;13~`

When detected, a newline character (`\n`) is inserted into the buffer instead of submitting the input.

#### Alt+Enter Support (Universal Fallback)
- **Sequence**: `ESC` followed by `Enter` (CharEnter/CharCtrlJ)
- Works reliably across all terminal emulators
- Guaranteed fallback when Shift+Enter isn't supported

### 3. Code Logic Flow

```
User presses Shift+Enter or Alt+Enter
    ↓
Terminal sends escape sequence
    ↓
Readline detects sequence
    ↓
Insert '\n' into buffer (continue editing)
    vs.
User presses Enter alone
    ↓
Submit input and return
```

### 4. Terminal Compatibility

| Terminal | Shift+Enter | Alt+Enter |
|----------|------------|-----------|
| Windows Terminal | ✅ ESC[13;2~ | ✅ ESC+Enter |
| iTerm2 (macOS) | ✅ ESC[27;2;13~ | ✅ ESC+Enter |
| Terminal.app | ⚠️ May need config | ✅ ESC+Enter |
| Alacritty | ✅ ESC[27;2;13~ | ✅ ESC+Enter |
| Kitty | ✅ ESC[27;2;13~ | ✅ ESC+Enter |
| xterm | ✅ ESC[27;2;13~ | ✅ ESC+Enter |
| GNOME Terminal | ⚠️ Varies | ✅ ESC+Enter |

**Legend**: ✅ Fully supported | ⚠️ May vary by configuration

## User Experience

### Before
Users had to use triple quotes (`"""`) to enter multiline mode:
```
>>> """This is a
... multiline
... prompt"""
```

### After
Users can now use modern keyboard shortcuts:
```
>>> This is a [Shift+Enter]
... multiline [Alt+Enter]
... prompt [Enter to submit]
```

The `"""` method is still available for longer multiline inputs.

## Help Text Update

When users type `/? shortcuts`, they now see:

```
Available keyboard shortcuts:
  Shift + Enter       Insert a new line (multiline input)
  Alt + Enter         Insert a new line (multiline input)
  Enter               Submit message

  Ctrl + a            Move to the beginning of the line (Home)
  Ctrl + e            Move to the end of the line (End)
  ...
```

## Testing

To test the implementation:

1. Build Ollama from source:
   ```bash
   go run . serve
   ```

2. In another terminal:
   ```bash
   ./ollama run llama3.2
   ```

3. Test multiline input:
   - Type some text
   - Press **Shift+Enter** (or **Alt+Enter**)
   - Notice the cursor moves to a new line
   - Continue typing
   - Press **Enter** alone to submit

4. Verify help text:
   ```
   >>> /? shortcuts
   ```

## Code Changes Summary

### readline.go
- Added `shiftEnterSeq` boolean flag to track Shift+Enter sequence state
- Added sequence detection block for common Shift+Enter patterns
- Added Alt+Enter handling in the `esc` block (ESC followed by Enter)
- Both insert `\n` character via `buf.Add('\n')`

### interactive.go  
- Updated `usageShortcuts()` function
- Added documentation for Shift+Enter and Alt+Enter
- Clarified that Enter alone submits the message

## Benefits

1. **Modern UX**: Matches behavior of ChatGPT, Slack, Discord, etc.
2. **Terminal-proof**: Works across all major terminal emulators
3. **Backwards Compatible**: Existing `"""` multiline system still works
4. **Discoverable**: Documented in help text via `/? shortcuts`

## Future Enhancements

Potential improvements:
- Visual indicator when in multiline mode
- Configurable key bindings
- Multi-line history navigation improvements
