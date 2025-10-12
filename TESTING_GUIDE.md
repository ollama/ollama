# Testing Guide: Multiline Input

## Quick Start

### Build Ollama
```bash
cd /app
go run . serve
```

### Run Interactive Mode
In a new terminal:
```bash
./ollama run llama3.2
```

## Test Scenarios

### Test 1: Basic Shift+Enter
1. Start typing: `Tell me a story about`
2. Press **Shift+Enter**
3. Expected: Cursor moves to new line with `... ` prompt
4. Continue typing: `a brave knight`
5. Press **Enter** alone
6. Expected: Message is submitted to the model

**Visual Example:**
```
>>> Tell me a story about [Shift+Enter]
... a brave knight [Enter]
```

### Test 2: Alt+Enter (Universal Fallback)
1. Start typing: `Write a haiku`
2. Press **Alt+Enter** (or **Option+Enter** on macOS)
3. Expected: Cursor moves to new line
4. Continue typing: `about coding`
5. Press **Enter** alone to submit

**Visual Example:**
```
>>> Write a haiku [Alt+Enter]
... about coding [Enter]
```

### Test 3: Multiple Newlines
1. Type: `Create a list:`
2. Press **Shift+Enter**
3. Type: `- Item 1`
4. Press **Shift+Enter**
5. Type: `- Item 2`
6. Press **Shift+Enter**
7. Type: `- Item 3`
8. Press **Enter** to submit

**Visual Example:**
```
>>> Create a list: [Shift+Enter]
... - Item 1 [Shift+Enter]
... - Item 2 [Shift+Enter]
... - Item 3 [Enter]
```

### Test 4: Mixed with Editing
1. Type: `First line`
2. Press **Shift+Enter**
3. Type: `Second line with typo`
4. Use **Ctrl+U** to delete current line
5. Type: `Second line corrected`
6. Press **Enter** to submit

### Test 5: Help Text
1. Type: `/? shortcuts`
2. Press **Enter**
3. Expected: See documentation for Shift+Enter and Alt+Enter at the top

**Expected Output:**
```
Available keyboard shortcuts:
  Shift + Enter       Insert a new line (multiline input)
  Alt + Enter         Insert a new line (multiline input)
  Enter               Submit message

  Ctrl + a            Move to the beginning of the line (Home)
  ...
```

### Test 6: Triple-Quote Still Works
1. Type: `"""`
2. Expected: Enters multiline mode with `... ` prompt
3. Type multiple lines normally
4. Type `"""` to exit multiline mode
5. Press **Enter** to submit

**Visual Example:**
```
>>> """This is [Enter]
... a multiline [Enter]
... message""" [Enter]
```

## Terminal-Specific Testing

### Windows Terminal
- Shift+Enter should work natively
- Alt+Enter should also work

### iTerm2/Terminal.app (macOS)
- Shift+Enter should work (may need configuration)
- Alt+Enter (Option+Enter) should always work

### Linux Terminals (GNOME Terminal, Konsole, etc.)
- Alt+Enter should work universally
- Shift+Enter may vary by configuration

## Troubleshooting

### Issue: Shift+Enter doesn't work
**Solution**: Use Alt+Enter instead - it's the universal fallback

### Issue: Neither Shift+Enter nor Alt+Enter work
**Possible causes**:
1. Terminal emulator intercepts the key combination
2. Check terminal preferences/settings
3. Use the triple-quote `"""` method as an alternative

### Issue: Submitted when I wanted newline
**Solution**: Make sure you're using Shift+Enter or Alt+Enter, not Enter alone

## Expected Behavior Summary

| Action | Result |
|--------|--------|
| Enter | Submit message |
| Shift+Enter | Insert newline (continue editing) |
| Alt+Enter | Insert newline (continue editing) |
| `"""` | Toggle multiline mode |
| Ctrl+C | Cancel input |
| Ctrl+D | Exit ollama |

## Code Verification

You can verify the implementation by checking:

```bash
# Verify changes were made
git diff readline/readline.go cmd/interactive.go

# Or run the verification script
./verify_multiline_changes.sh
```

## Performance Notes

- Multiline input is handled in real-time
- No performance impact compared to single-line input
- History navigation works with multiline entries

## Feedback

If you encounter issues or have suggestions:
1. Check which terminal you're using
2. Verify Go version compatibility (Go 1.24.0+)
3. Test with Alt+Enter as fallback
4. Report issues with terminal name and OS version
