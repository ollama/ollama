package readline

import (
    "testing"
)


// Test generated using Keploy
func TestNewBuffer_ValidPrompt_InitializesCorrectly(t *testing.T) {
    prompt := &Prompt{Prompt: ">", AltPrompt: ">>"}
    buffer, err := NewBuffer(prompt)
    if err != nil {
        t.Fatalf("Expected no error, got %v", err)
    }
    if buffer == nil {
        t.Fatalf("Expected buffer to be initialized, got nil")
    }
    if buffer.Prompt != prompt {
        t.Errorf("Expected prompt to be %v, got %v", prompt, buffer.Prompt)
    }
    if buffer.Width <= 0 || buffer.Height <= 0 {
        t.Errorf("Expected valid terminal dimensions, got width: %d, height: %d", buffer.Width, buffer.Height)
    }
    if buffer.LineWidth != buffer.Width-len(prompt.Prompt) {
        t.Errorf("Expected LineWidth to be %d, got %d", buffer.Width-len(prompt.Prompt), buffer.LineWidth)
    }
}

// Test generated using Keploy
func TestBuffer_MoveLeft_UpdatesPositionCorrectly(t *testing.T) {
    prompt := &Prompt{Prompt: ">"}
    buffer, _ := NewBuffer(prompt)
    buffer.Buf.Add('a')
    buffer.Pos = 1
    buffer.DisplayPos = 1

    buffer.MoveLeft()

    if buffer.Pos != 0 {
        t.Errorf("Expected position to be 0, got %d", buffer.Pos)
    }
    if buffer.DisplayPos != 0 {
        t.Errorf("Expected display position to be 0, got %d", buffer.DisplayPos)
    }
}


// Test generated using Keploy
func TestBuffer_Add_AddsRuneCorrectly(t *testing.T) {
    prompt := &Prompt{Prompt: ">"}
    buffer, _ := NewBuffer(prompt)

    buffer.Add('a')

    if buffer.Pos != 1 {
        t.Errorf("Expected position to be 1, got %d", buffer.Pos)
    }
    if buffer.DisplayPos != 1 {
        t.Errorf("Expected display position to be 1, got %d", buffer.DisplayPos)
    }
    if buffer.Buf.Size() != 1 {
        t.Errorf("Expected buffer size to be 1, got %d", buffer.Buf.Size())
    }
    if r, _ := buffer.Buf.Get(0); r != 'a' {
        t.Errorf("Expected rune to be 'a', got %c", r)
    }
}


// Test generated using Keploy
func TestBuffer_IsEmpty_ReturnsTrueForEmptyBuffer(t *testing.T) {
    prompt := &Prompt{Prompt: ">"}
    buffer, _ := NewBuffer(prompt)

    if !buffer.IsEmpty() {
        t.Errorf("Expected buffer to be empty, but it is not")
    }

    buffer.Add('a')

    if buffer.IsEmpty() {
        t.Errorf("Expected buffer to not be empty, but it is")
    }
}


// Test generated using Keploy
func TestBuffer_Replace_ReplacesContentCorrectly(t *testing.T) {
    prompt := &Prompt{Prompt: ">"}
    buffer, _ := NewBuffer(prompt)

    buffer.Replace([]rune("hello"))

    if buffer.Buf.Size() != 5 {
        t.Errorf("Expected buffer size to be 5, got %d", buffer.Buf.Size())
    }
    if buffer.String() != "hello" {
        t.Errorf("Expected buffer content to be 'hello', got %s", buffer.String())
    }
}


// Test generated using Keploy
func TestBuffer_AddChar_InsertsRuneCorrectly(t *testing.T) {
    prompt := &Prompt{Prompt: ">"}
    buffer, _ := NewBuffer(prompt)

    buffer.Replace([]rune("helo"))
    buffer.Pos = 2
    buffer.AddChar('l', true)

    if buffer.String() != "hello" {
        t.Errorf("Expected buffer content to be 'hello', got %s", buffer.String())
    }
    if buffer.Pos != 3 {
        t.Errorf("Expected position to be 3, got %d", buffer.Pos)
    }
}


// Test generated using Keploy
func TestBuffer_Delete_RemovesCharacter(t *testing.T) {
    prompt := &Prompt{Prompt: ">"}
    buffer, _ := NewBuffer(prompt)

    buffer.Replace([]rune("hello"))
    buffer.Pos = 1
    buffer.Delete()

    if buffer.String() != "hllo" {
        t.Errorf("Expected buffer content to be 'hllo', got %s", buffer.String())
    }
    if buffer.Pos != 1 {
        t.Errorf("Expected position to remain 1, got %d", buffer.Pos)
    }
}


// Test generated using Keploy
func TestBuffer_MoveToStart_SetsPositionToStart(t *testing.T) {
    prompt := &Prompt{Prompt: ">"}
    buffer, _ := NewBuffer(prompt)

    buffer.Replace([]rune("hello"))
    buffer.Pos = 3
    buffer.MoveToStart()

    if buffer.Pos != 0 {
        t.Errorf("Expected position to be 0, got %d", buffer.Pos)
    }
}


// Test generated using Keploy
func TestBuffer_MoveToEnd_SetsPositionToEnd(t *testing.T) {
    prompt := &Prompt{Prompt: ">"}
    buffer, _ := NewBuffer(prompt)

    buffer.Replace([]rune("hello"))
    buffer.Pos = 2
    buffer.MoveToEnd()

    if buffer.Pos != buffer.Buf.Size() {
        t.Errorf("Expected position to be %d, got %d", buffer.Buf.Size(), buffer.Pos)
    }
}


// Test generated using Keploy
func TestBuffer_DeleteWord_RemovesPreviousWord(t *testing.T) {
    prompt := &Prompt{Prompt: ">"}
    buffer, _ := NewBuffer(prompt)

    buffer.Replace([]rune("hello world"))
    buffer.Pos = 11
    buffer.DeleteWord()

    if buffer.String() != "hello " {
        t.Errorf("Expected buffer content to be 'hello ', got %s", buffer.String())
    }
    if buffer.Pos != 6 {
        t.Errorf("Expected position to be 6, got %d", buffer.Pos)
    }
}


// Test generated using Keploy
func TestBuffer_DeleteBefore_RemovesCharactersBeforeCursor(t *testing.T) {
    prompt := &Prompt{Prompt: ">"}
    buffer, _ := NewBuffer(prompt)

    buffer.Replace([]rune("hello world"))
    buffer.Pos = 5 // Cursor after 'hello'
    buffer.DeleteBefore()

    if buffer.String() != " world" {
        t.Errorf("Expected buffer content to be ' world', got '%s'", buffer.String())
    }
    if buffer.Pos != 0 {
        t.Errorf("Expected cursor position to be 0, got %d", buffer.Pos)
    }
}


// Test generated using Keploy
func TestBuffer_DeleteRemaining_RemovesAllCharactersAfterCursor(t *testing.T) {
    prompt := &Prompt{Prompt: ">"}
    buffer, _ := NewBuffer(prompt)

    buffer.Replace([]rune("hello world"))
    buffer.Pos = 5 // Cursor after 'hello'
    buffer.DeleteRemaining()

    if buffer.String() != "hello" {
        t.Errorf("Expected buffer content to be 'hello', got '%s'", buffer.String())
    }
    if buffer.Pos != 5 {
        t.Errorf("Expected cursor position to remain at 5, got %d", buffer.Pos)
    }
}


// Test generated using Keploy
func TestBuffer_MoveLeftWord_MovesCursorToStartOfPreviousWord(t *testing.T) {
    prompt := &Prompt{Prompt: ">"}
    buffer, _ := NewBuffer(prompt)

    buffer.Replace([]rune("hello world"))
    buffer.Pos = 11 // Cursor at end of buffer
    buffer.MoveLeftWord()

    if buffer.Pos != 6 {
        t.Errorf("Expected cursor position to be 6, got %d", buffer.Pos)
    }
}


// Test generated using Keploy
func TestBuffer_MoveRightWord_MovesCursorToStartOfNextWord(t *testing.T) {
    prompt := &Prompt{Prompt: ">"}
    buffer, _ := NewBuffer(prompt)

    buffer.Replace([]rune("hello world"))
    buffer.Pos = 0 // Cursor at start
    buffer.MoveRightWord()

    if buffer.Pos != 5 {
        t.Errorf("Expected cursor position to be 5, got %d", buffer.Pos)
    }
}

