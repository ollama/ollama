package config

import (
	"bytes"
	"strings"
	"testing"
)

func TestFilterItems(t *testing.T) {
	items := []selectItem{
		{Name: "llama3.2:latest"},
		{Name: "qwen2.5:7b"},
		{Name: "deepseek-v3:cloud"},
		{Name: "GPT-OSS:20b"},
	}

	t.Run("EmptyFilter_ReturnsAllItems", func(t *testing.T) {
		result := filterItems(items, "")
		if len(result) != len(items) {
			t.Errorf("expected %d items, got %d", len(items), len(result))
		}
	})

	t.Run("CaseInsensitive_UppercaseFilterMatchesLowercase", func(t *testing.T) {
		result := filterItems(items, "LLAMA")
		if len(result) != 1 || result[0].Name != "llama3.2:latest" {
			t.Errorf("expected llama3.2:latest, got %v", result)
		}
	})

	t.Run("CaseInsensitive_LowercaseFilterMatchesUppercase", func(t *testing.T) {
		result := filterItems(items, "gpt")
		if len(result) != 1 || result[0].Name != "GPT-OSS:20b" {
			t.Errorf("expected GPT-OSS:20b, got %v", result)
		}
	})

	t.Run("PartialMatch", func(t *testing.T) {
		result := filterItems(items, "deep")
		if len(result) != 1 || result[0].Name != "deepseek-v3:cloud" {
			t.Errorf("expected deepseek-v3:cloud, got %v", result)
		}
	})

	t.Run("NoMatch_ReturnsEmpty", func(t *testing.T) {
		result := filterItems(items, "nonexistent")
		if len(result) != 0 {
			t.Errorf("expected 0 items, got %d", len(result))
		}
	})
}

func TestSelectState(t *testing.T) {
	items := []selectItem{
		{Name: "item1"},
		{Name: "item2"},
		{Name: "item3"},
	}

	t.Run("InitialState", func(t *testing.T) {
		s := newSelectState(items)
		if s.selected != 0 {
			t.Errorf("expected selected=0, got %d", s.selected)
		}
		if s.filter != "" {
			t.Errorf("expected empty filter, got %q", s.filter)
		}
		if s.scrollOffset != 0 {
			t.Errorf("expected scrollOffset=0, got %d", s.scrollOffset)
		}
	})

	t.Run("Enter_SelectsCurrentItem", func(t *testing.T) {
		s := newSelectState(items)
		done, result, err := s.handleInput(eventEnter, 0)
		if !done || result != "item1" || err != nil {
			t.Errorf("expected (true, item1, nil), got (%v, %v, %v)", done, result, err)
		}
	})

	t.Run("Enter_WithFilter_SelectsFilteredItem", func(t *testing.T) {
		s := newSelectState(items)
		s.filter = "item3"
		done, result, err := s.handleInput(eventEnter, 0)
		if !done || result != "item3" || err != nil {
			t.Errorf("expected (true, item3, nil), got (%v, %v, %v)", done, result, err)
		}
	})

	t.Run("Enter_EmptyFilteredList_ReturnsFilter", func(t *testing.T) {
		s := newSelectState(items)
		s.filter = "nonexistent"
		done, result, err := s.handleInput(eventEnter, 0)
		if !done || result != "nonexistent" || err != nil {
			t.Errorf("expected (true, 'nonexistent', nil), got (%v, %v, %v)", done, result, err)
		}
	})

	t.Run("Enter_EmptyFilteredList_EmptyFilter_DoesNothing", func(t *testing.T) {
		s := newSelectState([]selectItem{})
		done, result, err := s.handleInput(eventEnter, 0)
		if done || result != "" || err != nil {
			t.Errorf("expected (false, '', nil), got (%v, %v, %v)", done, result, err)
		}
	})

	t.Run("Escape_ReturnsCancelledError", func(t *testing.T) {
		s := newSelectState(items)
		done, result, err := s.handleInput(eventEscape, 0)
		if !done || result != "" || err != errCancelled {
			t.Errorf("expected (true, '', errCancelled), got (%v, %v, %v)", done, result, err)
		}
	})

	t.Run("Down_MovesSelection", func(t *testing.T) {
		s := newSelectState(items)
		s.handleInput(eventDown, 0)
		if s.selected != 1 {
			t.Errorf("expected selected=1, got %d", s.selected)
		}
	})

	t.Run("Down_AtBottom_StaysAtBottom", func(t *testing.T) {
		s := newSelectState(items)
		s.selected = 2
		s.handleInput(eventDown, 0)
		if s.selected != 2 {
			t.Errorf("expected selected=2 (stayed at bottom), got %d", s.selected)
		}
	})

	t.Run("Up_MovesSelection", func(t *testing.T) {
		s := newSelectState(items)
		s.selected = 2
		s.handleInput(eventUp, 0)
		if s.selected != 1 {
			t.Errorf("expected selected=1, got %d", s.selected)
		}
	})

	t.Run("Up_AtTop_StaysAtTop", func(t *testing.T) {
		s := newSelectState(items)
		s.handleInput(eventUp, 0)
		if s.selected != 0 {
			t.Errorf("expected selected=0 (stayed at top), got %d", s.selected)
		}
	})

	t.Run("Char_AppendsToFilter", func(t *testing.T) {
		s := newSelectState(items)
		s.handleInput(eventChar, 'i')
		s.handleInput(eventChar, 't')
		s.handleInput(eventChar, 'e')
		s.handleInput(eventChar, 'm')
		s.handleInput(eventChar, '2')
		if s.filter != "item2" {
			t.Errorf("expected filter='item2', got %q", s.filter)
		}
		filtered := s.filtered()
		if len(filtered) != 1 || filtered[0].Name != "item2" {
			t.Errorf("expected [item2], got %v", filtered)
		}
	})

	t.Run("Char_ResetsSelectionToZero", func(t *testing.T) {
		s := newSelectState(items)
		s.selected = 2
		s.handleInput(eventChar, 'x')
		if s.selected != 0 {
			t.Errorf("expected selected=0 after typing, got %d", s.selected)
		}
	})

	t.Run("Backspace_RemovesLastFilterChar", func(t *testing.T) {
		s := newSelectState(items)
		s.filter = "test"
		s.handleInput(eventBackspace, 0)
		if s.filter != "tes" {
			t.Errorf("expected filter='tes', got %q", s.filter)
		}
	})

	t.Run("Backspace_EmptyFilter_DoesNothing", func(t *testing.T) {
		s := newSelectState(items)
		s.handleInput(eventBackspace, 0)
		if s.filter != "" {
			t.Errorf("expected filter='', got %q", s.filter)
		}
	})

	t.Run("Backspace_ResetsSelectionToZero", func(t *testing.T) {
		s := newSelectState(items)
		s.filter = "test"
		s.selected = 2
		s.handleInput(eventBackspace, 0)
		if s.selected != 0 {
			t.Errorf("expected selected=0 after backspace, got %d", s.selected)
		}
	})

	t.Run("Scroll_DownPastVisibleItems_ScrollsViewport", func(t *testing.T) {
		// maxDisplayedItems is 10, so with 15 items we need to scroll
		manyItems := make([]selectItem, 15)
		for i := range manyItems {
			manyItems[i] = selectItem{Name: string(rune('a' + i))}
		}
		s := newSelectState(manyItems)

		// move down 12 times (past the 10-item viewport)
		for range 12 {
			s.handleInput(eventDown, 0)
		}

		if s.selected != 12 {
			t.Errorf("expected selected=12, got %d", s.selected)
		}
		if s.scrollOffset != 3 {
			t.Errorf("expected scrollOffset=3 (12-10+1), got %d", s.scrollOffset)
		}
	})

	t.Run("Scroll_UpPastScrollOffset_ScrollsViewport", func(t *testing.T) {
		manyItems := make([]selectItem, 15)
		for i := range manyItems {
			manyItems[i] = selectItem{Name: string(rune('a' + i))}
		}
		s := newSelectState(manyItems)
		s.selected = 5
		s.scrollOffset = 5

		s.handleInput(eventUp, 0)

		if s.selected != 4 {
			t.Errorf("expected selected=4, got %d", s.selected)
		}
		if s.scrollOffset != 4 {
			t.Errorf("expected scrollOffset=4, got %d", s.scrollOffset)
		}
	})
}

func TestMultiSelectState(t *testing.T) {
	items := []selectItem{
		{Name: "item1"},
		{Name: "item2"},
		{Name: "item3"},
	}

	t.Run("InitialState_NoPrechecked", func(t *testing.T) {
		s := newMultiSelectState(items, nil)
		if s.highlighted != 0 {
			t.Errorf("expected highlighted=0, got %d", s.highlighted)
		}
		if s.selectedCount() != 0 {
			t.Errorf("expected 0 selected, got %d", s.selectedCount())
		}
		if s.focusOnButton {
			t.Error("expected focusOnButton=false initially")
		}
	})

	t.Run("InitialState_WithPrechecked", func(t *testing.T) {
		s := newMultiSelectState(items, []string{"item2", "item3"})
		if s.selectedCount() != 2 {
			t.Errorf("expected 2 selected, got %d", s.selectedCount())
		}
		if !s.checked[1] || !s.checked[2] {
			t.Error("expected item2 and item3 to be checked")
		}
	})

	t.Run("Prechecked_PreservesSelectionOrder", func(t *testing.T) {
		// order matters: first checked = default model
		s := newMultiSelectState(items, []string{"item3", "item1"})
		if len(s.checkOrder) != 2 {
			t.Fatalf("expected 2 in checkOrder, got %d", len(s.checkOrder))
		}
		if s.checkOrder[0] != 2 || s.checkOrder[1] != 0 {
			t.Errorf("expected checkOrder=[2,0] (item3 first), got %v", s.checkOrder)
		}
	})

	t.Run("Prechecked_IgnoresInvalidNames", func(t *testing.T) {
		s := newMultiSelectState(items, []string{"item1", "nonexistent"})
		if s.selectedCount() != 1 {
			t.Errorf("expected 1 selected (nonexistent ignored), got %d", s.selectedCount())
		}
	})

	t.Run("Toggle_ChecksUncheckedItem", func(t *testing.T) {
		s := newMultiSelectState(items, nil)
		s.toggleItem()
		if !s.checked[0] {
			t.Error("expected item1 to be checked after toggle")
		}
	})

	t.Run("Toggle_UnchecksCheckedItem", func(t *testing.T) {
		s := newMultiSelectState(items, []string{"item1"})
		s.toggleItem()
		if s.checked[0] {
			t.Error("expected item1 to be unchecked after toggle")
		}
	})

	t.Run("Toggle_RemovesFromCheckOrder", func(t *testing.T) {
		s := newMultiSelectState(items, []string{"item1", "item2", "item3"})
		s.highlighted = 1 // toggle item2
		s.toggleItem()

		if len(s.checkOrder) != 2 {
			t.Fatalf("expected 2 in checkOrder, got %d", len(s.checkOrder))
		}
		// should be [0, 2] (item1, item3) with item2 removed
		if s.checkOrder[0] != 0 || s.checkOrder[1] != 2 {
			t.Errorf("expected checkOrder=[0,2], got %v", s.checkOrder)
		}
	})

	t.Run("Enter_TogglesWhenNotOnButton", func(t *testing.T) {
		s := newMultiSelectState(items, nil)
		s.handleInput(eventEnter, 0)
		if !s.checked[0] {
			t.Error("expected item1 to be checked after enter")
		}
	})

	t.Run("Enter_OnButton_ReturnsSelection", func(t *testing.T) {
		s := newMultiSelectState(items, []string{"item2", "item1"})
		s.focusOnButton = true

		done, result, err := s.handleInput(eventEnter, 0)

		if !done || err != nil {
			t.Errorf("expected done=true, err=nil, got done=%v, err=%v", done, err)
		}
		// result should preserve selection order
		if len(result) != 2 || result[0] != "item2" || result[1] != "item1" {
			t.Errorf("expected [item2, item1], got %v", result)
		}
	})

	t.Run("Enter_OnButton_EmptySelection_DoesNothing", func(t *testing.T) {
		s := newMultiSelectState(items, nil)
		s.focusOnButton = true
		done, result, err := s.handleInput(eventEnter, 0)
		if done || result != nil || err != nil {
			t.Errorf("expected (false, nil, nil), got (%v, %v, %v)", done, result, err)
		}
	})

	t.Run("Tab_SwitchesToButton_WhenHasSelection", func(t *testing.T) {
		s := newMultiSelectState(items, []string{"item1"})
		s.handleInput(eventTab, 0)
		if !s.focusOnButton {
			t.Error("expected focus on button after tab")
		}
	})

	t.Run("Tab_DoesNothing_WhenNoSelection", func(t *testing.T) {
		s := newMultiSelectState(items, nil)
		s.handleInput(eventTab, 0)
		if s.focusOnButton {
			t.Error("tab should not focus button when nothing selected")
		}
	})

	t.Run("Tab_TogglesButtonFocus", func(t *testing.T) {
		s := newMultiSelectState(items, []string{"item1"})
		s.handleInput(eventTab, 0)
		if !s.focusOnButton {
			t.Error("expected focus on button after first tab")
		}
		s.handleInput(eventTab, 0)
		if s.focusOnButton {
			t.Error("expected focus back on list after second tab")
		}
	})

	t.Run("Escape_ReturnsCancelledError", func(t *testing.T) {
		s := newMultiSelectState(items, []string{"item1"})
		done, result, err := s.handleInput(eventEscape, 0)
		if !done || result != nil || err != errCancelled {
			t.Errorf("expected (true, nil, errCancelled), got (%v, %v, %v)", done, result, err)
		}
	})

	t.Run("IsDefault_TrueForFirstChecked", func(t *testing.T) {
		s := newMultiSelectState(items, []string{"item2", "item1"})
		if !(len(s.checkOrder) > 0 && s.checkOrder[0] == 1) {
			t.Error("expected item2 (idx 1) to be default (first checked)")
		}
		if len(s.checkOrder) > 0 && s.checkOrder[0] == 0 {
			t.Error("expected item1 (idx 0) to NOT be default")
		}
	})

	t.Run("IsDefault_FalseWhenNothingChecked", func(t *testing.T) {
		s := newMultiSelectState(items, nil)
		if len(s.checkOrder) > 0 && s.checkOrder[0] == 0 {
			t.Error("expected isDefault=false when nothing checked")
		}
	})

	t.Run("Down_MovesHighlight", func(t *testing.T) {
		s := newMultiSelectState(items, nil)
		s.handleInput(eventDown, 0)
		if s.highlighted != 1 {
			t.Errorf("expected highlighted=1, got %d", s.highlighted)
		}
	})

	t.Run("Up_MovesHighlight", func(t *testing.T) {
		s := newMultiSelectState(items, nil)
		s.highlighted = 1
		s.handleInput(eventUp, 0)
		if s.highlighted != 0 {
			t.Errorf("expected highlighted=0, got %d", s.highlighted)
		}
	})

	t.Run("Arrow_ReturnsFocusFromButton", func(t *testing.T) {
		s := newMultiSelectState(items, []string{"item1"})
		s.focusOnButton = true
		s.handleInput(eventDown, 0)
		if s.focusOnButton {
			t.Error("expected focus to return to list on arrow key")
		}
	})

	t.Run("Char_AppendsToFilter", func(t *testing.T) {
		s := newMultiSelectState(items, nil)
		s.handleInput(eventChar, 'x')
		if s.filter != "x" {
			t.Errorf("expected filter='x', got %q", s.filter)
		}
	})

	t.Run("Char_ResetsHighlightAndScroll", func(t *testing.T) {
		manyItems := make([]selectItem, 15)
		for i := range manyItems {
			manyItems[i] = selectItem{Name: string(rune('a' + i))}
		}
		s := newMultiSelectState(manyItems, nil)
		s.highlighted = 10
		s.scrollOffset = 5

		s.handleInput(eventChar, 'x')

		if s.highlighted != 0 {
			t.Errorf("expected highlighted=0, got %d", s.highlighted)
		}
		if s.scrollOffset != 0 {
			t.Errorf("expected scrollOffset=0, got %d", s.scrollOffset)
		}
	})

	t.Run("Backspace_RemovesLastFilterChar", func(t *testing.T) {
		s := newMultiSelectState(items, nil)
		s.filter = "test"
		s.handleInput(eventBackspace, 0)
		if s.filter != "tes" {
			t.Errorf("expected filter='tes', got %q", s.filter)
		}
	})

	t.Run("Backspace_RemovesFocusFromButton", func(t *testing.T) {
		s := newMultiSelectState(items, []string{"item1"})
		s.filter = "x"
		s.focusOnButton = true
		s.handleInput(eventBackspace, 0)
		if s.focusOnButton {
			t.Error("expected focusOnButton=false after backspace")
		}
	})
}

func TestParseInput(t *testing.T) {
	t.Run("Enter", func(t *testing.T) {
		event, char, err := parseInput(bytes.NewReader([]byte{13}))
		if err != nil || event != eventEnter || char != 0 {
			t.Errorf("expected (eventEnter, 0, nil), got (%v, %v, %v)", event, char, err)
		}
	})

	t.Run("Escape", func(t *testing.T) {
		event, _, err := parseInput(bytes.NewReader([]byte{27}))
		if err != nil || event != eventEscape {
			t.Errorf("expected eventEscape, got %v", event)
		}
	})

	t.Run("CtrlC_TreatedAsEscape", func(t *testing.T) {
		event, _, err := parseInput(bytes.NewReader([]byte{3}))
		if err != nil || event != eventEscape {
			t.Errorf("expected eventEscape for Ctrl+C, got %v", event)
		}
	})

	t.Run("Tab", func(t *testing.T) {
		event, _, err := parseInput(bytes.NewReader([]byte{9}))
		if err != nil || event != eventTab {
			t.Errorf("expected eventTab, got %v", event)
		}
	})

	t.Run("Backspace", func(t *testing.T) {
		event, _, err := parseInput(bytes.NewReader([]byte{127}))
		if err != nil || event != eventBackspace {
			t.Errorf("expected eventBackspace, got %v", event)
		}
	})

	t.Run("UpArrow", func(t *testing.T) {
		event, _, err := parseInput(bytes.NewReader([]byte{27, 91, 65}))
		if err != nil || event != eventUp {
			t.Errorf("expected eventUp, got %v", event)
		}
	})

	t.Run("DownArrow", func(t *testing.T) {
		event, _, err := parseInput(bytes.NewReader([]byte{27, 91, 66}))
		if err != nil || event != eventDown {
			t.Errorf("expected eventDown, got %v", event)
		}
	})

	t.Run("PrintableChars", func(t *testing.T) {
		tests := []struct {
			name string
			char byte
		}{
			{"lowercase", 'a'},
			{"uppercase", 'Z'},
			{"digit", '5'},
			{"space", ' '},
			{"tilde", '~'},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				event, char, err := parseInput(bytes.NewReader([]byte{tt.char}))
				if err != nil || event != eventChar || char != tt.char {
					t.Errorf("expected (eventChar, %q), got (%v, %q)", tt.char, event, char)
				}
			})
		}
	})
}

func TestRenderSelect(t *testing.T) {
	items := []selectItem{
		{Name: "item1", Description: "first item"},
		{Name: "item2"},
	}

	t.Run("ShowsPromptAndItems", func(t *testing.T) {
		s := newSelectState(items)
		var buf bytes.Buffer
		lineCount := renderSelect(&buf, "Select:", s)

		output := buf.String()
		if !strings.Contains(output, "Select:") {
			t.Error("expected prompt in output")
		}
		if !strings.Contains(output, "item1") {
			t.Error("expected item1 in output")
		}
		if !strings.Contains(output, "first item") {
			t.Error("expected description in output")
		}
		if !strings.Contains(output, "item2") {
			t.Error("expected item2 in output")
		}
		if lineCount != 3 { // 1 prompt + 2 items
			t.Errorf("expected 3 lines, got %d", lineCount)
		}
	})

	t.Run("EmptyFilteredList_ShowsPullPrompt", func(t *testing.T) {
		s := newSelectState(items)
		s.filter = "xyz"
		var buf bytes.Buffer
		renderSelect(&buf, "Select:", s)

		output := buf.String()
		if !strings.Contains(output, "Download model: 'xyz'?") {
			t.Errorf("expected 'Download model: xyz?' message, got: %s", output)
		}
	})

	t.Run("EmptyFilteredList_EmptyFilter_ShowsNoMatches", func(t *testing.T) {
		s := newSelectState([]selectItem{})
		var buf bytes.Buffer
		renderSelect(&buf, "Select:", s)

		if !strings.Contains(buf.String(), "no matches") {
			t.Error("expected 'no matches' message for empty list with no filter")
		}
	})

	t.Run("LongList_ShowsRemainingCount", func(t *testing.T) {
		manyItems := make([]selectItem, 15)
		for i := range manyItems {
			manyItems[i] = selectItem{Name: string(rune('a' + i))}
		}
		s := newSelectState(manyItems)
		var buf bytes.Buffer
		renderSelect(&buf, "Select:", s)

		// 15 items - 10 displayed = 5 more
		if !strings.Contains(buf.String(), "5 more") {
			t.Error("expected '5 more' indicator")
		}
	})
}

func TestRenderMultiSelect(t *testing.T) {
	items := []selectItem{
		{Name: "item1"},
		{Name: "item2"},
	}

	t.Run("ShowsCheckboxes", func(t *testing.T) {
		s := newMultiSelectState(items, []string{"item1"})
		var buf bytes.Buffer
		renderMultiSelect(&buf, "Select:", s)

		output := buf.String()
		if !strings.Contains(output, "[x]") {
			t.Error("expected checked checkbox [x]")
		}
		if !strings.Contains(output, "[ ]") {
			t.Error("expected unchecked checkbox [ ]")
		}
	})

	t.Run("ShowsDefaultMarker", func(t *testing.T) {
		s := newMultiSelectState(items, []string{"item1"})
		var buf bytes.Buffer
		renderMultiSelect(&buf, "Select:", s)

		if !strings.Contains(buf.String(), "(default)") {
			t.Error("expected (default) marker for first checked item")
		}
	})

	t.Run("ShowsSelectedCount", func(t *testing.T) {
		s := newMultiSelectState(items, []string{"item1", "item2"})
		var buf bytes.Buffer
		renderMultiSelect(&buf, "Select:", s)

		if !strings.Contains(buf.String(), "2 selected") {
			t.Error("expected '2 selected' in output")
		}
	})

	t.Run("NoSelection_ShowsHelperText", func(t *testing.T) {
		s := newMultiSelectState(items, nil)
		var buf bytes.Buffer
		renderMultiSelect(&buf, "Select:", s)

		if !strings.Contains(buf.String(), "Select at least one") {
			t.Error("expected 'Select at least one' helper text")
		}
	})
}

func TestErrCancelled(t *testing.T) {
	t.Run("NotNil", func(t *testing.T) {
		if errCancelled == nil {
			t.Error("errCancelled should not be nil")
		}
	})

	t.Run("Message", func(t *testing.T) {
		if errCancelled.Error() != "cancelled" {
			t.Errorf("expected 'cancelled', got %q", errCancelled.Error())
		}
	})
}

// Edge case tests for selector.go

// TestSelectState_SingleItem verifies that single item list works without crash.
// List with only one item should still work.
func TestSelectState_SingleItem(t *testing.T) {
	items := []selectItem{{Name: "only-one"}}

	s := newSelectState(items)

	// Down should do nothing (already at bottom)
	s.handleInput(eventDown, 0)
	if s.selected != 0 {
		t.Errorf("down on single item: expected selected=0, got %d", s.selected)
	}

	// Up should do nothing (already at top)
	s.handleInput(eventUp, 0)
	if s.selected != 0 {
		t.Errorf("up on single item: expected selected=0, got %d", s.selected)
	}

	// Enter should select the only item
	done, result, err := s.handleInput(eventEnter, 0)
	if !done || result != "only-one" || err != nil {
		t.Errorf("enter on single item: expected (true, 'only-one', nil), got (%v, %q, %v)", done, result, err)
	}
}

// TestSelectState_ExactlyMaxItems verifies boundary condition at maxDisplayedItems.
// List with exactly maxDisplayedItems items should not scroll.
func TestSelectState_ExactlyMaxItems(t *testing.T) {
	items := make([]selectItem, maxDisplayedItems)
	for i := range items {
		items[i] = selectItem{Name: string(rune('a' + i))}
	}

	s := newSelectState(items)

	// Move to last item
	for range maxDisplayedItems - 1 {
		s.handleInput(eventDown, 0)
	}

	if s.selected != maxDisplayedItems-1 {
		t.Errorf("expected selected=%d, got %d", maxDisplayedItems-1, s.selected)
	}

	// Should not scroll when exactly at max
	if s.scrollOffset != 0 {
		t.Errorf("expected scrollOffset=0 for exactly maxDisplayedItems, got %d", s.scrollOffset)
	}

	// One more down should do nothing
	s.handleInput(eventDown, 0)
	if s.selected != maxDisplayedItems-1 {
		t.Errorf("down at max: expected selected=%d, got %d", maxDisplayedItems-1, s.selected)
	}
}

// TestFilterItems_RegexSpecialChars verifies that filter is literal, not regex.
// User typing "model.v1" shouldn't match "modelsv1".
func TestFilterItems_RegexSpecialChars(t *testing.T) {
	items := []selectItem{
		{Name: "model.v1"},
		{Name: "modelsv1"},
		{Name: "model-v1"},
	}

	// Filter with dot should only match literal dot
	result := filterItems(items, "model.v1")
	if len(result) != 1 {
		t.Errorf("expected 1 exact match, got %d", len(result))
	}
	if len(result) > 0 && result[0].Name != "model.v1" {
		t.Errorf("expected 'model.v1', got %s", result[0].Name)
	}

	// Other regex special chars should be literal too
	items2 := []selectItem{
		{Name: "test[0]"},
		{Name: "test0"},
		{Name: "test(1)"},
	}

	result2 := filterItems(items2, "test[0]")
	if len(result2) != 1 || result2[0].Name != "test[0]" {
		t.Errorf("expected only 'test[0]', got %v", result2)
	}
}

// TestMultiSelectState_DuplicateNames documents handling of duplicate item names.
// itemIndex uses name as key - duplicates cause collision. This documents
// the current behavior: the last index for a duplicate name is stored
func TestMultiSelectState_DuplicateNames(t *testing.T) {
	// Duplicate names - this is an edge case that shouldn't happen in practice
	items := []selectItem{
		{Name: "duplicate"},
		{Name: "duplicate"},
		{Name: "unique"},
	}

	s := newMultiSelectState(items, nil)

	// DOCUMENTED BEHAVIOR: itemIndex maps name to LAST index
	// When there are duplicates, only the last occurrence's index is stored
	if s.itemIndex["duplicate"] != 1 {
		t.Errorf("itemIndex should map 'duplicate' to last index (1), got %d", s.itemIndex["duplicate"])
	}

	// Toggle item at highlighted=0 (first "duplicate")
	// Due to name collision, toggleItem uses itemIndex["duplicate"] = 1
	// So it actually toggles the SECOND duplicate item, not the first
	s.toggleItem()

	// This documents the potentially surprising behavior:
	// We toggled at highlighted=0, but itemIndex lookup returned 1
	if !s.checked[1] {
		t.Error("toggle should check index 1 (due to name collision in itemIndex)")
	}
	if s.checked[0] {
		t.Log("Note: index 0 is NOT checked, even though highlighted=0 (name collision behavior)")
	}
}

// TestSelectState_FilterReducesBelowSelection verifies selection resets when filter reduces list.
// Prevents index-out-of-bounds on next keystroke
func TestSelectState_FilterReducesBelowSelection(t *testing.T) {
	items := []selectItem{
		{Name: "apple"},
		{Name: "banana"},
		{Name: "cherry"},
	}

	s := newSelectState(items)
	s.selected = 2 // Select "cherry"

	// Type a filter that removes cherry from results
	s.handleInput(eventChar, 'a') // Filter to "a" - matches "apple" and "banana"

	// Selection should reset to 0
	if s.selected != 0 {
		t.Errorf("expected selected=0 after filter, got %d", s.selected)
	}

	filtered := s.filtered()
	if len(filtered) != 2 {
		t.Errorf("expected 2 filtered items, got %d", len(filtered))
	}
}

// TestFilterItems_UnicodeCharacters verifies filtering works with UTF-8.
// Model names might contain unicode characters
func TestFilterItems_UnicodeCharacters(t *testing.T) {
	items := []selectItem{
		{Name: "llama-日本語"},
		{Name: "模型-chinese"},
		{Name: "émoji-🦙"},
		{Name: "regular-model"},
	}

	t.Run("filter japanese", func(t *testing.T) {
		result := filterItems(items, "日本")
		if len(result) != 1 || result[0].Name != "llama-日本語" {
			t.Errorf("expected llama-日本語, got %v", result)
		}
	})

	t.Run("filter chinese", func(t *testing.T) {
		result := filterItems(items, "模型")
		if len(result) != 1 || result[0].Name != "模型-chinese" {
			t.Errorf("expected 模型-chinese, got %v", result)
		}
	})

	t.Run("filter emoji", func(t *testing.T) {
		result := filterItems(items, "🦙")
		if len(result) != 1 || result[0].Name != "émoji-🦙" {
			t.Errorf("expected émoji-🦙, got %v", result)
		}
	})

	t.Run("filter accented char", func(t *testing.T) {
		result := filterItems(items, "émoji")
		if len(result) != 1 || result[0].Name != "émoji-🦙" {
			t.Errorf("expected émoji-🦙, got %v", result)
		}
	})
}

// TestMultiSelectState_FilterReducesBelowHighlight verifies highlight resets when filter reduces list.
func TestMultiSelectState_FilterReducesBelowHighlight(t *testing.T) {
	items := []selectItem{
		{Name: "apple"},
		{Name: "banana"},
		{Name: "cherry"},
	}

	s := newMultiSelectState(items, nil)
	s.highlighted = 2 // Highlight "cherry"

	// Type a filter that removes cherry
	s.handleInput(eventChar, 'a')

	if s.highlighted != 0 {
		t.Errorf("expected highlighted=0 after filter, got %d", s.highlighted)
	}
}

// TestMultiSelectState_EmptyItems verifies handling of empty item list.
// Empty list should be handled gracefully.
func TestMultiSelectState_EmptyItems(t *testing.T) {
	s := newMultiSelectState([]selectItem{}, nil)

	// Toggle should not panic on empty list
	s.toggleItem()

	if s.selectedCount() != 0 {
		t.Errorf("expected 0 selected for empty list, got %d", s.selectedCount())
	}

	// Render should handle empty list
	var buf bytes.Buffer
	lineCount := renderMultiSelect(&buf, "Select:", s)
	if lineCount == 0 {
		t.Error("renderMultiSelect should produce output even for empty list")
	}
	if !strings.Contains(buf.String(), "no matches") {
		t.Error("expected 'no matches' for empty list")
	}
}

// TestSelectState_RenderWithDescriptions verifies rendering items with descriptions.
func TestSelectState_RenderWithDescriptions(t *testing.T) {
	items := []selectItem{
		{Name: "item1", Description: "First item description"},
		{Name: "item2", Description: ""},
		{Name: "item3", Description: "Third item"},
	}

	s := newSelectState(items)
	var buf bytes.Buffer
	renderSelect(&buf, "Select:", s)

	output := buf.String()
	if !strings.Contains(output, "First item description") {
		t.Error("expected description to be rendered")
	}
	if !strings.Contains(output, "item2") {
		t.Error("expected item without description to be rendered")
	}
}
