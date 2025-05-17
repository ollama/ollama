package common

import (
	"unicode/utf8"

	"github.com/ollama/ollama/llm"
)

// ResponseSequence interface defines the common operations for sending and managing responses in a sequence.
type ResponseSequence interface {
	// Pending returns the current pending responses
	Pending() []llm.CompletionResponse

	// Clear clears the pending responses
	Clear()

	// Send attempts to send a response, returns false if sending should stop
	Send(resp llm.CompletionResponse) bool
}

func FlushPending(seq ResponseSequence) bool {
	pending := seq.Pending()
	if len(pending) == 0 {
		return true
	}

	// We need to find if any response has Done=true
	hasDoneResponse := false
	doneIndex := -1
	for i, resp := range pending {
		if resp.Done {
			hasDoneResponse = true
			doneIndex = i
			break
		}
	}

	// Handle responses one by one
	var pendingUTF8 string // Buffer for incomplete UTF-8 sequences

	for i, resp := range pending {
		currentResp := resp

		// Combine any pending UTF-8 with current content
		combinedContent := pendingUTF8 + currentResp.Content
		pendingUTF8 = ""

		// Check if there are any partial UTF-8 characters remaining.
		// We already check and queue as we are generating but some may
		// still make it here:
		// - Sequence is ending, e.g. generation limit has been hit
		// - Invalid characters in the middle of a string
		// This is a stricter check to ensure we never output invalid Unicode.
		if utf8.ValidString(combinedContent) {
			currentResp.Content = combinedContent
		} else {
			// Content has invalid UTF-8
			// If this is the last response or the response with Done=true, trim it
			if i == len(pending)-1 || (hasDoneResponse && i == doneIndex) {
				// Trim incomplete UTF-8 characters
				trimmedContent := combinedContent
				for !utf8.ValidString(trimmedContent) && len(trimmedContent) > 0 {
					trimmedContent = trimmedContent[:len(trimmedContent)-1]
				}
				currentResp.Content = trimmedContent
			} else {
				// Not the last response or Done response - keep incomplete UTF-8 for the next response
				// Determine valid and invalid parts
				validPrefix := combinedContent
				for !utf8.ValidString(validPrefix) && len(validPrefix) > 0 {
					validPrefix = validPrefix[:len(validPrefix)-1]
				}

				// If we have a valid prefix, send it
				if len(validPrefix) > 0 {
					currentResp.Content = validPrefix
					// The remainder becomes pending UTF-8
					pendingUTF8 = combinedContent[len(validPrefix):]
				} else {
					// No valid prefix, all content is invalid UTF-8
					pendingUTF8 = combinedContent
					// Skip sending this response
					continue
				}
			}
		}

		if !seq.Send(currentResp) {
			return false
		}
	}

	// Clear pending responses
	seq.Clear()
	return true
}
