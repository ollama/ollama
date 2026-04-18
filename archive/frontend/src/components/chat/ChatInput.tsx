/**
 * Chat Input Component
 * Message input with send functionality
 */

'use client'

import { useState, useRef, KeyboardEvent } from 'react'
import { useChatStore } from '@/store/chatStore'
import { PaperAirplaneIcon } from '@heroicons/react/24/solid'
import toast from 'react-hot-toast'

export default function ChatInput() {
  const { sendMessage, isStreaming, error, clearError } = useChatStore()
  const [input, setInput] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const handleSubmit = async () => {
    if (!input.trim() || isStreaming) return

    const message = input.trim()
    setInput('')

    // Reset textarea height
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }

    try {
      await sendMessage(message)
    } catch (error: any) {
      toast.error(error.message || 'Failed to send message')
    }
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value)

    // Auto-resize textarea
    const textarea = e.target
    textarea.style.height = 'auto'
    textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`
  }

  if (error) {
    toast.error(error)
    clearError()
  }

  return (
    <div className="px-4 py-6">
      <div className="max-w-4xl mx-auto">
        <div className="flex gap-3 items-end">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={handleInput}
            onKeyDown={handleKeyDown}
            placeholder="Type your message... (Shift+Enter for new line)"
            disabled={isStreaming}
            className="input-field flex-1 resize-none min-h-[52px] max-h-[200px]"
            rows={1}
          />
          <button
            onClick={handleSubmit}
            disabled={!input.trim() || isStreaming}
            className="btn-primary px-4 py-3 disabled:opacity-50 disabled:cursor-not-allowed"
            title="Send message (Enter)"
          >
            <PaperAirplaneIcon className="h-5 w-5" />
          </button>
        </div>

        <p className="text-xs text-dark-500 mt-2 text-center">
          Ollama Elite AI can make mistakes. Verify important information.
        </p>
      </div>
    </div>
  )
}
