/**
 * Chat Sidebar Component
 * Conversation list and management
 */

'use client'

import { useState } from 'react'
import { useChatStore } from '@/store/chatStore'
import { PlusIcon, Bars3Icon, XMarkIcon, ChatBubbleLeftIcon } from '@heroicons/react/24/outline'

interface ChatSidebarProps {
  isOpen: boolean
  onToggle: () => void
}

export default function ChatSidebar({ isOpen, onToggle }: ChatSidebarProps) {
  const { conversations, currentConversation, selectConversation, createConversation } = useChatStore()
  const [newChatTitle, setNewChatTitle] = useState('')
  const [showNewChatInput, setShowNewChatInput] = useState(false)

  const handleCreateConversation = async () => {
    if (!newChatTitle.trim()) return

    await createConversation(newChatTitle)
    setNewChatTitle('')
    setShowNewChatInput(false)
  }

  return (
    <>
      {/* Mobile Overlay */}
      {isOpen && (
        <div
          className="lg:hidden fixed inset-0 bg-black/50 z-40"
          onClick={onToggle}
        />
      )}

      {/* Sidebar */}
      <aside
        className={`
          ${isOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
          fixed lg:relative z-50 h-full w-72 bg-dark-900 border-r border-dark-800
          transition-transform duration-300 ease-in-out flex flex-col
        `}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-dark-800">
          <h2 className="text-lg font-semibold">Conversations</h2>
          <button onClick={onToggle} className="lg:hidden p-2 hover:bg-dark-800 rounded-lg">
            <XMarkIcon className="h-5 w-5" />
          </button>
        </div>

        {/* New Chat Button */}
        <div className="p-4">
          {!showNewChatInput ? (
            <button
              onClick={() => setShowNewChatInput(true)}
              className="w-full btn-primary flex items-center justify-center gap-2"
            >
              <PlusIcon className="h-5 w-5" />
              New Chat
            </button>
          ) : (
            <div className="flex gap-2">
              <input
                type="text"
                value={newChatTitle}
                onChange={(e) => setNewChatTitle(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleCreateConversation()}
                placeholder="Chat title..."
                className="input-field flex-1 py-2 text-sm"
                autoFocus
              />
              <button
                onClick={handleCreateConversation}
                disabled={!newChatTitle.trim()}
                className="btn-primary px-3"
              >
                <PlusIcon className="h-5 w-5" />
              </button>
              <button
                onClick={() => {
                  setShowNewChatInput(false)
                  setNewChatTitle('')
                }}
                className="btn-secondary px-3"
              >
                <XMarkIcon className="h-5 w-5" />
              </button>
            </div>
          )}
        </div>

        {/* Conversation List */}
        <div className="flex-1 overflow-y-auto scrollbar-thin p-2">
          {conversations.length === 0 ? (
            <div className="text-center text-dark-500 py-8">
              <ChatBubbleLeftIcon className="h-12 w-12 mx-auto mb-2 opacity-50" />
              <p className="text-sm">No conversations yet</p>
            </div>
          ) : (
            conversations.map((conv) => (
              <button
                key={conv.id}
                onClick={() => selectConversation(conv.id)}
                className={`
                  w-full text-left p-3 rounded-lg mb-2 transition-colors duration-200
                  ${
                    currentConversation === conv.id
                      ? 'bg-primary-900/30 border border-primary-700/50'
                      : 'hover:bg-dark-800 border border-transparent'
                  }
                `}
              >
                <h3 className="font-medium text-sm truncate">{conv.title}</h3>
                <p className="text-xs text-dark-500 mt-1">
                  {new Date(conv.created_at).toLocaleDateString()}
                </p>
              </button>
            ))
          )}
        </div>
      </aside>

      {/* Mobile Toggle Button */}
      {!isOpen && (
        <button
          onClick={onToggle}
          className="lg:hidden fixed bottom-4 left-4 z-30 p-3 bg-primary-600 hover:bg-primary-700 rounded-full shadow-lg"
        >
          <Bars3Icon className="h-6 w-6 text-white" />
        </button>
      )}
    </>
  )
}
