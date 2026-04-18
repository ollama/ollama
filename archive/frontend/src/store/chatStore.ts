/**
 * Chat State Management
 * Manages chat messages, conversations, and streaming
 */

import { create } from 'zustand'
import { api, ChatMessage } from '@/lib/api'

interface Conversation {
  id: string
  title: string
  created_at: string
}

interface ChatState {
  conversations: Conversation[]
  currentConversation: string | null
  messages: ChatMessage[]
  isStreaming: boolean
  selectedModel: string
  error: string | null

  // Actions
  loadConversations: () => Promise<void>
  createConversation: (title: string) => Promise<void>
  selectConversation: (id: string) => Promise<void>
  sendMessage: (content: string) => Promise<void>
  setSelectedModel: (model: string) => void
  clearError: () => void
}

export const useChatStore = create<ChatState>((set, get) => ({
  conversations: [],
  currentConversation: null,
  messages: [],
  isStreaming: false,
  selectedModel: 'llama3.2',
  error: null,

  loadConversations: async () => {
    try {
      const response = await api.getConversations()
      set({ conversations: response.conversations || [] })
    } catch (error: any) {
      set({ error: error.message })
    }
  },

  createConversation: async (title: string) => {
    try {
      const conversation = await api.createConversation(title)
      set((state) => ({
        conversations: [conversation, ...state.conversations],
        currentConversation: conversation.id,
        messages: [],
      }))
    } catch (error: any) {
      set({ error: error.message })
    }
  },

  selectConversation: async (id: string) => {
    try {
      const response = await api.getMessages(id)
      set({
        currentConversation: id,
        messages: response.messages || [],
      })
    } catch (error: any) {
      set({ error: error.message })
    }
  },

  sendMessage: async (content: string) => {
    const { messages, selectedModel } = get()
    
    // Add user message
    const userMessage: ChatMessage = { role: 'user', content }
    set((state) => ({
      messages: [...state.messages, userMessage],
      isStreaming: true,
    }))

    try {
      // Start streaming response
      let assistantContent = ''
      const assistantMessage: ChatMessage = {
        role: 'assistant',
        content: '',
      }

      // Add empty assistant message that we'll update
      set((state) => ({
        messages: [...state.messages, assistantMessage],
      }))

      const request = {
        model: selectedModel,
        messages: [...messages, userMessage],
        stream: true,
      }

      for await (const chunk of api.streamChat(request)) {
        assistantContent += chunk
        
        // Update the last message (assistant) with new content
        set((state) => {
          const newMessages = [...state.messages]
          newMessages[newMessages.length - 1] = {
            role: 'assistant',
            content: assistantContent,
          }
          return { messages: newMessages }
        })
      }

      set({ isStreaming: false })
    } catch (error: any) {
      set({ error: error.message, isStreaming: false })
      
      // Remove failed assistant message
      set((state) => ({
        messages: state.messages.slice(0, -1),
      }))
    }
  },

  setSelectedModel: (model: string) => set({ selectedModel: model }),
  
  clearError: () => set({ error: null }),
}))
