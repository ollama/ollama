/**
 * Main Chat Page
 * Full-featured chat interface with streaming support
 */

'use client'

import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import { useAuthStore } from '@/store/authStore'
import { useChatStore } from '@/store/chatStore'
import ChatSidebar from '@/components/chat/ChatSidebar'
import ChatMessages from '@/components/chat/ChatMessages'
import ChatInput from '@/components/chat/ChatInput'
import ModelSelector from '@/components/chat/ModelSelector'
import Header from '@/components/layout/Header'
import LoadingSpinner from '@/components/ui/LoadingSpinner'

export default function ChatPage() {
  const router = useRouter()
  const { user, loading: authLoading } = useAuthStore()
  const { loadConversations } = useChatStore()
  const [sidebarOpen, setSidebarOpen] = useState(true)

  useEffect(() => {
    if (!authLoading && !user) {
      router.push('/')
    }
  }, [user, authLoading, router])

  useEffect(() => {
    if (user) {
      loadConversations()
    }
  }, [user, loadConversations])

  if (authLoading) {
    return (
      <div className="h-full flex items-center justify-center">
        <LoadingSpinner size="large" />
      </div>
    )
  }

  if (!user) {
    return null
  }

  return (
    <div className="h-full flex flex-col bg-dark-950">
      <Header />
      
      <div className="flex-1 flex overflow-hidden">
        {/* Sidebar */}
        <ChatSidebar isOpen={sidebarOpen} onToggle={() => setSidebarOpen(!sidebarOpen)} />

        {/* Main Chat Area */}
        <div className="flex-1 flex flex-col">
          {/* Model Selector */}
          <div className="border-b border-dark-800 px-6 py-4">
            <ModelSelector />
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-hidden">
            <ChatMessages />
          </div>

          {/* Input */}
          <div className="border-t border-dark-800">
            <ChatInput />
          </div>
        </div>
      </div>
    </div>
  )
}
