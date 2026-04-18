/**
 * Chat Messages Component
 * Displays conversation messages with streaming support
 */

'use client'

import { useEffect, useRef } from 'react'
import { useChatStore } from '@/store/chatStore'
import ReactMarkdown from 'react-markdown'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { vscDarkPlus } from 'react-syntax-highlighter/dist/cjs/styles/prism'
import { UserIcon, SparklesIcon } from '@heroicons/react/24/solid'

export default function ChatMessages() {
  const { messages, isStreaming } = useChatStore()
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages, isStreaming])

  if (messages.length === 0) {
    return (
      <div className="h-full flex items-center justify-center text-dark-500">
        <div className="text-center">
          <SparklesIcon className="h-16 w-16 mx-auto mb-4 opacity-50" />
          <p className="text-lg">Start a conversation</p>
          <p className="text-sm mt-2">Ask me anything about AI, coding, or general knowledge</p>
        </div>
      </div>
    )
  }

  return (
    <div className="h-full overflow-y-auto scrollbar-thin px-4 py-6">
      <div className="max-w-4xl mx-auto space-y-4">
        {messages.map((message, index) => (
          <div
            key={index}
            className={`chat-message ${
              message.role === 'user' ? 'chat-message-user max-w-3xl' : 'chat-message-assistant'
            }`}
          >
            <div className="flex items-start gap-3">
              {/* Avatar */}
              <div
                className={`
                  flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center
                  ${message.role === 'user' ? 'bg-primary-600' : 'bg-dark-700'}
                `}
              >
                {message.role === 'user' ? (
                  <UserIcon className="h-5 w-5 text-white" />
                ) : (
                  <SparklesIcon className="h-5 w-5 text-primary-400" />
                )}
              </div>

              {/* Content */}
              <div className="flex-1 min-w-0">
                <div className="text-xs text-dark-500 mb-1">
                  {message.role === 'user' ? 'You' : 'Ollama AI'}
                </div>
                <div className="markdown-content prose prose-invert max-w-none">
                  <ReactMarkdown
                    components={{
                      code({ node, inline, className, children, ...props }) {
                        const match = /language-(\w+)/.exec(className || '')
                        return !inline && match ? (
                          <SyntaxHighlighter
                            style={vscDarkPlus as any}
                            language={match[1]}
                            PreTag="div"
                            className="rounded-lg"
                            {...props}
                          >
                            {String(children).replace(/\n$/, '')}
                          </SyntaxHighlighter>
                        ) : (
                          <code className={className} {...props}>
                            {children}
                          </code>
                        )
                      },
                    }}
                  >
                    {message.content}
                  </ReactMarkdown>
                </div>
              </div>
            </div>
          </div>
        ))}

        {/* Streaming Indicator */}
        {isStreaming && (
          <div className="chat-message chat-message-assistant">
            <div className="flex items-start gap-3">
              <div className="flex-shrink-0 w-8 h-8 rounded-full bg-dark-700 flex items-center justify-center">
                <SparklesIcon className="h-5 w-5 text-primary-400 animate-pulse" />
              </div>
              <div className="flex-1">
                <div className="text-xs text-dark-500 mb-1">Ollama AI</div>
                <div className="flex gap-1">
                  <span className="animate-pulse">●</span>
                  <span className="animate-pulse delay-100">●</span>
                  <span className="animate-pulse delay-200">●</span>
                </div>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>
    </div>
  )
}
