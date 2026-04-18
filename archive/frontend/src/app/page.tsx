/**
 * Landing Page with Authentication
 */

'use client'

import { useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { useAuthStore } from '@/store/authStore'
import toast from 'react-hot-toast'
import { SparklesIcon, ChatBubbleLeftRightIcon, DocumentTextIcon, CpuChipIcon } from '@heroicons/react/24/outline'

export default function HomePage() {
  const router = useRouter()
  const { user, signIn, loading } = useAuthStore()

  useEffect(() => {
    if (user) {
      router.push('/chat')
    }
  }, [user, router])

  const handleSignIn = async () => {
    try {
      await signIn()
      toast.success('Successfully signed in!')
    } catch (error: any) {
      toast.error(error.message || 'Failed to sign in')
    }
  }

  if (loading) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="animate-pulse-slow text-primary-500">
          <SparklesIcon className="h-16 w-16" />
        </div>
      </div>
    )
  }

  return (
    <div className="h-full flex flex-col bg-gradient-to-br from-dark-950 via-dark-900 to-dark-950">
      {/* Hero Section */}
      <div className="flex-1 flex items-center justify-center px-4">
        <div className="max-w-4xl w-full text-center">
          {/* Logo */}
          <div className="flex items-center justify-center mb-8">
            <SparklesIcon className="h-16 w-16 text-primary-500" />
          </div>

          {/* Title */}
          <h1 className="text-6xl font-bold mb-6 bg-gradient-to-r from-primary-400 to-primary-600 bg-clip-text text-transparent">
            Ollama Elite AI
          </h1>

          <p className="text-xl text-dark-300 mb-12 max-w-2xl mx-auto">
            Production-grade local AI infrastructure for building, deploying, and monitoring large language models.
            Enterprise reliability, zero cloud dependencies.
          </p>

          {/* Sign In Button */}
          <button
            onClick={handleSignIn}
            className="btn-primary text-lg px-8 py-4 rounded-xl shadow-xl hover:shadow-2xl hover:scale-105 transform transition-all duration-200"
          >
            Sign in with Google
          </button>

          {/* Features */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mt-20">
            <div className="card hover:border-primary-700 transition-colors duration-200">
              <ChatBubbleLeftRightIcon className="h-12 w-12 text-primary-500 mb-4 mx-auto" />
              <h3 className="text-xl font-semibold mb-2">Real-time Chat</h3>
              <p className="text-dark-400">
                Stream responses from LLMs with full conversation history and context management
              </p>
            </div>

            <div className="card hover:border-primary-700 transition-colors duration-200">
              <DocumentTextIcon className="h-12 w-12 text-primary-500 mb-4 mx-auto" />
              <h3 className="text-xl font-semibold mb-2">Document Processing</h3>
              <p className="text-dark-400">
                Upload, process, and embed documents for RAG-powered AI interactions
              </p>
            </div>

            <div className="card hover:border-primary-700 transition-colors duration-200">
              <CpuChipIcon className="h-12 w-12 text-primary-500 mb-4 mx-auto" />
              <h3 className="text-xl font-semibold mb-2">Model Management</h3>
              <p className="text-dark-400">
                Deploy and manage multiple LLMs with optimized inference and caching
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="py-8 text-center text-dark-500 border-t border-dark-800">
        <p>© 2026 Ollama Elite AI • Powered by elevatediq.ai</p>
      </footer>
    </div>
  )
}
