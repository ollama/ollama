/**
 * Header Component
 * Top navigation with user menu
 */

'use client'

import { useState } from 'react'
import { useAuthStore } from '@/store/authStore'
import { useRouter } from 'next/navigation'
import { SparklesIcon, ArrowRightOnRectangleIcon, UserCircleIcon } from '@heroicons/react/24/outline'
import toast from 'react-hot-toast'

export default function Header() {
  const router = useRouter()
  const { user, signOut } = useAuthStore()
  const [menuOpen, setMenuOpen] = useState(false)

  const handleSignOut = async () => {
    try {
      await signOut()
      toast.success('Signed out successfully')
      router.push('/')
    } catch (error: any) {
      toast.error(error.message || 'Failed to sign out')
    }
  }

  return (
    <header className="bg-dark-900 border-b border-dark-800">
      <div className="px-6 py-4 flex items-center justify-between">
        {/* Logo */}
        <div className="flex items-center gap-3">
          <SparklesIcon className="h-8 w-8 text-primary-500" />
          <div>
            <h1 className="text-xl font-bold">Ollama Elite AI</h1>
            <p className="text-xs text-dark-500">elevatediq.ai/ollama</p>
          </div>
        </div>

        {/* User Menu */}
        <div className="relative">
          <button
            onClick={() => setMenuOpen(!menuOpen)}
            className="flex items-center gap-2 px-3 py-2 hover:bg-dark-800 rounded-lg transition-colors duration-200"
          >
            {user?.photoURL ? (
              <img
                src={user.photoURL}
                alt={user.displayName || 'User'}
                className="h-8 w-8 rounded-full"
              />
            ) : (
              <UserCircleIcon className="h-8 w-8 text-dark-500" />
            )}
            <span className="text-sm font-medium hidden sm:block">
              {user?.displayName || user?.email}
            </span>
          </button>

          {menuOpen && (
            <>
              <div className="fixed inset-0 z-10" onClick={() => setMenuOpen(false)} />
              <div className="absolute right-0 mt-2 w-48 bg-dark-800 border border-dark-700 rounded-lg shadow-xl z-20 overflow-hidden">
                <div className="py-1">
                  <button
                    onClick={handleSignOut}
                    className="w-full text-left px-4 py-2 hover:bg-dark-700 transition-colors duration-200 flex items-center gap-2 text-red-400"
                  >
                    <ArrowRightOnRectangleIcon className="h-5 w-5" />
                    Sign Out
                  </button>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </header>
  )
}
