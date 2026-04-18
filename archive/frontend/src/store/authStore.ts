/**
 * Authentication State Management
 * Manages user authentication state using Zustand
 */

import { create } from 'zustand'
import { User } from 'firebase/auth'
import { signInWithGoogle, signOutUser, onAuthStateChange } from '@/lib/firebase'

interface AuthState {
  user: User | null
  loading: boolean
  error: string | null
  
  // Actions
  signIn: () => Promise<void>
  signOut: () => Promise<void>
  initialize: () => void
  clearError: () => void
}

export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  loading: true,
  error: null,

  signIn: async () => {
    set({ loading: true, error: null })
    try {
      const user = await signInWithGoogle()
      set({ user, loading: false })
    } catch (error: any) {
      set({ error: error.message, loading: false })
      throw error
    }
  },

  signOut: async () => {
    set({ loading: true, error: null })
    try {
      await signOutUser()
      set({ user: null, loading: false })
    } catch (error: any) {
      set({ error: error.message, loading: false })
      throw error
    }
  },

  initialize: () => {
    const unsubscribe = onAuthStateChange((user) => {
      set({ user, loading: false })
    })

    // Return cleanup function
    return () => unsubscribe()
  },

  clearError: () => set({ error: null }),
}))
