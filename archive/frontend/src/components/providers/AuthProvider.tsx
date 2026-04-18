/**
 * Authentication Provider
 * Initializes Firebase auth and manages auth state
 */

'use client'

import { useEffect } from 'react'
import { useAuthStore } from '@/store/authStore'

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const initialize = useAuthStore((state) => state.initialize)

  useEffect(() => {
    const cleanup = initialize()
    return () => cleanup?.()
  }, [initialize])

  return <>{children}</>
}
