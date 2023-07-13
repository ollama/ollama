'use client'

import { useEffect } from 'react'

export default function ({ url }: { url: string }) {
  useEffect(() => {
    window.location.href = url
  }, [])

  return null
}
