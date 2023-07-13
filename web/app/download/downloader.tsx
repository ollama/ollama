'use client'

import { useEffect } from 'react'

export default function Downloader({ url }: { url: string }) {
  useEffect(() => {
    window.location.href = url
  }, [])

  return null
}
