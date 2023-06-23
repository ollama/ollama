import { Metadata } from 'next'

import '@/app/globals.css'

export const metadata: Metadata = {
  title: {
    default: 'Keypair',
    template: `%s - Keypair`,
  },
  themeColor: [
    { media: '(prefers-color-scheme: light)', color: 'white' },
    { media: '(prefers-color-scheme: dark)', color: 'black' },
  ],
  icons: {
    icon: '/favicon.ico',
    shortcut: '/favicon-16x16.png',
    apple: '/apple-touch-icon.png',
  },
}

interface RootLayoutProps {
  children: React.ReactNode
}

export default function RootLayout({ children }: RootLayoutProps) {
  return (
    <html lang='en' suppressHydrationWarning>
      <body className='flex min-h-screen w-full bg-white font-sans antialiased'>{children}</body>
    </html>
  )
}
