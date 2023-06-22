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
      <head />
      <body className='font-sans antialiased min-h-screen flex'>
        <aside className='w-52 flex-none'></aside>
        <section className='flex-1 bg-white border-l border-gray-300'>
          <header className='sticky top-0 z-50 flex h-16 w-full shrink-0 items-center justify-between px-4 backdrop-blur-xl'></header>
          <section className='flex flex-col flex-1'>{children}</section>
        </section>
      </body>
    </html>
  )
}
