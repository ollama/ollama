import './globals.css'

export const metadata = {
  title: 'Ollama',
  description: 'A tool for running large language models',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang='en'>
      <body className='antialiased'>{children}</body>
    </html>
  )
}
