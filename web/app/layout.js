import './globals.css'

export const metadata = {
  title: 'Ollama',
  description: 'A tool for running large language models',
}

export default function RootLayout({ children }) {
  return (
    <html lang='en'>
      <body>{children}</body>
    </html>
  )
}
