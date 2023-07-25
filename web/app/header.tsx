import Link from "next/link"

const navigation = [
  { name: 'Discord', href: 'https://discord.gg/MrfB5FbNWN' },
  { name: 'GitHub', href: 'https://github.com/jmorganca/ollama' },
  { name: 'Download', href: '/download' },
]

export default function Header() {  
  return (
    <header className="absolute inset-x-0 top-0 z-50">
      <nav className="mx-auto flex items-center justify-between px-10 py-4">        
        <Link className="flex-1 font-bold" href="/">
          Ollama
        </Link>
        <div className="flex space-x-8">
          {navigation.map((item) => (
            <Link key={item.name} href={item.href} className="text-sm leading-6 text-gray-900">
              {item.name}
            </Link>
          ))}
        </div>
      </nav>
    </header>
  )
}