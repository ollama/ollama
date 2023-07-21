import Image from 'next/image'
import Link from 'next/link'

import Header from './header'

export default async function Home() {
  return (
    <>
      <Header />
      <main className='flex min-h-screen max-w-6xl flex-col py-20 px-16 md:p-32 items-center mx-auto'>
        <Image src='/ollama.png' width={64} height={64} alt='ollamaIcon' />
        <section className='my-12 text-center'>
          <div className='flex flex-col space-y-2'>
            <h2 className='md:max-w-md mx-auto my-2 text-3xl tracking-tight'>
              Get up and running with large language models, locally.
            </h2>
            <h3 className='md:max-w-xs mx-auto text-base text-neutral-500'>
              Run Llama 2 and other models on macOS. Customize and create your own.
            </h3>
          </div>
          <div className='mx-auto max-w-xs flex flex-col space-y-4 mt-12'>
            <Link
              href='/download'
              className='md:mx-10 lg:mx-14 bg-black text-white rounded-full px-4 py-2 focus:outline-none cursor-pointer'
            >
              Download
            </Link>
            <p className='text-neutral-500 text-sm '>
              Available for macOS with Apple Silicon <br />
              Windows & Linux support coming soon.
            </p>
          </div>
        </section>
      </main>
    </>
  )
}
