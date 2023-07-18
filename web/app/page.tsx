import { AiFillApple } from 'react-icons/ai'

import models from '../../models.json'
import Header from './header'

export default async function Home() {
  return (
    <>
      <Header />
      <main className='flex min-h-screen max-w-6xl flex-col p-20 lg:p-32 items-center mx-auto'>
        <img src='/ollama.png' className='w-16 h-auto' />
        <section className='my-12 text-center'>
          <div className='flex flex-col space-y-2'>
            <h2 className='max-w-[18rem] mx-auto my-2 text-3xl tracking-tight'>Portable large language models</h2>
            <h3 className='max-w-xs mx-auto text-base text-neutral-500'>
              Bundle a model’s weights, configuration, prompts, data and more into self-contained packages that run anywhere.
            </h3>
          </div>
          <div className='mx-auto flex flex-col space-y-4 mt-12'>
            <a href='/download' className='mx-14 bg-black text-white rounded-full px-4 py-2 focus:outline-none cursor-pointer'>
              Download
            </a>
            <p className='text-neutral-500 text-sm'>
            Available for macOS with Apple Silicon <br />
            Windows & Linux support coming soon.
            </p>
          </div>
        </section>
      </main>
    </>
  )
}
