import { AiFillApple } from 'react-icons/ai'

import models from '../../models.json'

export default async function Home() {
  return (
    <main className='flex min-h-screen max-w-2xl flex-col p-4 lg:p-24'>
      <img src='/ollama.png' className='w-20 h-auto' />
      <section className='my-4'>
        <p className='my-3 max-w-md'>
          <a className='underline' href='https://github.com/jmorganca/ollama'>
            Ollama
          </a>{' '}
          is a tool for running large language models, currently for macOS with Windows and Linux coming soon.
          <br />
          <br />
          <a href='/download' target='_blank'>
            <button className='bg-black text-white text-sm py-2 px-3 rounded-lg flex items-center gap-2'>
              <AiFillApple className='h-auto w-5 relative -top-px' /> Download for macOS
            </button>
          </a>
        </p>
      </section>
      <section className='my-4'>
        <h2 className='mb-4 text-lg'>Example models you can try running:</h2>
        {models.map(m => (
          <div className='my-2 grid font-mono' key={m.name}>
            <code className='py-0.5'>ollama run {m.name}</code>
          </div>
        ))}
      </section>
    </main>
  )
}
