import Downloader from './downloader'
import Signup from './signup'

export default async function Download() {
  const res = await fetch('https://api.github.com/repos/jmorganca/ollama/releases', { next: { revalidate: 60 } })
  const data = await res.json()

  if (data.length === 0) {
    return null
  }

  const latest = data[0]
  const assets = latest.assets || []

  if (assets.length === 0) {
    return null
  }

  // todo: get the correct asset for the current arch/os
  const asset = assets.find(
    (a: any) => a.name.toLowerCase().includes('darwin') && a.name.toLowerCase().includes('.zip')
  )

  if (!asset) {
    return null
  }

  return (
    <main className='flex min-h-screen max-w-2xl flex-col p-4 lg:p-24 items-center mx-auto'>
      <img src='/ollama.png' className='w-16 h-auto' />
      <section className='my-12 text-center'>
        <h2 className='my-2 max-w-md text-3xl tracking-tight'>Downloading Ollama</h2>
        <h3 className='text-sm text-neutral-500'>
          Problems downloading?{' '}
          <a href={asset.browser_download_url} className='underline'>
            Try again
          </a>
        </h3>
        <Downloader url={asset.browser_download_url} />
      </section>
      <section className='max-w-sm flex flex-col w-full items-center border border-neutral-200 rounded-xl px-8 pt-8 pb-2'>
        <p className='text-lg leading-tight text-center mb-6 max-w-[260px]'>Sign up for updates</p>
        <Signup />
      </section>
    </main>
  )
}
