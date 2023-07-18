import Header from '../header'
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
    <>
      <Header />
      <main className='flex min-h-screen max-w-6xl flex-col p-20 lg:p-32 items-center mx-auto'>
        <img src='/ollama.png' className='w-16 h-auto' />
        <section className='mt-12 mb-8 text-center'>
          <h2 className='my-2 max-w-md text-3xl tracking-tight'>Downloading...</h2>
          <h3 className='text-base text-neutral-500 mt-12 max-w-[16rem]'>
            While Ollama downloads, sign up to get notified of new updates.
          </h3>
          {/* <Downloader url={asset.browser_download_url} /> */}
        </section>
        <Signup />
      </main>
    </>
  )
}
