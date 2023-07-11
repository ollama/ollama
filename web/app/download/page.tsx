import { redirect } from 'next/navigation'

export default async function Download() {
  const res = await fetch('https://api.github.com/repos/jmorganca/ollama/releases', { next: { revalidate: 60 } })
  const data = await res.json()

  if (data.length === 0) {
    return new Response('not found', { status: 404 })
  }

  const latest = data[0]
  const assets = latest.assets || []

  if (assets.length === 0) {
    return new Response('not found', { status: 404 })
  }

  // todo: get the correct asset for the current arch/os
  const asset = assets.find(
    (a: any) => a.name.toLowerCase().includes('darwin') && a.name.toLowerCase().includes('.zip')
  )

  if (!asset) {
    return new Response('not found', { status: 404 })
  }

  if (asset) {
    redirect(asset.browser_download_url)
  }

  return null
}
