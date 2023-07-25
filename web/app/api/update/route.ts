import { NextResponse } from 'next/server'
import semver from 'semver'

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url)

  const os = searchParams.get('os') || 'darwin'
  const version = searchParams.get('version') || '0.0.0'

  if (!version) {
    return new Response('not found', { status: 404 })
  }

  const res = await fetch('https://api.github.com/repos/jmorganca/ollama/releases', { next: { revalidate: 60 } })
  const data = await res.json()

  const latest = data?.filter((f: any) => !f.prerelease)?.[0]

  if (!latest) {
    return new Response('not found', { status: 404 })
  }

  const assets = latest.assets || []

  if (assets.length === 0) {
    return new Response('not found', { status: 404 })
  }

  // todo: get the correct asset for the current arch/os
  const asset = assets.find((a: any) => a.name.toLowerCase().includes(os) && a.name.toLowerCase().includes('.zip'))

  if (!asset) {
    return new Response('not found', { status: 404 })
  }

  console.log(asset)

  if (semver.lt(version, latest.tag_name)) {
    return NextResponse.json({ version: data.tag_name, url: asset.browser_download_url })
  }

  return new Response(null, { status: 204 })
}
