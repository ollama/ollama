import { NextResponse } from 'next/server'
import semver from 'semver'
import { Octokit } from '@octokit/rest'
import { RequestError } from '@octokit/types'

const octokit = new Octokit()

export async function GET(req: Request) {
  const { searchParams } = new URL(req.url)

  const os = searchParams.get('os') || ''
  const version = searchParams.get('version') || ''

  try {
    const { data } = await octokit.repos.getLatestRelease({
      owner: 'jmorganca',
      repo: 'ollama',
    })

    // todo: get the correct asset for the current arch/os
    const asset = data.assets.find(a => a.name.toLowerCase().includes(os))

    if (!asset) {
      return new Response('up to date', { status: 204 })
    }

    if (semver.lt(version, data.tag_name)) {
      return NextResponse.json({ version: data.tag_name, url: asset.browser_download_url })
    }
  } catch (error) {
    const e = error as RequestError
    if (e.status === 404) {
      return new Response('not found', { status: 404 })
    }

    return new Response('internal server error', { status: 500 })
  }

  return new Response('up to date', { status: 204 })
}
