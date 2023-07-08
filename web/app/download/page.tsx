import { Octokit } from '@octokit/rest'
import { redirect } from 'next/navigation'

const octokit = new Octokit()

export const revalidate = 60

export default async function Download() {
  const { data } = await octokit.repos.getLatestRelease({
    owner: 'jmorganca',
    repo: 'ollama',
  })

  // todo: get the correct asset for the current arch/os
  const asset = data.assets.find(a => a.name.toLowerCase().includes('darwin') && a.name.toLowerCase().includes('.zip'))

  if (asset) {
    redirect(asset.browser_download_url)
  }

  return null
}
