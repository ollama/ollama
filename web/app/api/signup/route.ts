import { Analytics } from '@segment/analytics-node'
import { v4 as uuid } from 'uuid'

const analytics = new Analytics({ writeKey: process.env.TELEMETRY_WRITE_KEY || '<empty>' })

export async function POST(req: Request) {
  const { email } = await req.json()

  analytics.identify({
    anonymousId: uuid(),
    traits: {
      email,
    },
  })

  return new Response(null, { status: 200 })
}
