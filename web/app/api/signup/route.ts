import { Analytics } from '@segment/analytics-node'
import { v4 as uuid } from 'uuid'

const analytics = new Analytics({ writeKey: process.env.TELEMETRY_WRITE_KEY || '<empty>' })

export async function POST(req: Request) {
  const { email } = await req.json()

  const id = uuid()

  await analytics.identify({
    anonymousId: id,
    traits: {
      email,
    },
  })

  await analytics.track({
    anonymousId: id,
    event: 'signup',
    properties: {
      email,
    },
  })

  return new Response(null, { status: 200 })
}
