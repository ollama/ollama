import { Analytics } from '@segment/analytics-node'
import { v4 as uuidv4 } from 'uuid'
import Store from 'electron-store'

const store = new Store()

export const analytics = new Analytics({ writeKey: process.env.TELEMETRY_WRITE_KEY || '<empty>' })

export function id(): string {
  const id = store.get('id') as string

  if (id) {
    return id
  }

  const uuid = uuidv4()
  store.set('id', uuid)
  return uuid
}
