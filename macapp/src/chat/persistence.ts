import Store from 'electron-store'
import { ChatMessage } from '../providers/ChatProvider'

interface PersistShape {
  sessions: ChatSession[]
  activeSessionId?: string
}

export interface ChatSession {
  id: string
  title: string
  createdAt: string
  updatedAt: string
  messages: ChatMessage[]
}

const store = new Store<PersistShape>({ name: 'chat-sessions' })

export function loadSessions(): ChatSession[] {
  return store.get('sessions') || []
}

export function saveSessions(sessions: ChatSession[]) {
  store.set('sessions', sessions)
}

export function upsertSession(session: ChatSession) {
  const sessions = loadSessions()
  const idx = sessions.findIndex(s => s.id === session.id)
  if (idx >= 0) {
    sessions[idx] = session
  } else {
    sessions.unshift(session) // Add to the beginning
  }
  saveSessions(sessions)
}

export function deleteSession(sessionId: string) {
  let sessions = loadSessions()
  sessions = sessions.filter(s => s.id !== sessionId)
  saveSessions(sessions)
}
