import React from 'react'
import { ChatSession } from './persistence'

interface SessionManagerProps {
  sessions: ChatSession[]
  currentSession: ChatSession | null
  onNewSession: () => void
  onSelectSession: (session: ChatSession) => void
  onDeleteSession: (session: ChatSession) => void
  collapsed?: boolean
  onToggle?: () => void
  onRename?: (id: string, title: string) => void
}

function groupSessions(sessions: ChatSession[]): { label: string; items: ChatSession[] }[] {
  const today: ChatSession[] = []
  const thisWeek: ChatSession[] = []
  const older: ChatSession[] = []
  const now = new Date()
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime()
  const startOfWeek = startOfToday - (now.getDay() * 24 * 60 * 60 * 1000)
  sessions.forEach(s => {
    const t = new Date(s.createdAt).getTime()
    if (t >= startOfToday) today.push(s)
    else if (t >= startOfWeek) thisWeek.push(s)
    else older.push(s)
  })
  const groups: { label: string; items: ChatSession[] }[] = []
  if (today.length) groups.push({ label: 'Today', items: today })
  if (thisWeek.length) groups.push({ label: 'This week', items: thisWeek })
  if (older.length) groups.push({ label: 'Older', items: older })
  return groups
}

export const SessionManager: React.FC<SessionManagerProps> = ({
  sessions,
  currentSession,
  onNewSession,
  onSelectSession,
  onDeleteSession,
  collapsed = false,
  onToggle,
  onRename,
}) => {
  const [editingId, setEditingId] = React.useState<string | null>(null)
  const [editingValue, setEditingValue] = React.useState('')
  const [query, setQuery] = React.useState('')
  const inputRef = React.useRef<HTMLInputElement | null>(null)
  React.useEffect(()=>{ if(editingId && inputRef.current){ inputRef.current.focus(); inputRef.current.select() } },[editingId])
  const filtered = query.trim() ? sessions.filter(s => (s.title||'').toLowerCase().includes(query.toLowerCase())) : sessions
  const groups = collapsed ? [] : groupSessions(filtered)
  function commitRename() {
    if (editingId && onRename) {
      const val = editingValue.trim() || 'Untitled'
      onRename(editingId, val)
    }
    setEditingId(null); setEditingValue('')
  }
  function cancelRename() { setEditingId(null); setEditingValue('') }
  const Chevron = ({className='',dir='right'}:{className?:string;dir?:'right'|'left'}) => (
    <svg className={className + ' w-3 h-3'} viewBox='0 0 8 12' fill='none' stroke='currentColor' strokeWidth='1.5'>
      {dir==='right' ? <path d='M2 2l4 4-4 4' strokeLinecap='round' strokeLinejoin='round'/> : <path d='M6 2L2 6l4 4' strokeLinecap='round' strokeLinejoin='round'/>}
    </svg>
  )
  if (collapsed) {
    return (
      <div className='h-full flex flex-col items-center justify-start bg-[#0f0f0f] w-10 select-none transition-[width] duration-150 ease-out border-r border-[#1c1c1c]'>
        <button
          onClick={onToggle}
          title='Expand sidebar'
          className='mt-2 w-8 h-8 flex items-center justify-center rounded hover:bg-[#1e1e1e] text-gray-400 hover:text-gray-200 transition'
        >
          <Chevron dir='right' />
        </button>
      </div>
    )
  }
  return (
    <div className='w-64 bg-[#0f0f0f] flex flex-col h-full text-gray-200 font-[system-ui] text-[12px] transition-[width] duration-150 ease-out border-r border-[#1c1c1c]'>
      <div className='px-3 py-3 flex items-center gap-2 border-b border-[#1c1c1c]'>
        <button
          onClick={onNewSession}
          className='flex-1 bg-gray-200 hover:bg-white text-black text-xs font-medium py-2 rounded-md transition shadow-sm'
        >
          New Chat
        </button>
        <button
          onClick={onToggle}
          title='Collapse sidebar'
          className='w-8 h-8 flex items-center justify-center rounded hover:bg-[#1e1e1e] text-gray-400 hover:text-gray-200 transition'
        >
          <Chevron dir='left' />
        </button>
      </div>
      <div className='px-3 pt-2'>
        <input
          value={query}
          onChange={e=>setQuery(e.target.value)}
          placeholder='Search'
          className='w-full text-[11px] bg-[#141414] border border-[#262626] focus:border-[#3a3a3a] outline-none rounded px-2 py-1 placeholder-gray-500'
        />
      </div>
      <div className='flex-1 overflow-y-auto custom-scroll thin-scroll px-2 pt-3 pb-4 space-y-4'>
        {groups.map(g => (
          <div key={g.label}>
            <div className='px-1 mb-1 uppercase tracking-wide text-[10px] text-gray-500'>{g.label}</div>
            <div className='space-y-[2px]'>
              {g.items.map(s => {
                const active = currentSession?.id === s.id
                const editing = editingId === s.id
                return (
                  <div
                    key={s.id}
                    onClick={() => !editing && onSelectSession(s)}
                    onDoubleClick={() => { setEditingId(s.id); setEditingValue(s.title) }}
                    className={`group relative flex items-center rounded-md px-2 py-2 cursor-pointer transition text-[12px] ${active ? 'bg-[#1e1e1e] text-white shadow-inner' : 'hover:bg-[#1a1a1a] text-gray-300'}`}
                  >
                    {!editing && <span className='truncate pr-6'>{s.title || 'Untitled'}</span>}
                    {editing && (
                      <input
                        ref={inputRef}
                        value={editingValue}
                        onChange={e=>setEditingValue(e.target.value)}
                        onBlur={commitRename}
                        onKeyDown={e=>{ if(e.key==='Enter') commitRename(); else if(e.key==='Escape') { cancelRename(); (e.target as HTMLInputElement).blur() } }}
                        className='flex-1 bg-[#1c1c1c] border border-[#333] rounded px-1 py-[2px] text-[12px] outline-none focus:border-[#555]'
                      />
                    )}
                    {!editing && (
                      <button
                        onClick={(e) => { e.stopPropagation(); onDeleteSession(s) }}
                        className='absolute right-1 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 text-[10px] px-1 py-[1px] rounded bg-[#333] hover:bg-red-600 text-white transition'
                        title='Delete'
                      >
                        ×
                      </button>)
                    }
                  </div>
                )
              })}
            </div>
          </div>
        ))}
        {!groups.length && (
          <div className='text-gray-500 text-[11px] px-1'>No sessions{query ? ' match search.' : ' yet.'}</div>
        )}
      </div>
      <div className='px-3 py-2 border-t border-[#1c1c1c] text-[10px] text-gray-500'>
        {sessions.length} session{sessions.length===1?'':'s'}
      </div>
    </div>
  )
}
