import React from 'react'
import { createRoot } from 'react-dom/client'
import { SettingsProvider } from '../settings/SettingsContext'
import '../app.css'
import '../chat/tokens.css'
import { QuickAskPanel } from './QuickAskPanel'

// Standalone renderer entry for dedicated Quick Ask BrowserWindow
function Root() {
  return (
    <SettingsProvider>
      <QuickAskPanel standalone onClose={() => { window.close() }} />
    </SettingsProvider>
  )
}

const el = document.createElement('div')
el.id = 'root'
document.body.appendChild(el)
createRoot(el).render(<Root />)
