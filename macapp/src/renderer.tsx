import App from './app'
import './app.css'
import { createRoot } from 'react-dom/client'
import { SettingsProvider } from './settings/SettingsContext'

try {
	console.log('[renderer] boot script start')
	const container = document.getElementById('app')
	if (!container) {
		console.error('[renderer] #app container not found')
	} else {
		const root = createRoot(container)
		root.render(<SettingsProvider><App /></SettingsProvider>)
		console.log('[renderer] render invoked')
	}
} catch (e) {
	console.error('[renderer] boot error', e)
}
