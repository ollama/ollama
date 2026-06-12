import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.tsx'
import AdminApp from './AdminApp.tsx'
import './index.css'

// Simple path-based routing: /admin goes to AdminApp, everything else to App
const isAdminPath = window.location.pathname.startsWith('/admin')

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    {isAdminPath ? <AdminApp /> : <App />}
  </React.StrictMode>,
)
