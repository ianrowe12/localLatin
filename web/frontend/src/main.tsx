import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './index.css'

async function boot() {
  if (import.meta.env.VITE_USE_MOCKS === 'true') {
    const { installMockHandler } = await import('./mock/handler')
    installMockHandler()
    console.log('[LocalLatin] Mock API handler installed')
  }

  ReactDOM.createRoot(document.getElementById('root')!).render(
    <React.StrictMode>
      <App />
    </React.StrictMode>,
  )
}

boot()
