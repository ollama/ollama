/**
 * Model Selector Component
 * Dropdown to select AI model
 */

'use client'

import { useState, useEffect } from 'react'
import { useChatStore } from '@/store/chatStore'
import { api } from '@/lib/api'
import { ChevronDownIcon, CpuChipIcon } from '@heroicons/react/24/outline'

export default function ModelSelector() {
  const { selectedModel, setSelectedModel } = useChatStore()
  const [models, setModels] = useState<string[]>([])
  const [isOpen, setIsOpen] = useState(false)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadModels()
  }, [])

  const loadModels = async () => {
    try {
      const response = await api.listModels()
      const modelNames = response.models.map((m) => m.name)
      setModels(modelNames)
      
      // Set default if not already set
      if (!selectedModel && modelNames.length > 0) {
        setSelectedModel(modelNames[0])
      }
    } catch (error) {
      console.error('Failed to load models:', error)
      // Fallback models
      setModels(['llama3.2', 'mistral', 'codellama'])
      if (!selectedModel) {
        setSelectedModel('llama3.2')
      }
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center gap-2 text-dark-500">
        <CpuChipIcon className="h-5 w-5" />
        <span className="text-sm">Loading models...</span>
      </div>
    )
  }

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 px-4 py-2 bg-dark-800 hover:bg-dark-700 rounded-lg transition-colors duration-200 border border-dark-700"
      >
        <CpuChipIcon className="h-5 w-5 text-primary-500" />
        <span className="font-medium">{selectedModel}</span>
        <ChevronDownIcon className={`h-4 w-4 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
      </button>

      {isOpen && (
        <>
          <div className="fixed inset-0 z-10" onClick={() => setIsOpen(false)} />
          <div className="absolute top-full mt-2 left-0 z-20 w-64 bg-dark-800 border border-dark-700 rounded-lg shadow-xl overflow-hidden">
            <div className="py-1">
              {models.map((model) => (
                <button
                  key={model}
                  onClick={() => {
                    setSelectedModel(model)
                    setIsOpen(false)
                  }}
                  className={`
                    w-full text-left px-4 py-2 hover:bg-dark-700 transition-colors duration-200
                    ${selectedModel === model ? 'bg-primary-900/30 text-primary-400' : ''}
                  `}
                >
                  {model}
                </button>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  )
}
