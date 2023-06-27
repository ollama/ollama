import { useState } from 'react'
import path from 'path'
import os from 'os'
import { dialog, getCurrentWindow } from '@electron/remote'

const API_URL = 'http://127.0.0.1:5001'

type Message = {
  sender: 'bot' | 'human'
  content: string
}

const userInfo = os.userInfo()

async function generate(prompt: string, model: string, callback: (res: string) => void) {
  const result = await fetch(`${API_URL}/generate`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      prompt,
      model,
    }),
  })

  if (!result.ok) {
    return
  }

  let reader = result.body.getReader()

  while (true) {
    const { done, value } = await reader.read()

    if (done) {
      break
    }

    let decoder = new TextDecoder()
    let str = decoder.decode(value)
    let re = /}{/g
    str = '[' + str.replace(re, '},{') + ']'
    let messages = JSON.parse(str)

    for (const message of messages) {
      const choice = message.choices[0]
      if (choice.finish_reason === 'stop') {
        break
      }

      callback(choice.text)
    }
  }

  return
}

export default function () {
  const [prompt, setPrompt] = useState('')
  const [messages, setMessages] = useState<Message[]>([])
  const [model, setModel] = useState('')
  const [generating, setGenerating] = useState(false)

  return (
    <div className='flex min-h-screen flex-1 flex-col justify-between bg-white'>
      <header className='drag sticky top-0 z-50 flex h-14 w-full flex-row items-center border-b border-black/10 bg-white/75 backdrop-blur-md'>
        <div className='mx-auto w-full max-w-xl leading-none'>
          <h1 className='text-sm font-medium'>{path.basename(model).replace('.bin', '')}</h1>
        </div>
      </header>
      {model ? (
        <section className='mx-auto mb-10 w-full max-w-xl flex-1 break-words'>
          {messages.map((m, i) => (
            <div className='my-4 flex gap-4' key={i}>
              <div className='flex-none pr-1 text-lg'>
                {m.sender === 'human' ? (
                  <div className='mt-px flex h-6 w-6 items-center justify-center rounded-md bg-neutral-200 text-sm text-neutral-700'>
                    {userInfo.username[0].toUpperCase()}
                  </div>
                ) : (
                  <div className='mt-0.5 flex h-6 w-6 items-center justify-center rounded-md bg-blue-600 text-sm text-white'>
                    {path.basename(model)[0].toUpperCase()}
                  </div>
                )}
              </div>
              <div className='flex-1 text-gray-800'>
                {m.content}
                {m.sender === 'bot' && generating && (
                  <span className='blink relative -top-[3px] left-1 text-[10px]'>█</span>
                )}
              </div>
            </div>
          ))}
        </section>
      ) : (
        <section className='flex flex-1 select-none flex-col items-center justify-center pb-20'>
          <h2 className='text-3xl font-light text-neutral-400'>No model selected</h2>
          <button
            onClick={async () => {
              const res = await dialog.showOpenDialog(getCurrentWindow(), {
                properties: ['openFile', 'multiSelections'],
              })
              if (res.canceled) {
                return
              }

              setModel(res.filePaths[0])
            }}
            className='rounded-dm my-8 rounded-md bg-blue-600 px-4 py-2 text-sm text-white hover:brightness-110'
          >
            Open file...
          </button>
        </section>
      )}
      <div className='sticky bottom-0 bg-gradient-to-b from-transparent to-white'>
        {model && (
          <textarea
            autoFocus
            rows={1}
            value={prompt}
            placeholder='Send a message...'
            onChange={e => setPrompt(e.target.value)}
            className='mx-auto my-4 block w-full max-w-xl resize-none rounded-xl border border-gray-200 px-5 py-3.5 text-[15px] shadow-lg shadow-black/5 focus:outline-none'
            onKeyDownCapture={async e => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()

                if (generating) {
                  return
                }

                if (!prompt) {
                  return
                }

                await setMessages(messages => {
                  return [...messages, { sender: 'human', content: prompt }, { sender: 'bot', content: '' }]
                })

                setPrompt('')

                setGenerating(true)
                await generate(prompt, model, res => {
                  setMessages(messages => {
                    let last = messages[messages.length - 1]
                    return [...messages.slice(0, messages.length - 1), { ...last, content: last.content + res }]
                  })
                })
                setGenerating(false)
              }
            }}
          ></textarea>
        )}
      </div>
    </div>
  )
}
