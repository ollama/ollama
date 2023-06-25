import { useState } from 'react'

const API_URL = 'http://127.0.0.1:5001'

type Message = {
  sender: string
  content: string
}

async function completion(prompt: string, callback: (res: string) => void) {
  const result = await fetch(`${API_URL}/generate`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      prompt: `A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.

      ### Human: Hello, Assistant.
      ### Assistant: Hello. How may I help you today?
      ### Human: ${prompt}`,
      model: 'ggml-model-q4_0',
    }),
  })

  if (!result.ok || !result.body) {
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

  return (
    <div className='flex min-h-screen flex-1 flex-col justify-between bg-white'>
      <header className='drag sticky top-0 z-50 flex w-full flex-row items-center border-b border-black/5 bg-gray-50/75 p-3 backdrop-blur-md'>
        <div className='mx-auto w-full max-w-xl leading-none'>
          <h1 className='text-sm font-medium'>LLaMa</h1>
          <h2 className='text-xs text-black/50'>Meta Platforms, Inc.</h2>
        </div>
      </header>
      <section className='mx-auto mb-10 w-full max-w-xl flex-1 break-words'>
        {messages.map((m, i) => (
          <div className='my-4 flex gap-4' key={i}>
            <div className='flex-none pr-1 text-lg'>
              {m.sender === 'human' ? (
                <div className='bg-neutral-200 text-neutral-700 text-sm h-6 w-6 rounded-md flex items-center justify-center mt-px'>
                  H
                </div>
              ) : (
                <div className='bg-blue-600 text-white text-sm h-6 w-6 rounded-md flex items-center justify-center mt-0.5'>
                  L
                </div>
              )}
            </div>
            <div className='flex-1 text-gray-800'>
              {m.content}
              {m.sender === 'bot' && <span className='relative -top-[3px] left-1 text-[10px]'>⬤</span>}
            </div>
          </div>
        ))}
      </section>
      <div className='sticky bottom-0 bg-gradient-to-b from-transparent to-white'>
        <textarea
          autoFocus
          rows={1}
          value={prompt}
          placeholder='Send a message...'
          onChange={e => setPrompt(e.target.value)}
          className='mx-auto my-4 block w-full max-w-xl resize-none rounded-xl border border-gray-200 px-5 py-3.5 text-[15px] shadow-lg shadow-black/5 focus:outline-none'
          onKeyDownCapture={async e => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault() // Prevents the newline character from being inserted
              // Perform your desired action here, such as submitting the form or handling the entered text

              await setMessages(messages => {
                return [...messages, { sender: 'human', content: prompt }]
              })

              const index = messages.length + 1
              completion(prompt, res => {
                setMessages(messages => {
                  let message = messages[index]
                  if (!message) {
                    message = { sender: 'bot', content: '' }
                  }

                  message.content = message.content + res

                  return [...messages.slice(0, index), message]
                })
              })

              setPrompt('')
            }
          }}
        ></textarea>
      </div>
    </div>
  )
}
