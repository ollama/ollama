'use client'

import { useState } from 'react'

export default function () {
  const [email, setEmail] = useState('')
  const [success, setSuccess] = useState(false)

  return (
    <form
      onSubmit={async e => {
        e.preventDefault()

        await fetch('/api/signup', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ email }),
        })

        setSuccess(true)
        setEmail('')

        return false
      }}
      className='flex self-stretch flex-col gap-3 h-32'
    >
      <input
        required
        autoFocus
        value={email}
        onChange={e => setEmail(e.target.value)}
        type='email'
        placeholder='your@email.com'
        className='bg-neutral-100 rounded-lg px-4 py-2 focus:outline-none placeholder-neutral-500'
      />
      <input
        type='submit'
        value='Get updates'
        className='bg-black text-white rounded-lg px-4 py-2 focus:outline-none cursor-pointer'
      />
      {success && <p className='text-center text-sm'>You&apos;re signed up for updates</p>}
    </form>
  )
}
