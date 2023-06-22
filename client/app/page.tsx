'use client'

export default function Home() {
  return (
    <div className='flex min-h-screen flex-col items-center justify-between p-24'>
      hello
      <textarea
        autoFocus
        rows={1}
        className='w-full border border-gray-200 rounded-xl px-5 py-3.5 resize-none text-[15px] shadow-lg shadow-black/5 block mx-4 focus:outline-none'
        onKeyDownCapture={e => {
          if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault() // Prevents the newline character from being inserted
            // Perform your desired action here, such as submitting the form or handling the entered text
            console.log('Enter key pressed!')
          }
        }}
      ></textarea>
    </div>
  )
}
