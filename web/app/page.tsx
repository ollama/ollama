import models from '../../models.json'

export default async function Home() {
  return (
    <main className='flex min-h-screen max-w-2xl flex-col p-4 lg:p-24'>
      <h1 className='font-serif text-3xl'>ollama</h1>
      <section className='my-8'>
        <p className='my-3 mb-8 max-w-md'>
          <a className='underline' href='https://github.com/jmorganca/ollama'>
            Ollama
          </a>{' '}
          is a tool for running large language models.
          <br />
          <br />
          Get started with Ollama using pip:
        </p>
        <pre className='my-4'>
          <code>pip install ollama</code>
        </pre>
      </section>
      <section className='my-4'>
        <h2 className='mb-4 text-lg'>Example models you can try running:</h2>
        {models.map(m => (
          <div className='my-2 grid font-mono' key={m.name}>
            <code className='py-0.5'>ollama run {m.name}</code>
          </div>
        ))}
      </section>
    </main>
  )
}
