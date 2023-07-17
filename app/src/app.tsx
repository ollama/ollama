import { useState } from 'react'
import copy from 'copy-to-clipboard'
import { exec as cbExec } from 'child_process'
import * as path from 'path'
import * as fs from 'fs'
import { DocumentDuplicateIcon } from '@heroicons/react/24/outline'
import { app } from '@electron/remote'
import OllamaIcon from './ollama.svg'
import { promisify } from 'util'

const ollama = app.isPackaged ? path.join(process.resourcesPath, 'ollama') : path.resolve(process.cwd(), '..', 'ollama')
const exec = promisify(cbExec)

async function installCLI() {
  const symlinkPath = '/usr/local/bin/ollama'

  if (fs.existsSync(symlinkPath) && fs.readlinkSync(symlinkPath) === ollama) {
    return
  }

  const command = `do shell script "ln -F -s ${ollama} /usr/local/bin/ollama" with administrator privileges`

  try {
    await exec(`osascript -e '${command}'`)
  } catch (error) {
    console.error(`cli: failed to install cli: ${error.message}`)
    return
  }
}

enum Step {
  WELCOME = 0,
  CLI,
  FINISH,
}

export default function () {
  const [step, setStep] = useState<Step>(Step.WELCOME)

  const command = 'ollama run orca'

  return (
    <div className='drag mx-auto flex min-h-screen w-full flex-col justify-between bg-white px-4 pt-16'>
      {step === Step.WELCOME && (
        <>
          <div className='mx-auto text-center'>
            <h1 className='mb-6 mt-4 text-2xl tracking-tight text-gray-900'>Welcome to Ollama</h1>
            <p className='mx-auto w-[65%] text-sm text-gray-400'>
              Let's get you up and running with your own large language models.
            </p>
            <button
              onClick={() => {
                setStep(1)
              }}
              className='rounded-dm mx-auto my-8 w-[40%] rounded-md bg-black px-4 py-2 text-sm text-white hover:brightness-110'
            >
              Next
            </button>
          </div>
          <div className='mx-auto'>
            <OllamaIcon />
          </div>
        </>
      )}
      {step === Step.CLI && (
        <>
          <div className='mx-auto flex flex-col space-y-28 text-center'>
            <h1 className='mt-4 text-2xl tracking-tight text-gray-900'>Install the command line</h1>
            <pre className='mx-auto text-4xl text-gray-400'>&gt; ollama</pre>
            <div className='mx-auto'>
              <button
                onClick={async () => {
                  await installCLI()
                  window.focus()
                  setStep(2)
                }}
                className='rounded-dm mx-auto w-[60%] rounded-md bg-black px-4 py-2 text-sm text-white hover:brightness-110'
              >
                Install
              </button>
              <p className='mx-auto my-4 w-[70%] text-xs text-gray-400'>
                You will be prompted for administrator access
              </p>
            </div>
          </div>
        </>
      )}
      {step === Step.FINISH && (
        <>
          <div className='mx-auto flex flex-col space-y-20 text-center'>
            <h1 className='mt-4 text-2xl tracking-tight text-gray-900'>Run your first model</h1>
            <div className='flex flex-col'>
              <div className='group relative flex items-center'>
                <pre className='language-none text-2xs w-full rounded-md bg-gray-100 px-4 py-3 text-start leading-normal'>
                  {command}
                </pre>
                <button
                  className='absolute right-[5px] rounded-md border bg-white/90 px-2 py-2 text-gray-400 opacity-0 backdrop-blur-xl hover:text-gray-600 group-hover:opacity-100'
                  onClick={() => {
                    copy(command)
                  }}
                >
                  <DocumentDuplicateIcon className='h-4 w-4 text-gray-500' />
                </button>
              </div>
              <p className='mx-auto my-4 w-[70%] text-xs text-gray-400'>Run this command in your favorite terminal.</p>
            </div>
            <button
              onClick={() => {
                window.close()
              }}
              className='rounded-dm mx-auto w-[60%] rounded-md bg-black px-4 py-2 text-sm text-white hover:brightness-110'
            >
              Finish
            </button>
          </div>
        </>
      )}
    </div>
  )
}
