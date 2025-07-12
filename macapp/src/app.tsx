import { useState } from 'react'
import copy from 'copy-to-clipboard'
import { CheckIcon, DocumentDuplicateIcon } from '@heroicons/react/24/outline'
import Store from 'electron-store'
import { getCurrentWindow, app } from '@electron/remote'

import { install } from './install'
import OllamaIcon from './ollama.svg'

const store = new Store()

enum Step {
  WELCOME = 0,
  CLI,
  FINISH,
}

export default function () {
  const [step, setStep] = useState<Step>(Step.WELCOME)
  const [commandCopied, setCommandCopied] = useState<boolean>(false)

  const command = 'ollama run llama3.2'

  return (
    <div className='drag'>
      <div className='mx-auto flex min-h-screen w-full flex-col justify-between bg-white px-4 pt-16'>
        {step === Step.WELCOME && (
          <>
            <div className='mx-auto text-center'>
              <h1 className='mb-6 mt-4 text-2xl tracking-tight text-gray-900'>Welcome to Ollama</h1>
              <p className='mx-auto w-[65%] text-sm text-gray-400'>
                Let's get you up and running with your own large language models.
              </p>
              <button
                onClick={() => setStep(Step.CLI)}
                className='no-drag rounded-dm mx-auto my-8 w-[40%] rounded-md bg-black px-4 py-2 text-sm text-white hover:brightness-110'
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
            <div className='mx-auto flex flex-col gap-y-28 text-center'>
              <h1 className='mt-4 text-2xl tracking-tight text-gray-900'>Install the command line</h1>
              <pre className='mx-auto text-4xl text-gray-400'>&gt; ollama</pre>
              <div className='mx-auto'>
                <button
                  onClick={async () => {
                    try {
                      await install()
                      setStep(Step.FINISH)
                    } catch (e) {
                      console.error('could not install: ', e)
                    } finally {
                      getCurrentWindow().show()
                      getCurrentWindow().focus()
                    }
                  }}
                  className='no-drag rounded-dm mx-auto w-[60%] rounded-md bg-black px-4 py-2 text-sm text-white hover:brightness-110'
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
            <div className='mx-auto flex flex-col gap-y-16 text-center'>
              <h1 className='mt-4 text-2xl tracking-tight text-gray-900'>Run your first model</h1>
              <div>
                <div className='flex flex-col'>
                  <div className='group relative flex items-center'>
                    <pre className='language-none text-2xs w-full rounded-md bg-gray-100 px-4 py-3 text-start leading-normal'>
                      {command}
                    </pre>
                    <button
                      className={`no-drag absolute right-[5px] px-2 py-2 ${
                        commandCopied
                          ? 'text-gray-900 opacity-100 hover:cursor-auto'
                          : 'text-gray-200 opacity-50 hover:cursor-pointer'
                      } hover:font-bold hover:text-gray-900 group-hover:opacity-100`}
                      onClick={() => {
                        copy(command)
                        setCommandCopied(true)
                        setTimeout(() => setCommandCopied(false), 3000)
                      }}
                    >
                      {commandCopied ? (
                        <CheckIcon className='h-4 w-4 font-bold text-gray-500' />
                      ) : (
                        <DocumentDuplicateIcon className='h-4 w-4 text-gray-500' />
                      )}
                    </button>
                  </div>
                  <p className='mx-auto my-4 w-[70%] text-xs text-gray-400'>
                    Run this command in your favorite terminal.
                  </p>
                </div>
                <div
                  className='rounded-lg bg-blue-50 p-4 text-start text-sm text-blue-800 dark:bg-gray-800 dark:text-blue-400'
                  role='alert'
                >
                  <span className='font-semibold'>
                    {app.getLoginItemSettings().openAtLogin
                      ? 'Autostart is enabled by default.'
                      : 'Autostart is disabled by default.'}
                  </span>
                  <p>You can modify this setting in the preferences menu.</p>
                </div>
              </div>
              <button
                onClick={() => {
                  store.set('first-time-run', true)
                  window.close()
                }}
                className='no-drag rounded-dm mx-auto w-[60%] rounded-md bg-black px-4 py-2 text-sm text-white hover:brightness-110'
              >
                Finish
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  )
}
