import os
import subprocess
import speech_recognition as sr


def _generate_response(model, prompt):
    """
    Function to generate and parse a response from a given LLM to be parsed by Ollama.

    Params:
        model: str
        prompt: str
    """
    command = f'''curl http://localhost:11434/api/generate -d '{{ \"model\": \"{model}\", \"prompt\": \"{prompt}\" }}' | jq -r '.response' | tr -d '\\n' | sed 's/\\. /\\.\\n/g' | say'''
    try:
        subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError:
        pass

def _capture_audio():
    """
    Function to capture audio from the MAC.
    """
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        os.system('clear')
        print('say something...')
        audio = recognizer.listen(source, timeout=5)
    try:
        prompt = recognizer.recognize_google(audio)
        return prompt
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        return ''


def parse(model):
    """
    Function to parse input and output of the LLM.
    """
    try:
        prompt = _capture_audio()
        if prompt == 'quit':
            os.system('clear')
            os._exit(0)
        if not prompt:
            return
        _generate_response(model, prompt)
    except:
        pass


def main():
    os.system('clear')
    fetch = 'git fetch origin'
    merge = 'git merge origin/main'
    subprocess.run(fetch, shell=True, capture_output=True, text=True, check=True)
    subprocess.run(merge, shell=True, capture_output=True, text=True, check=True)
    os.system('clear')
    model = input('model: ')
    try:
        command = f'ollama pull {model}'
        print(command)
        subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError:
        print('no such model')
        os._exit(0)
    while True:
        parse(model)


if __name__ == '__main__':
        main()
