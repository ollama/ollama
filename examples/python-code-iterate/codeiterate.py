import json
import requests
import tempfile
import os
import subprocess
import sys
import atexit
from datetime import datetime

# NOTE: ollama must be running for this to work, start the ollama app or run `ollama serve`
model = "qwen2.5-coder:7b"

messages = []
datetimes = []
starttime = datetime.now()

def exit_handler():
    with open('chatlog.md', 'a+') as wfp:
        wfp.write(f"# Chat: Start Time = {starttime}\n")
        for ind in range(len(messages)):
            msg = messages[ind]
            dt = datetimes[ind]
            wfp.write(f"## {msg['role']} message at {dt}\n")
            wfp.write(f"{msg['content']}\n")

def flush_and_input(prompt):
    if sys.platform == 'win32':
        import msvcrt
        while msvcrt.kbhit():
            msvcrt.getch()
    else:
        import select
        while select.select([sys.stdin.fileno()], [], [], 0)[0]:
            sys.stdin.readline()
    return input(prompt)

def chat(messages):
    r = requests.post(
        "http://127.0.0.1:11434/api/chat",
        json={"model": model, "messages": messages, "stream": True},
	stream=True
    )
    r.raise_for_status()
    output = ""

    for line in r.iter_lines():
        body = json.loads(line)
        if "error" in body:
            raise Exception(body["error"])
        if body.get("done") is False:
            message = body.get("message", "")
            content = message.get("content", "")
            output += content
            # the response streams one token at a time, print that as we receive it
            print(content, end="", flush=True)

        if body.get("done", False):
            message["content"] = output
            datetimes.append(datetime.now())
            return message

def extract_code(message):
    lines = message['content'].splitlines()
    code_array = []
    in_code = False
    for line in lines:
        if in_code:
            if line.strip() == '```':
                code_array.append("\n\n")
                in_code = False
            else:
                code_array.append(line)
        else:
            if line.strip() == '```python':
                in_code = True
    return code_array

def main():
    atexit.register(exit_handler)
    code_array = []
    while True:
        if code_array and (prompt := flush_and_input("Would you like to test the generated code(y/n)? ")) == "y":
            code = '\n'.join(code_array)
            fd, path = tempfile.mkstemp(suffix=".py")
            with os.fdopen(fd, 'w') as f:
                f.write(code)
            if 'CONDA_EXE' in os.environ and 'CONDA_DEFAULT_ENV' in os.environ:
                command = [os.environ['CONDA_EXE'], 'run', '-n', os.environ['CONDA_DEFAULT_ENV'], 'python', path]
            else:
                command = ['python', path]
            process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                    stderr = subprocess.PIPE, text=True)
            stdout, stderr = process.communicate()
            if not process.returncode:
                print(f"Generated program finished without error. stdout={stdout}")
                exit()
            else:
                print(f"Generated program failed with {stderr}. Providing feedback to the language model for further refinement")
                messages.append({"role": "user", "content": f"I'm getting the following error: {stderr}. Please fix and provide the full modified program"})
                datetimes.append(datetime.now())
            os.remove(path)
        else:
            user_input = flush_and_input("Enter a prompt: ")
            if not user_input:
                exit()
            print()
            messages.append({"role": "user", "content": user_input})
            datetimes.append(datetime.now())
        message = chat(messages)
        messages.append(message)
        code_array = extract_code(message)
        print("\n\n")


if __name__ == "__main__":
    main()
