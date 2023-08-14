import json
import requests

# NOTE: ollama must be running for this to work, start the ollama app or run `ollama serve`
model = 'llama2' # TODO: update this for whatever model you wish to use
context = [] # the context stores a conversation history, you can use this to make the model more context aware

def generate(prompt):
    global context
    r = requests.post('http://localhost:11434/api/generate',
                      json={
                          'model': model,
                          'prompt': prompt,
                          'context': context,
                      },
                      stream=True)
    r.raise_for_status()

    for line in r.iter_lines():
        body = json.loads(line)
        response_part = body.get('response', '')
        # the response streams one token at a time, print that as we recieve it
        print(response_part, end='', flush=True)

        if 'error' in body:
            raise Exception(body['error'])

        if body.get('done', False):
            context = body['context']
            return

def main():
    while True:
        user_input = input("Enter a prompt: ")
        print()
        generate(user_input)
        print()

if __name__ == "__main__":
    main()