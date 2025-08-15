import requests
import json

class OllamaClient:
    def __init__(self, host='http://localhost:11434'):
        self.host = host

    def chat(self, model, messages, stream=False):
        url = f"{self.host}/api/chat"
        data = {
            "model": model,
            "messages": messages,
            "stream": stream
        }
        response = requests.post(url, json=data)
        response.raise_for_status()
        return response.json()

if __name__ == '__main__':
    # Example usage
    client = OllamaClient()
    messages = [
        {
            "role": "user",
            "content": "Why is the sky blue?"
        }
    ]
    try:
        # Before running this, make sure you have a model running, e.g. `ollama run llama2`
        response = client.chat("llama2", messages)
        print(response['message']['content'])
    except requests.exceptions.ConnectionError as e:
        print(f"Error connecting to Ollama: {e}")
        print("Please make sure Ollama is running and the model is available.")
