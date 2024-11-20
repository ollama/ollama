import requests
import json
import random

countries = [
    "United States",
    "United Kingdom",
    "the Netherlands",
    "Germany",
    "Mexico",
    "Canada",
    "France",
]
country = random.choice(countries)
model = "llama3.2"

prompt = f"generate one realistically believable sample data set of a persons first name, last name, address in {country}, and phone number. Do not use common names. Respond using JSON. Key names should have no backslashes, values should use plain ascii with no special characters."

data = {
    "prompt": prompt,
    "model": model,
    "format": "json",
    "stream": False,
    "options": {"temperature": 2.5, "top_p": 0.99, "top_k": 100},
}

print(f"Generating a sample user in {country}")
response = requests.post("http://localhost:11434/api/generate", json=data, stream=False)
json_data = json.loads(response.text)

print(json.dumps(json.loads(json_data["response"]), indent=2))
