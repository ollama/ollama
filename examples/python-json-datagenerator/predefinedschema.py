import requests
import json
import random

model = "llama3.2"
template = {
  "firstName": "",
  "lastName": "",
  "address": {
    "street": "",
    "city": "",
    "state": "",
    "zipCode": ""
  },
  "phoneNumber": ""
}

prompt = f"generate one realistically believable sample data set of a persons first name, last name, address in the US, and  phone number. \nUse the following template: {json.dumps(template)}."

data = {
    "prompt": prompt,
    "model": model,
    "format": "json",
    "stream": False,
    "options": {"temperature": 2.5, "top_p": 0.99, "top_k": 100},
}

print(f"Generating a sample user")
response = requests.post("http://localhost:11434/api/generate", json=data, stream=False)
json_data = json.loads(response.text)
print(json.dumps(json.loads(json_data["response"]), indent=2))
