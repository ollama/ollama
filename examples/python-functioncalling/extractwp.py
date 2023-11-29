import requests
import json

model = "orca2"

systemprompt = "You will be given a text along with a prompt and a schema. You will have to extract the information requested in the prompt from the text and generate output in JSON observing the schema provided. If the schema shows a type of integer or number, you must only show a integer for that field. A string should always be a valid string. If a value is unknown, leave it empty. Output the JSON with extra spaces to ensure that it pretty prints."

schema = {
    "people": [
        {
            "name": {"type": "string", "description": "Name of the person"},
            "title": {"type": "string", "description": "Title of the person"},
        }
    ],
}

# Read the content from the file
words = []
with open("wp.txt") as f:
    maxwords = 2000
    count = 0
    lines = f.readlines()
    for line in lines:
        for word in line.split(" "):
            count += 1
            if count > maxwords:
                break
            words.append(word)
content = ' '.join(words)

# Use the text and schema to set the prompt
prompt = f"Review the source text and determine 10 the most important people to focus on. Then extract the name and title for those people. Output should be in JSON.\n\nSchema: {schema}\n\nSource Text:\n{content}"


# Make the actual request to the model
r = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": model,
        "system": systemprompt,
        "prompt": prompt,
        "format": "json",
        "stream": False
    },
)

# Get the response as JSON.
j = json.loads(r.text)

# Return the result.
print(j["response"])

