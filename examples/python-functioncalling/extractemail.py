import requests
import json

model = "openchat"


def reportEvents(name, date, location):
    nameString = name if name else "an event"
    dateString = f" on {date}" if date else ""
    locationString = f" at {location}" if location else ""
    print(f"You have an event: {nameString}{dateString}{locationString}")


def reportAddresses(address):
    for field in address:
        if field == "city":
            city = address["city"]
            state = f", {address['state']}" if address["state"] else ""
            zip = f" {address['zip']}"
            print(f"{city}{state}{zip}\n")
            break
        else:
            print(address[field])


systemPrompt = "You will be given a text along with a prompt and a schema. You will have to extract the information requested in the prompt from the text and generate output in JSON observing the schema provided. If the schema shows a type of integer or number, you must only show a integer for that field. A string should always be a valid string. If a value is unknown, leave it empty. Output the JSON with extra spaces to ensure that it pretty prints."

schema = {
    "eventsQuantity": {
        "type": "integer",
        "description": "The number of events in the source text",
    },
    "addressesQuantity": {
        "type": "integer",
        "description": "The number of addresses in the source text",
    },
    "events": [
        {
            "name": {"type": "string", "description": "Name of the event"},
            "date": {"type": "string", "description": "Date of the event"},
            "location": {"type": "string", "description": "Location of the event"},
            "extraInfo": {
                "type": "string",
                "description": "Any extra information that is provided about the event.",
            },
        }
    ],
    "people": [
        {
            "name": {"type": "string", "description": "Name of the person"},
            "company": {
                "type": "string",
                "description": "Name of the company where they work",
            },
            "street": {
                "type": "string",
                "description": "Street address of the person or company. This is only the street name and the numerical address. Do not include city, state, or zip of the address in this field.",
            },
            "city": {
                "type": "string",
                "description": "City portion of the address of the person or company",
            },
            "state": {
                "type": "string",
                "description": "State portion of the address of the person or company",
            },
            "zip": {
                "type": "string",
                "description": "Zip code of the person or company",
            },
            "extraInfo": {
                "type": "string",
                "description": "Any extra information that is provided about the location.",
            },
        }
    ],
}

with open("emails.txt") as f:
    content=f.read()

prompt = f"The source text is a series of emails that have been put into a single file. They are separated by three dashes. Review the source text and determine the full address of the person sending each of the emails as well as any events that we need to track. If they provide a company address use that. If any extra info is provided, such as a description of the place, or a floor, add it to extraInfo. The first field in the address JSON is quantity of events and should be set to the number of events tracked and the second field should be set to the number of addresses tracked in the file. Don't stuff an event into the output that isn't an event. Only add data to the mostly appropriate field. Don't make up fields that aren't in the schema. If there isn't a value for a field, use null. Output should be in JSON.\n\nSchema: \n{schema}\n\nSource Text:\n{content}"


r = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": model,
        "system": systemPrompt,
        "prompt": prompt,
        "format": "json",
        "stream": False,
    },
)

j = json.loads(r.text)

output = json.loads(j["response"])
events = output["events"]
addresses = output["people"]

print(f"Here are your {output['eventsQuantity']} events:")
for event in events:
    reportEvents(event["name"], event["date"], event["location"])

print(f"\n\nHere are your {output['addressesQuantity']} addresses")
for address in addresses:
    reportAddresses(address)
