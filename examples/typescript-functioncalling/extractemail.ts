import { Ollama } from "ollama-node";
import { readFile } from "fs/promises";

// function to be called on events
function reportEvents(name: string, date: string, location: string) {
  const nameString = name ? `${name}` : `an event`;
  const dateString = date ? ` on ${date}` : ``;
  const locationString = location ? ` at ${location}` : ``;
  console.log(`You have an event: ${nameString}${dateString}${locationString}`)
}

// function to be called on addresses
function reportAddresses(address) {
  for (const field in address) {
    if (address[field]) {
      if (field === "city") {
        const city = address.city;
        const state = address.state ? `, ${address.state}` : '';
        const zip = address.zip ? ` ${address.zip}` : '';
        console.log(`${city}${state}${zip}`);
        break;
      } else {
        console.log(`${address[field]}`);
      }
    }
  }
  console.log(``);
}

async function main() {

  const ollama = new Ollama();

  const systemprompt = `You will be given a text along with a prompt and a schema. You will have to extract the information requested in the prompt from the text and generate output in JSON observing the schema provided. If the schema shows a type of integer or number, you must only show a integer for that field. A string should always be a valid string. If a value is unknown, leave it empty. Output the JSON with extra spaces to ensure that it pretty prints.`

  const schema = {
    "eventsQuantity": {
      "type": "integer",
      "description": "The number of events in the source text"
    },
    "addressesQuantity": {
      "type": "integer",
      "description": "The number of addresses in the source text"
    },
    "events": [{
      name: {
        "type": "string",
        description: "Name of the event"
      },
      "date": {
        "type": "string",
        "description": "Date of the event"
      },
      "location": {
        "type": "string",
        "description": "Location of the event"
      },
      "extraInfo": {
        "type": "string",
        "description": "Any extra information that is provided about the event."
      }
    }],
    "people": [{
      "name": {
        "type": "string",
        "description": "Name of the person"
      },
      "company": {
        "type": "string",
        "description": "Name of the company where they work"
      },
      "street": {
        "type": "string",
        "description": "Street address of the person or company. This is only the street name and the numerical address. Do not include city, state, or zip of the address in this field."
      },
      "city": {
        "type": "string",
        "description": "City portion of the address of the person or company"
      },
      "state": {
        "type": "string",
        "description": "State portion of the address of the person or company"
      },
      "zip": {
        "type": "string",
        "description": "Zip code of the person or company"
      },
      "extraInfo": {
        "type": "string",
        "description": "Any extra information that is provided about the location."
      }
    }]
  }

  const textcontent = await readFile("./info.txt", "utf-8").then((text) => text.split(" ").slice(0, 2000).join(" "));

  const prompt = `The source text is a series of emails that have been put into a single file. They are separated by three dashes. Review the source text and determine the full address of the person sending each of the emails as well as any events that we need to track. If they provide a company address use that. If any extra info is provided, such as a description of the place, or a floor, add it to extraInfo. The first field in the address JSON is quantity of events and should be set to the number of events tracked and the second field should be set to the number of addresses tracked in the file. Don't stuff an event into the output that isn't an event. Only add data to the mostly appropriate field. Don't make up fields that aren't in the schema. If there isn't a value for a field, use null. Output should be in JSON.\n\nSchema: \n${JSON.stringify(schema, null, 2)}\n\nSource Text:\n${textcontent}`

  await ollama.setModel("neural-chat");
  ollama.setSystemPrompt(systemprompt);
  ollama.setJSONFormat(true);
  const data = await ollama.generate(prompt);
  const output = JSON.parse(data.output);
  const events = output.events;
  const addresses = output.people;

  console.log(`Here are your ${output.eventsQuantity} events:`);
  for (const event of events) {
    reportEvents(event.name, event.date, event.location);
  }

  console.log(`\n\nHere are your ${output.addressesQuantity} addresses:`);
  for (const address of addresses) {
    reportAddresses(address);
  }
}

main();