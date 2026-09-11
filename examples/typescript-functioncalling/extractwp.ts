import { Ollama } from "ollama-node";
import { readFile } from "fs/promises";

async function main() {

  const ollama = new Ollama();

  // Set the system prompt to prepare the model to receive a prompt and a schema and set some rules for the output.
  const systemprompt = `You will be given a text along with a prompt and a schema. You will have to extract the information requested in the prompt from the text and generate output in JSON observing the schema provided. If the schema shows a type of integer or number, you must only show a integer for that field. A string should always be a valid string. If a value is unknown, leave it empty. Output the JSON with extra spaces to ensure that it pretty prints.`

  const schema = {
    "people": [{
      "name": {
        "type": "string",
        "description": "Name of the person"
      },
      "title": {
        "type": "string",
        "description": "Title of the person"
      }
    }],
  }

  // Depending on the model chosen, you may be limited by the size of the context window, so limit the context to 2000 words.
  const textcontent = await readFile("./wp.txt", "utf-8").then((text) => text.split(" ").slice(0, 2000).join(" "));

  // Specific instructions for this task
  const prompt = `Review the source text and determine the 10 most important people to focus on. Then extract the name and title for those people. Output should be in JSON.\n\nSchema: \n${JSON.stringify(schema, null, 2)}\n\nSource Text:\n${textcontent}`

  await ollama.setModel("neural-chat");
  ollama.setSystemPrompt(systemprompt);

  // setJSONFormat is the equivalent of setting 'format: json' in the API
  ollama.setJSONFormat(true);
  await ollama.streamingGenerate(prompt, (word) => { process.stdout.write(word) })
}

main();