import { Ollama } from 'langchain/llms/ollama';
import * as readline from "readline";

async function main() {
  const ollama = new Ollama({
    model: 'mistral'    
    // other parameters can be found at https://js.langchain.com/docs/api/llms_ollama/classes/Ollama
  });

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  rl.question("What is your question: \n", async (user_input) => {
    const stream = await ollama.stream(user_input);
  
    for await (const chunk of stream) {
      process.stdout.write(chunk);
    }
    rl.close();
  })
}

main();