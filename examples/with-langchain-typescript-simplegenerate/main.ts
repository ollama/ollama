import { Ollama} from 'langchain/llms/ollama';

async function main() {
  const ollama = new Ollama({
    model: 'mistral'    
    // other parameters can be found at https://js.langchain.com/docs/api/llms_ollama/classes/Ollama
  })
  const stream = await ollama.stream("Hello");

  for await (const chunk of stream) {
    process.stdout.write(chunk);
  }
}

main();