import { Ollama } from 'ollama-node'
import fs from 'fs';
import path from 'path';

async function characterGenerator() {
  const character = process.argv[2];
  console.log(`You are creating a character for ${character}.`);
  const foldername = character.replace(/\s/g, '').toLowerCase();
  const directory = path.join(__dirname, foldername);
  if (!fs.existsSync(directory)) {
    fs.mkdirSync(directory, { recursive: true });
  }

  const ollama = new Ollama();
  ollama.setModel("stablebeluga2:70b-q4_K_M");
  const bio = await ollama.generate(`create a bio of ${character} in a single long paragraph. Instead of saying '${character} is...' or '${character} was...' use language like 'You are...' or 'You were...'. Then create a paragraph describing the speaking mannerisms and style of ${character}. Don't include anything about how ${character} looked or what they sounded like, just focus on the words they said. Instead of saying '${character} would say...' use language like 'You should say...'. If you use quotes, always use single quotes instead of double quotes. If there are any specific words or phrases you used a lot, show how you used them. `);

  const thecontents = `FROM llama3\nSYSTEM """\n${bio.response.replace(/(\r\n|\n|\r)/gm, " ").replace('would', 'should')} All answers to questions should be related back to what you are most known for.\n"""`;

  fs.writeFile(path.join(directory, 'Modelfile'), thecontents, (err: any) => {
    if (err) throw err;
    console.log('The file has been saved!');
  });
}

characterGenerator();
