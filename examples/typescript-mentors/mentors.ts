import { Ollama } from 'ollama-node';

const mentorCount = 3;
const ollama = new Ollama();
type Mentor = { ns: string, char: string };

function getMentors(): Mentor[] {
  const mentors = [{ ns: 'mattw', char: 'Gary Vaynerchuk' }, { ns: 'mattw', char: 'Kanye West'}, {ns: 'mattw', char: 'Martha Stewart'}, {ns: 'mattw', char: 'Neil deGrasse Tyson'}, {ns: 'mattw', char: 'Owen Wilson'}, {ns: 'mattw', char: 'Ronald Reagan'}, {ns: 'mattw', char: 'Donald Trump'}, {ns: 'mattw', char: 'Barack Obama'}, {ns: 'mattw', char: 'Jeff Bezos'}];
  const chosenMentors: Mentor[] = [];
  for (let i = 0; i < mentorCount; i++) {
    const mentor = mentors[Math.floor(Math.random() * mentors.length)];
    chosenMentors.push(mentor);
    mentors.splice(mentors.indexOf(mentor), 1);
  }
  return chosenMentors;
}

function getMentorFileName(mentor: Mentor): string {
  const model = mentor.char.toLowerCase().replace(/\s/g, '');
  return `${mentor.ns}/${model}`;
}

async function getSystemPrompt(mentor: Mentor, isLast: boolean, question: string): Promise<string> {
  ollama.setModel(getMentorFileName(mentor));
  const info = await ollama.showModelInfo()
  let SystemPrompt = info.system || '';
  SystemPrompt += ` You should continue the conversation as if you were ${mentor} and acknowledge the people before you in the conversation. You should adopt their mannerisms and tone, but also not use language they wouldn't use. If they are not known to know about the concept in the question, don't offer an answer. Your answer should be no longer than 1 paragraph. And definitely try not to sound like anyone else. Don't repeat any slang or phrases already used. And if it is a question the original ${mentor} wouldn't have know the answer to, just say that you don't know, in the style of ${mentor}. And think about the time the person lived. Don't use terminology that they wouldn't have used.`

  if (isLast) {
    SystemPrompt += ` End your answer with something like I hope our answers help you out`;
  } else {
    SystemPrompt += ` Remember, this is a conversation, so you don't need a conclusion, but end your answer with a question related to the first question: "${question}".`;
  }
  return SystemPrompt;
}

async function main() {
  const mentors = getMentors();
  const question = process.argv[2];
  let theConversation = `Here is the conversation so far.\nYou: ${question}\n`

  for await (const mentor of mentors) {
    const SystemPrompt = await getSystemPrompt(mentor, mentor === mentors[mentorCount - 1], question);
    ollama.setModel(getMentorFileName(mentor));
    ollama.setSystemPrompt(SystemPrompt);
    let output = '';
    process.stdout.write(`\n${mentor.char}: `);
    for await (const chunk of ollama.streamingGenerate(theConversation + `Continue the conversation as if you were ${mentor.char} on the question "${question}".`)) {
      if (chunk.response) {
        output += chunk.response;
        process.stdout.write(chunk.response);
      } else {
        process.stdout.write('\n');
      }
    }
    theConversation += `${mentor.char}: ${output}\n\n`
  }
}

main();