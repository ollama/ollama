import { Ollama } from 'ollama-node';

const mentorCount = 3;
const ollama = new Ollama();

function getMentors(): string[] {
  const mentors = ['Gary Vaynerchuk', 'Kanye West', 'Martha Stewart', 'Neil deGrasse Tyson', 'Owen Wilson', 'Ronald Reagan', 'Donald Trump', 'Barack Obama', 'Jeff Bezos'];
  const chosenMentors: string[] = [];
  for (let i = 0; i < mentorCount; i++) {
    const mentor = mentors[Math.floor(Math.random() * mentors.length)];
    chosenMentors.push(mentor);
    mentors.splice(mentors.indexOf(mentor), 1);
  }
  return chosenMentors;
}

function getMentorFileName(mentor: string): string {
  const model = mentor.toLowerCase().replace(/\s/g, '');
  return `mattw/${model}`;
}

async function getSystemPrompt(mentor: string, isLast: boolean, question: string): Promise<string> {
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
    process.stdout.write(`\n${mentor}: `);
    for await (const chunk of ollama.streamingGenerate(theConversation + `Continue the conversation as if you were ${mentor} on the question "${question}".`)) {
      if (chunk.response) {
        output += chunk.response;
        process.stdout.write(chunk.response);
      } else {
        process.stdout.write('\n');
      }
    }
    theConversation += `${mentor}: ${output}\n\n`
  }
}

main();