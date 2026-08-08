# Ask the Mentors

This example demonstrates how one would create a set of 'mentors' you can have a conversation with. The mentors are generated using the `character-generator.ts` file. This will use **Stable Beluga 70b** to create a bio and list of verbal ticks and common phrases used by each person. Then `mentors.ts` will take a question, and choose three of the 'mentors' and start a conversation with them. Occasionally, they will talk to each other, and other times they will just deliver a set of monologues. It's fun to see what they do and say.

## Usage

1. Add llama3 to have the mentors ask your questions:

   ```bash
   ollama pull llama3
   ```

2. Install prerequisites:

   ```bash
   npm install
   ```

3. Ask a question:

   ```bash
   npm start "what is a jackalope"
   ```

You can also add your own character to be chosen at random when you ask a question.

1. Make sure you have the right model installed:

   ```bash
   ollama pull stablebeluga2:70b-q4_K_M
   ```

2. Create a new character:

   ```bash
   npm run charactergen "Lorne Greene"
   ```

   You can choose any well-known person you like. This example will create `lornegreene/Modelfile`.

3. Now you can create a model with this command:

   ```bash
   ollama create <username>/lornegreene -f lornegreene/Modelfile
   ```

   `username` is whatever name you set up when you signed up at [https://ollama.com/signup](https://ollama.com/signup).

4. To add this to your mentors, you will have to update the code as follows. On line 8 of `mentors.ts`, add an object to the array, replacing `<username>` with the username you used above.

   ```bash
   {ns: "<username>", char: "Lorne Greene"}
   ```

## Review the Code

There are two scripts you can run in this example. The first is the main script to ask the mentors a question. The other one lets you generate a character to add to the mentors. Both scripts are mostly about adjusting the prompts at each inference stage.

### mentors.ts

In the **main** function, it starts by generating a list of mentors. This chooses 3 from a list of interesting characters. Then we ask for a question, and then things get interesting. We set the prompt for each of the 3 mentors a little differently. And the 2nd and 3rd mentors see what the previous folks said. The other functions in mentors sets the prompts for each mentor.

### character-generator.ts

**Character Generator** simply customizes the prompt to build a character profile for any famous person. And most of the script is just tweaking the prompt. This uses Stable Beluga 2 70b parameters. The 70b models tend to do better writing a bio about a character than smaller models, and Stable Beluga seemed to do better than Llama 2. Since this is used at development time for the characters, it doesn't affect the runtime of asking the mentors for their input.
