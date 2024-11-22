# RAG Hallucination Checker using Bespoke-Minicheck

This example allows the user to ask questions related to a document, which can be specified via an article url. Relevant chunks are retreived from the document and given to `llama3.2` as context to answer the question. Then each sentence in the answer is checked against the retrieved chunks using `bespoke-minicheck` to ensure that the answer does not contain hallucinations. 

## Running the Example

1. Ensure `all-minilm` (embedding) `llama3.2` (chat) and `bespoke-minicheck` (check) models installed:

   ```bash
   ollama pull all-minilm
   ollama pull llama3.2
   ollama pull bespoke-minicheck
   ```

2. Install the dependencies.

   ```bash
   pip install -r requirements.txt
   ```

3. Run the example:

   ```bash
   python main.py
   ```

## Expected Output

```text
Enter the URL of an article you want to chat with, or press Enter for default example:

Loaded, chunked, and embedded text from https://www.theverge.com/2024/9/12/24242439/openai-o1-model-reasoning-strawberry-chatgpt.

Enter your question or type quit: Who is the CEO of openai?

Retrieved chunks:
OpenAI is releasing a new model called o1 , the first in a planned series of “ reasoning ” models that have been trained to answer more complex questions , faster than a human can . It ’ s being released alongside o1-mini , a smaller , cheaper version . And yes , if you ’ re steeped in AI rumors : this is , in fact , the extremely hyped Strawberry model . For OpenAI , o1 represents a step toward its broader goal of human-like artificial intelligence .

OpenAI is releasing a new model called o1 , the first in a planned series of “ reasoning ” models that have been trained to answer more complex questions , faster than a human can . It ’ s being released alongside o1-mini , a smaller , cheaper version . And yes , if you ’ re steeped in AI rumors : this is , in fact , the extremely hyped Strawberry model . For OpenAI , o1 represents a step toward its broader goal of human-like artificial intelligence . More practically , it does a better job at writing code and solving multistep problems than previous models . But it ’ s also more expensive and slower to use than GPT-4o . OpenAI is calling this release of o1 a “ preview ” to emphasize how nascent it is . ChatGPT Plus and Team users get access to both o1-preview and o1-mini starting today , while Enterprise and Edu users will get access early next week .

More practically , it does a better job at writing code and solving multistep problems than previous models . But it ’ s also more expensive and slower to use than GPT-4o . OpenAI is calling this release of o1 a “ preview ” to emphasize how nascent it is . ChatGPT Plus and Team users get access to both o1-preview and o1-mini starting today , while Enterprise and Edu users will get access early next week . OpenAI says it plans to bring o1-mini access to all the free users of ChatGPT but hasn ’ t set a release date yet . Developer access to o1 is really expensive : In the API , o1-preview is $ 15 per 1 million input tokens , or chunks of text parsed by the model , and $ 60 per 1 million output tokens . For comparison , GPT-4o costs $ 5 per 1 million input tokens and $ 15 per 1 million output tokens .

OpenAI says it plans to bring o1-mini access to all the free users of ChatGPT but hasn ’ t set a release date yet . Developer access to o1 is really expensive : In the API , o1-preview is $ 15 per 1 million input tokens , or chunks of text parsed by the model , and $ 60 per 1 million output tokens . For comparison , GPT-4o costs $ 5 per 1 million input tokens and $ 15 per 1 million output tokens . The training behind o1 is fundamentally different from its predecessors , OpenAI ’ s research lead , Jerry Tworek , tells me , though the company is being vague about the exact details . He says o1 “ has been trained using a completely new optimization algorithm and a new training dataset specifically tailored for it. ” Image : OpenAI OpenAI taught previous GPT models to mimic patterns from its training data .

LLM Answer:
The text does not mention the CEO of OpenAI. It only discusses the release of a new model called o1 and some details about it, but does not provide information on the company's leadership.

LLM Claim: The text does not mention the CEO of OpenAI.
Is this claim supported by the context according to bespoke-minicheck? Yes

LLM Claim: It only discusses the release of a new model called o1 and some details about it, but does not provide information on the company's leadership.
Is this claim supported by the context according to bespoke-minicheck? No
```

The second claim is unsupported since the text mentions the research lead. 

Another tricky example:

```text

Enter your question or type quit: what sets o1 apart from gpt-4o?

Retrieved chunks: 
OpenAI says it plans to bring o1-mini access to all the free users of ChatGPT but hasn ’ t set a release date yet . Developer access to o1 is really expensive : In the API , o1-preview is $ 15 per 1 million input tokens , or chunks of text parsed by the model , and $ 60 per 1 million output tokens . For comparison , GPT-4o costs $ 5 per 1 million input tokens and $ 15 per 1 million output tokens . The training behind o1 is fundamentally different from its predecessors , OpenAI ’ s research lead , Jerry Tworek , tells me , though the company is being vague about the exact details . He says o1 “ has been trained using a completely new optimization algorithm and a new training dataset specifically tailored for it. ” Image : OpenAI OpenAI taught previous GPT models to mimic patterns from its training data .

He says OpenAI also tested o1 against a qualifying exam for the International Mathematics Olympiad , and while GPT-4o only correctly solved only 13 percent of problems , o1 scored 83 percent . “ We can ’ t say we solved hallucinations ” In online programming contests known as Codeforces competitions , this new model reached the 89th percentile of participants , and OpenAI claims the next update of this model will perform “ similarly to PhD students on challenging benchmark tasks in physics , chemistry and biology. ” At the same time , o1 is not as capable as GPT-4o in a lot of areas . It doesn ’ t do as well on factual knowledge about the world .

More practically , it does a better job at writing code and solving multistep problems than previous models . But it ’ s also more expensive and slower to use than GPT-4o . OpenAI is calling this release of o1 a “ preview ” to emphasize how nascent it is . ChatGPT Plus and Team users get access to both o1-preview and o1-mini starting today , while Enterprise and Edu users will get access early next week . OpenAI says it plans to bring o1-mini access to all the free users of ChatGPT but hasn ’ t set a release date yet . Developer access to o1 is really expensive : In the API , o1-preview is $ 15 per 1 million input tokens , or chunks of text parsed by the model , and $ 60 per 1 million output tokens . For comparison , GPT-4o costs $ 5 per 1 million input tokens and $ 15 per 1 million output tokens .

“ We can ’ t say we solved hallucinations ” In online programming contests known as Codeforces competitions , this new model reached the 89th percentile of participants , and OpenAI claims the next update of this model will perform “ similarly to PhD students on challenging benchmark tasks in physics , chemistry and biology. ” At the same time , o1 is not as capable as GPT-4o in a lot of areas . It doesn ’ t do as well on factual knowledge about the world . It also doesn ’ t have the ability to browse the web or process files and images . Still , the company believes it represents a brand-new class of capabilities . It was named o1 to indicate “ resetting the counter back to 1. ” “ I ’ m gon na be honest : I think we ’ re terrible at naming , traditionally , ” McGrew says .
LLM Answer: According to the text, several things set o1 apart from GPT-4o:

* In online programming contests (Codeforces competitions), o1 scored 83% correct solutions compared to GPT-4o's 13%.
* The training behind o1 is "fundamentally different" from its predecessors, including a completely new optimization algorithm and a new training dataset specifically tailored for it.
* o1 has been shown to perform similarly to PhD students on challenging benchmark tasks in physics, chemistry, and biology, while GPT-4o does not have this level of performance.
* However, o1 is also less capable than GPT-4o in some areas, particularly with regard to factual knowledge about the world.

LLM Claim: According to the text, several things set o1 apart from GPT-4o:

* In online programming contests (Codeforces competitions), o1 scored 83% correct solutions compared to GPT-4o's 13%.
Is this claim supported by the context according to bespoke-minicheck? Yes

LLM Claim: * The training behind o1 is "fundamentally different" from its predecessors, including a completely new optimization algorithm and a new training dataset specifically tailored for it.
Is this claim supported by the context according to bespoke-minicheck? Yes

LLM Claim: * o1 has been shown to perform similarly to PhD students on challenging benchmark tasks in physics, chemistry, and biology, while GPT-4o does not have this level of performance.
Is this claim supported by the context according to bespoke-minicheck? No

LLM Claim: * However, o1 is also less capable than GPT-4o in some areas, particularly with regard to factual knowledge about the world.
Is this claim supported by the context according to bespoke-minicheck? Yes
```

We see that the third claim "* o1 has been shown to perform similarly to PhD students on challenging benchmark tasks in physics, chemistry, and biology, while GPT-4o does not have this level of performance." is not supported by the context. This is because the context only mentions that o1 "is claimed to perform" which is different from "has been shown to perform".
