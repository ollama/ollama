# Using LangChain with Ollama using JavaScript

In this tutorial, we are going to use JavaScript with LangChain and Ollama to learn about something just a touch more recent. In August 2023, there was a series of wildfires on Maui. There is no way an LLM trained before that time can know about this, since their training data would not include anything as recent as that. So we can find the [Wikipedia article about the fires](https://en.wikipedia.org/wiki/2023_Hawaii_wildfires) and ask questions about the contents.

To get started, let's just use **LangChain** to ask a simple question to a model. To do this with JavaScript, we need to install **LangChain**:

```bash
npm install @langchain/community
```

Now we can start building out our JavaScript:

```javascript
import { Ollama } from "@langchain/community/llms/ollama";

const ollama = new Ollama({
  baseUrl: "http://localhost:11434",
  model: "llama3.2",
});

const answer = await ollama.invoke(`why is the sky blue?`);

console.log(answer);
```

That will get us the same thing as if we ran `ollama run llama3.2 "why is the sky blue"` in the terminal. But we want to load a document from the web to ask a question against. **Cheerio** is a great library for ingesting a webpage, and **LangChain** uses it in their **CheerioWebBaseLoader**. So let's install **Cheerio** and build that part of the app.

```bash
npm install cheerio
```

```javascript
import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";

const loader = new CheerioWebBaseLoader("https://en.wikipedia.org/wiki/2023_Hawaii_wildfires");
const data = await loader.load();
```

That will load the document. Although this page is smaller than the Odyssey, it is certainly bigger than the context size for most LLMs. So we are going to need to split into smaller pieces, and then select just the pieces relevant to our question. This is a great use for a vector datastore. In this example, we will use the **MemoryVectorStore** that is part of **LangChain**. But there is one more thing we need to get the content into the datastore. We have to run an embeddings process that converts the tokens in the text into a series of vectors. And for that, we are going to use **Tensorflow**. There is a lot of stuff going on in this one. First, install the **Tensorflow** components that we need.

```javascript
npm install @tensorflow/tfjs-core@3.6.0 @tensorflow/tfjs-converter@3.6.0 @tensorflow-models/universal-sentence-encoder@1.3.3 @tensorflow/tfjs-node@4.10.0
```

If you just install those components without the version numbers, it will install the latest versions, but there are conflicts within **Tensorflow**, so you need to install the compatible versions.

```javascript
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter"
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import "@tensorflow/tfjs-node";
import { TensorFlowEmbeddings } from "langchain/embeddings/tensorflow";

// Split the text into 500 character chunks. And overlap each chunk by 20 characters
const textSplitter = new RecursiveCharacterTextSplitter({
 chunkSize: 500,
 chunkOverlap: 20
});
const splitDocs = await textSplitter.splitDocuments(data);

// Then use the TensorFlow Embedding to store these chunks in the datastore
const vectorStore = await MemoryVectorStore.fromDocuments(splitDocs, new TensorFlowEmbeddings());
```

To connect the datastore to a question asked to a LLM, we need to use the concept at the heart of **LangChain**: the chain. Chains are a way to connect a number of activities together to accomplish a particular tasks. There are a number of chain types available, but for this tutorial we are using the **RetrievalQAChain**.

```javascript
import { RetrievalQAChain } from "langchain/chains";

const retriever = vectorStore.asRetriever();
const chain = RetrievalQAChain.fromLLM(ollama, retriever);
const result = await chain.call({query: "When was Hawaii's request for a major disaster declaration approved?"});
console.log(result.text)
```

So we created a retriever, which is a way to return the chunks that match a query from a datastore. And then connect the retriever and the model via a chain. Finally, we send a query to the chain, which results in an answer using our document as a source. The answer it returned was correct, August 10, 2023.

And that is a simple introduction to what you can do with **LangChain** and **Ollama.**
