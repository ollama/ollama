# Using LangChain with Ollama using JavaScript

このチュートリアルでは、LangChain と Ollama を使用して JavaScript を学び、少し新しい情報を得ることを目指します。2023年8月にマウイで発生した一連の山火事について学びます。その時点より前にトレーニングされたLLMは、トレーニングデータにそれよりも新しい情報が含まれていないため、これについて知ることはできません。したがって、[山火事に関するWikipediaの記事](https://en.wikipedia.org/wiki/2023_Hawaii_wildfires)を見つけ、その内容について質問してみましょう。

始めるために、単純な質問をモデルに尋ねるために **LangChain** を使ってみましょう。これを JavaScript で行うためには、**LangChain** をインストールする必要があります：

さて、JavaScript を構築し始めることができます：

```javascript
import { Ollama } from "langchain/llms/ollama";

const ollama = new Ollama({
  baseUrl: "http://localhost:11434",
  model: "llama2",
});

const answer = await ollama.call(`why is the sky blue?`);

console.log(answer);
```

これにより、ターミナルで `ollama run llama2 "why is the sky blue"` を実行したのと同じ結果が得られます。ただし、質問を尋ねるためにウェブからドキュメントを読み込みたいです。**Cheerio** はウェブページを取り込むための優れたライブラリで、**LangChain** では **CheerioWebBaseLoader** で使用されています。そのため、**Cheerio** をインストールしてアプリのその部分を構築しましょう。

```bash
npm install cheerio 
```

```javascript
import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";

const loader = new CheerioWebBaseLoader("https://en.wikipedia.org/wiki/2023_Hawaii_wildfires");
const data = await loader.load();
```
それにより、ドキュメントが読み込まれます。このページは「オデュッセイア」よりは小さいですが、ほとんどのLLMのコンテキストサイズよりも大きいです。したがって、より小さな部分に分割し、質問に関連する部分だけを選択する必要があります。これにはベクトルデータストアが非常に役立ちます。この例では、**LangChain** の一部である **MemoryVectorStore** を使用します。ただし、データストアにコンテンツを取り込むためにはもう1つ必要なものがあります。テキスト内のトークンをベクトルの系列に変換する埋め込みプロセスを実行する必要があります。そしてそのために、**Tensorflow** を使用します。これにはいくつかのステップが含まれています。まず、必要な **Tensorflow** コンポーネントをインストールします。


```javascript
npm install @tensorflow/tfjs-core@3.6.0 @tensorflow/tfjs-converter@3.6.0 @tensorflow-models/universal-sentence-encoder@1.3.3 @tensorflow/tfjs-node@4.10.0
```

もしバージョン番号なしでこれらのコンポーネントをインストールすると、最新バージョンがインストールされますが、**Tensorflow** 内での競合があるため、互換性のあるバージョンをインストールする必要があります。


```javascript
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter"
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import "@tensorflow/tfjs-node";
import { TensorFlowEmbeddings } from "langchain/embeddings/tensorflow";

// テキストを500文字のチャンクに分割し、各チャンクを20文字ずつオーバーラップさせます。
const textSplitter = new RecursiveCharacterTextSplitter({
 chunkSize: 500,
 chunkOverlap: 20
});
const splitDocs = await textSplitter.splitDocuments(data);

// 次に、TensorFlowの埋め込みを使用してこれらのチャンクをデータストアに格納します。
const vectorStore = await MemoryVectorStore.fromDocuments(splitDocs, new TensorFlowEmbeddings());
```

データストアをLLMに対する質問に接続するには、**LangChain** の中心にある概念を使用する必要があります。それが「チェーン」です。チェーンは、特定のタスクを達成するために複数の活動を結びつける方法です。いくつかの種類のチェーンが利用可能ですが、このチュートリアルでは **RetrievalQAChain** を使用します。


```javascript
import { RetrievalQAChain } from "langchain/chains";

const retriever = vectorStore.asRetriever();
const chain = RetrievalQAChain.fromLLM(ollama, retriever);
const result = await chain.call({query: "ハワイの大規模災害宣言の要請はいつ承認されましたか？"});
console.log(result.text)
```

したがって、リトリーバーを作成しました。これはデータストアからクエリに一致するチャンクを返す方法です。そして、その後、リトリーバーとモデルをチェーンを介して接続します。最後に、チェーンにクエリを送信し、文書をソースとして回答を得ます。返された回答は正確で、2023年8月10日でした。

これで、**LangChain** と **Ollama** で何ができるかの簡単な紹介が完了しました。

