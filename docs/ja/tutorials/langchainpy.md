# Using LangChain with Ollama in Python

さて、私たちが古典文学、例えば **ホメロス** の **『オデュッセイア』** を勉強していると想像しましょう。ネレウスとその家族に関する質問があるかもしれません。その情報を尋ねるために llama2 に質問すると、次のような情報が得られるかもしれません:

> 申し訳ありませんが、私は大規模な言語モデルであり、現実に存在しない個人や家族に関する情報は提供できません。ネレウスは実在の人物やキャラクターではなく、そのため家族や他の個人的な詳細はありません。混乱を招いた場合は申し訳ありません。他にお手伝いできることはありますか？

これは典型的な検閲された回答のようですが、llama2-uncensored でもまずまずの回答が得られます:

> ネレウスはピュロスの伝説的な王であり、アルゴナウタイの一人であるネストールの父でした。彼の母は海のニンフ、クリュメネで、父は海の神ネプチューンでした。

それでは、**LangChain** を Ollama と連携させ、Python を使用して実際の文書、ホメロスの『オデュッセイア』に質問する方法を考えてみましょう。

まず、**Ollama** を使用して **Llama2** モデルから回答を取得できる簡単な質問をしてみましょう。まず、**LangChain** パッケージをインストールする必要があります：

`pip install langchain`

それからモデルを作成し、質問をすることができます：

```python
from langchain.llms import Ollama
ollama = Ollama(base_url='http://localhost:11434',
model="llama2")
print(ollama("なぜ空は青いのか"))
```

モデルと Ollama の基本 URL を定義していることに注意してください。

さて、質問を行うためのドキュメントをロードしてみましょう。私はホメロスの『オデュッセイア』を読み込みますが、これは Project Gutenberg で見つけることができます。**LangChain** の一部である**WebBaseLoader** が必要です。また、私のマシンではこれを動作させるために **bs4** もインストールする必要がありました。したがって、`pip install bs4`を実行してください。

```python
from langchain.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://www.gutenberg.org/files/1727/1727-h/1727-h.htm")
data = loader.load()
```

このファイルはかなり大きいです。序文だけで3000トークンあります。つまり、完全なドキュメントはモデルのコンテキストに収まりません。したがって、それをより小さな部分に分割する必要があります。

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)
```

分割されていますが、関連する分割を見つけて、それらをモデルに送信する必要があります。これを行うには、埋め込みを作成してそれらをベクトルデータベースに保存します。この例では、ベクトルデータベースとして ChromaDB を使用します。埋め込みモデルをインスタンス化するために、Ollama を直接使用できます。`pip install chromadb`

```python
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
oembed = OllamaEmbeddings(base_url="http://localhost:11434", model="nomic-embed-text")
vectorstore = Chroma.from_documents(documents=all_splits, embedding=oembed)
```

それでは、ドキュメントから質問をしてみましょう。**ネレウスは誰で、彼の家族には誰がいますか？** ネレウスはオデュッセイアの登場人物で、答えはテキスト内にあります。

```python
question="ネレウスは誰で、ネレウスの家族には誰がいますか？"
docs = vectorstore.similarity_search(question)
len(docs)
```

これにより、検索に類似したデータのマッチング数が出力されます。

次に、質問と文書の関連部分をモデルに送信して適切な回答を得ることができるかどうかを確認することです。ただし、プロセスの2つの部分を結合しているため、これを「チェーン」と呼びます。これはチェーンを定義する必要があることを意味します：

```python
from langchain.chains import RetrievalQA
qachain=RetrievalQA.from_chain_type(ollama, retriever=vectorstore.as_retriever())
qachain.invoke({"query": question})
```

このチェーンから受け取った回答は以下の通りです：

> ネレウスはホメロスの「オデュッセイア」に登場するキャラクターで、ペネロペの求婚者たちの文脈で言及されています。ネレウスはクロリスの父であり、クロリスはネレウスと結婚し、ネストル、クロミオス、ペリクリュメノス、およびペロを含む複数の子供をもうけました。また、ニーソスの息子であるアンフィノムスもペネロペの求婚者として言及されており、その優れた性格と愉快な会話で知られています。

これは完璧な回答ではありません。なぜなら、実際にはクロリスは「イアソスの息子でミューニュオン・オルコメノスの王であるアンフィオンの末娘であり、ピュロスの女王でした」。

テキスト分割のための chunk_overlap を 20 に更新して再試行しました。すると、はるかに良い回答が得られました：

> ネレウスはホメロスの叙事詩「オデュッセイア」に登場するキャラクターです。彼はイアソスの息子でミューニュオン・オルコメノスの王であるアンフィオンの末娘であるクロリスと結婚しています。ネレウスはクロリスとの間に、ネストル、クロミオス、ペリクリュメノス、およびペロを含む複数の子供がいます。

これがはるかに良い回答です。

