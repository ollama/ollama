# OpenAI の互換性

> **注意:** OpenAI の互換性は実験的であり、大規模な変更や互換性のない変更が加えられる可能性があります。Ollama API のすべての機能を使用するには、Ollama の[Python ライブラリ](https://github.com/ollama/ollama-python)、[JavaScript ライブラリ](https://github.com/ollama/ollama-js)、および [REST API](https://github.com/jmorganca/ollama/blob/main/docs/api.md) を参照してください。

Ollama は、既存のアプリケーションを Ollama に接続するのに役立つよう、[OpenAI API](https://platform.openai.com/docs/api-reference) の一部との実験的な互換性を提供します。

## 使用法

### OpenAI Python ライブラリ

```python
from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:11434/v1/',

    # 必須ですが無視されます
    api_key='ollama',
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            'role': 'user',
            'content': 'これはテストですと言う',
        }
    ],
    model='llama2',
)
```

### OpenAI JavaScript ライブラリ

```javascript
import OpenAI from 'openai'

const openai = new OpenAI({
  baseURL: 'http://localhost:11434/v1/',

  // 必須ですが無視されます
  apiKey: 'ollama',
})

const chatCompletion = await openai.chat.completions.create({
  messages: [{ role: 'user', content: 'これはテストですと言う' }],
  model: 'llama2',
})
```

### `curl`

```
curl http://localhost:11434/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "llama2",
        "messages": [
            {
                "role": "system",
                "content": "あなたは役に立つアシスタントです。"
            },
            {
                "role": "user",
                "content": "こんにちは！"
            }
        ]
    }'
```

## エンドポイント

### `/v1/chat/completions`

#### サポートされている機能

- [x] チャット補完
- [x] ストリーミング
- [x] JSON モード
- [x] 再現可能な出力
- [ ] ビジョン
- [ ] 関数の呼び出し
- [ ] ログプロブ

#### サポートされているリクエストフィールド

- [x] `model`
- [x] `messages`
  - [x] テキスト `content`
  - [ ] `content` の部分の配列
- [x] `frequency_penalty`
- [x] `presence_penalty`
- [x] `response_format`
- [x] `seed`
- [x] `stop`
- [x] `stream`
- [x] `temperature`
- [x] `top_p`
- [x] `max_tokens`
- [ ] `logit_bias`
- [ ] `tools`
- [ ] `tool_choice`
- [ ] `user`
- [ ] `n`

#### 注意事項

- `seed` を設定すると、常に `temperature` が `0` に設定されます。
- `finish_reason` は常に `stop` になります。
- プロンプト評価がキャッシュされている補完では、`usage.prompt_tokens` は `0` になります。

## モデル

モデルを使用する前に、ローカルにプルしてください `ollama pull`：

```shell
ollama pull llama2
```

### デフォルトのモデル名

`gpt-3.5-turbo` などのデフォルトの OpenAI モデル名を使用するツールについては、既存のモデル名を一時的な名前にコピーするには `ollama cp` を使用してください：

```
ollama cp llama2 gpt-3.5-turbo
```

その後、この新しいモデル名を `model` フィールドで指定できます：

```shell
curl http://localhost:11434/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "gpt-3.5-turbo",
        "messages": [
            {
                "role": "user",
                "content": "こんにちは！"
            }
        ]
    }'
```