# Hands-on LLM Notebook

## Chapter 1: Intro to Language Models

- The first step is to load our model onto a GPU for faster inference
- Load the tokenizer and model separately
  - That is not always the case
- After loading the model and the tokenizer, we can use them directly, but it is easier to wrap them in a pipeline object
- Then we can prompt as a user

## Chapter 2: Tokens and Embeddings

- First we load the model into the GPU for faster inference
- We load them separately to explore them separately
- Pretrained model:
  ```python
  model = AutoModelForCausalLM.from_pretrained(
      "microsoft/Phi-3-mini-4k-instruct",
      device_map="cuda",
      torch_dtype="auto",
      trust_remote_code=False,
  )
  ```
- Tokenizer:
  - Turns words (or parts of words) into representations called tokens
  - `tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")`
  - Output tokens can include roles (i.e., assistant) as well as the output
  - There are many kinds of tokenizers
  - Can use token embeddings to recommend songs

## Chapter 3: Looking Inside Transformer LLM

- You can speed up generation by caching keys and values

## Chapter 4: Text Classification

- We can use many different models for sentiment analysis
  - Assuming we have a training and testing set, we can try to predict what words are positive and what words are negative using the training set and then compare what we found to the correct answers in the testing set
  - We can leverage embeddings to facilitate which words are positive/negative

## Chapter 5: Text Clustering and Topic Modeling

### A Step-by-Step Guide to Text Clustering

1. Embed the document
2. Reduce the dimensionality of the embedding
3. Cluster the reduced embedding
4. Fit the model to embeddings and extract the clusters
5. Plot the cluster for inspection

### Topic Modeling

- Using c-TF-IDF to extract representative keywords per cluster
- Providing multiple representation models to refine topic labels:
  - **KeyBERTInspired**: Selects keywords most similar to cluster centroid
  - **MaximalMarginalRelevance**: Balances relevance with diversity
  - **Text Generation (Flan-T5/GPT)**: Generates human-readable topic descriptions

## Chapter 6: Prompt Engineering

### Creating Templates

You can create templates for how models are supposed to respond:

```python
messages = [
    {"role": "user", "content": "Create a funny joke about chickens."}
]
prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False)
# Output:
# <s><|user|>
# Create a funny joke about chickens.<|end|>
# <|assistant|>
```

### Complex Prompts

- You can create complex prompts by creating a query that includes: persona, instructions, context, data_format, audience, tone, your personal data
- You can also provide examples in the prompt as a template
- You can also break up a problem using chain prompting
  - One prompt returns an output that ends up becoming context for the next prompt

### Chain of Thought

You can show the model how to think before answering:

```python
cot_prompt = [
    {"role": "user", "content": "Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?"},
    {"role": "assistant", "content": "Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11. The answer is 11."},
    {"role": "user", "content": "The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?"}
]
```

## Chapter 7: Advanced Text Generation Techniques and Tools

### Chains

Create a template:

```python
template = """<s><|user|>
Create a title for a story about {summary}. Only return the title.<|end|>
<|assistant|>"""
title_prompt = PromptTemplate(template=template, input_variables=["summary"])
title = LLMChain(llm=llm, prompt=title_prompt, output_key="title")
```

Invoke template:

```python
title.invoke({"summary": "a girl that lost her mother"})
```

### ConversationBuffer

Create a prompt that includes chat_history:

```python
template = """<s><|user|>Current conversation:{chat_history}

{input_prompt}<|end|>

<|assistant|>"""

prompt = PromptTemplate(
    template=template,
    input_variables=["input_prompt", "chat_history"]
)
```

This code allows your model to remember the conversation:

```python
from langchain.memory import ConversationBufferMemory

# Define the type of Memory we will use
memory = ConversationBufferMemory(memory_key="chat_history")

# Chain the LLM, Prompt, and Memory together
llm_chain = LLMChain(
    prompt=prompt,
    llm=llm,
    memory=memory
)
```

We can force the model to only remember the last two conversations:

```python
memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history")
```

### Agents

- Tell your models to use tools

## Chapter 8: Semantic Search and Retrieval Augmented Generation

### Dense Retrieval Example

1. Split the text into sentences
2. Remove whitespace
3. Embed the text chunks
4. Build the search index
5. Search the index

### RAG (Retrieval Augmented Generation)

1. Chunk the data
2. Embed the data
3. Send to vector database
4. Retrieve relevant data for query

## Chapter 9: Multimodal Large Language Models

### Text to Image

1. Load a tokenizer to preprocess the text
2. Load a processor to preprocess the images
3. Load CLIP model for generating text and image embeddings
4. Tokenize input
5. Convert input back to tokens
6. Create a text embedding
7. Preprocess image
8. Prepare image for visualization
9. Visualize preprocessed image
10. Create the image embedding
11. Normalize the embeddings
12. Calculate their similarity

## Chapter 10: Creating Text Embedding Models

### Supervised Methods

#### SoftmaxLoss (Classification-based)

- Trains on labeled data (e.g., MNLI: entailment/neutral/contradiction)
- Treats embedding as classification task
- Performance: ~0.45 Spearman correlation on STSB

#### CosineSimilarityLoss (Regression-based)

- Directly optimizes for semantic similarity
- Uses sentence pairs with similarity scores
- Performance: ~0.73 Spearman correlation (60% improvement)

#### MultipleNegativesRankingLoss (Contrastive)

- Most effective supervised method
- Uses in-batch negatives for efficient contrastive learning
- Creates positive pairs from entailment, treats others as negatives
- Performance: ~0.87 Spearman correlation (best supervised result)

### Advanced Methods

#### Augmented SBERT (Knowledge Distillation)

1. Train cross-encoder on labeled data (slower but accurate)
2. Use cross-encoder to label unlabeled data
3. Train bi-encoder on augmented dataset
4. Leverages unlabeled data to improve performance

### Unsupervised Learning

#### TSDAE (Transformer-based Denoising AutoEncoder)

- No labeled data required
- Corrupts sentences by deleting words (default 60% deletion)
- Trains model to reconstruct original from corrupted version
- Learns semantic representations through reconstruction task

### Evaluation Framework

#### STSB (Semantic Textual Similarity Benchmark)

- Primary evaluation metric during training
- Measures correlation between predicted and human similarity scores

#### MTEB (Massive Text Embedding Benchmark)

- Comprehensive benchmark across multiple tasks:
  - Classification (e.g., Banking77)
  - Clustering
  - Retrieval
  - Semantic similarity
- Industry-standard for comparing embedding models

## Chapter 11: Fine-tuning Representation Models for Classification

### Supervised Classification

1. Load model and tokenizer
2. Pad to the longest sequence in the batch
3. Tokenize train/test data
4. Create training arguments for parameter tuning:

```python
training_args = TrainingArguments(
    "model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
    save_strategy="epoch",
    report_to="none"
)
```

5. Create trainer which executes the training process:

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
```

6. Freeze blocks that are not the trainable classification head

### Few-Shot Learning with SetFit

- Uses only 16 examples per class (32 total)
- Leverages pre-trained sentence embeddings (all-mpnet-base-v2)
  1. **Contrastive learning**: Fine-tune embeddings on generated sentence pairs
  2. **Classification head**: Train logistic regression on embeddings
- Result: Competitive performance with minimal data
- Advantages:
  - Extremely data-efficient
  - Fast training (3 epochs)
  - No need for large labeled datasets

### Masked Language Modeling (MLM)

#### Domain Adaptation

- Continue pre-training BERT on domain-specific text
- Randomly mask 15% of tokens
- Model learns to predict masked tokens
- Adapts vocabulary and language patterns to new domain

**Use Case**: Before fine-tuning on classification, adapt BERT to domain-specific language (e.g., medical, legal, movie reviews)

**Example**:
- Before MLM: "What a horrible [MASK]!" → "day", "thing"
- After MLM on movie reviews: "What a horrible [MASK]!" → "movie", "film"

### Named Entity Recognition (NER)

#### Token Classification Task

- Predict labels for each token (not entire sequence)
- Dataset: CoNLL-2003 (PER, ORG, LOC, MISC entities)
- Challenge: Handle sub-word tokenization

#### Label Alignment Strategy

- BERT splits words into sub-tokens (e.g., "Maarten" → ["Ma", "##arten"])
- First sub-token gets original label (B-PER)
- Subsequent sub-tokens get continuation label (I-PER)
- Special tokens ([CLS], [SEP]) get -100 (ignored in loss)

#### Architecture

- BERT encoder + token classification head
- Outputs label per token (9 classes: O, B-PER, I-PER, etc.)
- Evaluation: Sequence-level F1 score
- Early layers learn general features; late layers learn task-specific patterns

## Chapter 12: Fine-Tuning Generation Models

### Two-Stage Training

#### Stage 1: Supervised Fine-tuning (SFT)

- Transforming a base language model into an instruction-following assistant that responds to user prompts
- May not align with human preferences (e.g., verbosity, helpfulness, safety)

#### Stage 2: Preference Tuning (DPO)

- Align the SFT model with human preferences by learning from chosen vs. rejected response pairs
- **Direct Preference Optimization (DPO)**:
  - No separate reward model needed
  - More stable training
  - Simpler implementation
  - Direct optimization of policy

**Training Process**:

1. Load SFT model (from Stage 1)
2. Apply new LoRA adapters on top of SFT model
3. Train with DPO loss:
   - Increases probability of chosen responses
   - Decreases probability of rejected responses
   - Beta parameter (0.1) controls strength of preference

**Merging Adapters** (Two-step merge):

1. Merge SFT LoRA → Base model
2. Merge DPO LoRA → SFT model
3. Result: Single model with both instruction-following and preference alignment

### Practical Applications

- Fine-tune open-source models for specific domains
- Align models with company/user preferences
- Create custom assistants with limited compute
- Iterate quickly with efficient training

