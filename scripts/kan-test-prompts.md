# KAN Attention Test Prompts

## How to run

```bash
# Terminal 1: Start server with KAN enabled
OLLAMA_KAN_ATTENTION=1 OLLAMA_FLASH_ATTENTION=false OLLAMA_DEBUG=1 ollama serve

# Terminal 2: Chat with a tiny model
ollama run tinyllama
```

Watch the server logs in Terminal 1 for KAN convergence events.
Run each prompt below and save the responses. Then restart WITHOUT
KAN (`ollama serve`) and run them again. Compare side by side.

---

## Warm-up prompts (these train the KAN)

Just chat normally. Each token generated is one KAN training step.
After ~100-200 tokens, you should see "KAN attention converged" in the logs.

```
Tell me a story about a dragon who is afraid of fire.
```

```
Explain how a computer works to a 5 year old. Be very detailed.
```

```
Write a recipe for chocolate chip cookies. Include exact measurements.
```

```
What are the differences between Python, JavaScript, and Rust?
Compare them in terms of speed, safety, and ease of use.
```

```
Describe what happens inside a star from birth to death.
Include every stage in detail.
```

By now the KAN should be converging. Check the logs for:
- "KAN attention converged layer=layer_N"
- "KAN Phase 2 activated"

---

## The screaming tests

These prompts are specifically designed to expose attention quality.
Tiny models with soft/uniform attention struggle with ALL of these.
Sharper attention should produce noticeably better results.

### Test 1: Needle in a haystack
(Tests whether the model can attend to specific details in long context)

```
Here is a list of facts:
- The sky is blue
- Water boils at 100 degrees Celsius
- The secret code is PURPLE-ELEPHANT-42
- Grass is green
- The speed of light is 299,792,458 m/s
- Dogs are mammals
- The Earth orbits the Sun
- Honey never spoils

What is the secret code?
```

### Test 2: Instruction following precision
(Tests whether the model attends to the constraint)

```
Name exactly 3 countries that start with the letter J. Do not name more
than 3. Do not explain anything. Just list them.
```

### Test 3: Long-range dependency
(Tests whether attention can connect distant context)

```
Alice gave Bob a red ball. Bob gave Carol a blue ball. Carol gave Dave
a green ball. Dave gave Eve the ball he received from Carol.

What color ball does Eve have?
```

### Test 4: Negation tracking
(Soft attention often misses negations)

```
Which of these statements is FALSE?
A) The sun rises in the east
B) Water freezes at 0 degrees Celsius  
C) The moon is larger than the Earth
D) Humans need oxygen to breathe

Just give the letter.
```

### Test 5: Format compliance
(Sharper attention = better format adherence)

```
Output ONLY a valid JSON object with these fields: name, age, city.
Use the values: John, 30, London. No explanation, no markdown, just JSON.
```

### Test 6: Poetry with structure
(Tests attention to rhyme scheme and meter constraints)

```
Write a limerick about a cat. It must follow AABBA rhyme scheme exactly.
```

### Test 7: Mathematical reasoning chain
(Tests whether attention tracks intermediate values)

```
If x = 3 and y = x + 2 and z = y * x, what is z?
Show your work step by step.
```

### Test 8: Selective repetition
(Soft attention causes repetition; sharp attention shouldn't)

```
Tell me about the solar system. Mention each planet exactly once.
Do not repeat any planet name.
```

---

## What to look for

### Signs the KAN is working (sharper attention):
- More precise answers (needle-in-haystack found correctly)
- Better instruction following (exactly 3, not 4 or 5)
- Less repetition and rambling
- Better format compliance (valid JSON, correct rhyme scheme)
- Correct logical reasoning chains
- Correct handling of negation

### Signs of Phase 2 self-evolution:
- Responses get progressively crisper over time
- The model becomes more "decisive" — less hedging
- Mathematical reasoning chains become more accurate
- Complex multi-step instructions are followed more reliably

### Quantitative comparison:
For each test, score 0/1 on whether the model:
1. Got the right answer
2. Followed the format constraint
3. Avoided unnecessary rambling
4. Showed no repetition

Compare total scores: baseline softmax vs KAN attention.
A score difference of 3+ out of 8 tests is significant for a tiny model.
