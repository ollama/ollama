from difflib import SequenceMatcher

model_prompts = {
    "alpaca": """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{prompt}

### Response:

""",
    "oasst": "<|prompter|>{prompt}<|endoftext|><|assistant|>",
    "vicuna": """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

USER: {prompt}
ASSISTANT:""",
    "hermes": """### Instruction:
{prompt}

### Response:
""",
    "gpt4": """### Instruction:
{prompt}

### Response:
""",
    "qlora": """### Human: {prompt}
### Assistant:""",
    "tulu": """<|user|>
{prompt}
<|assistant|>
(include newline)""",
    "wizardlm-7b": """{prompt}

### Response:""",
    "wizardlm-13b": """{prompt}

### Response:""",
    "wizardlm-30b": """{prompt}

### Response:""",
}


def template(model, prompt):
    max_ratio = 0
    closest_key = ""
    model_name = model.lower()
    # Find the specialized prompt with the closest name match
    for key in model_prompts.keys():
        ratio = SequenceMatcher(None, model_name, key).ratio()
        if ratio > max_ratio:
            max_ratio = ratio
            closest_key = key
    # Return the value of the closest match
    p = model_prompts.get(closest_key)  # .format(placeholder=prompt)
    return p.format(prompt=prompt)
