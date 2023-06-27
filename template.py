from difflib import SequenceMatcher

model_prompts = {
    "alpaca": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{prompt}\n\n### Response:\n\n",
    "ggml": "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n### Human: Hello, Assistant.\n### Assistant: Hello. How may I help you today?\n### Human: ${prompt}",
    "gpt4": "### Instruction:\n{prompt}\n\n### Response:\n",
    "hermes": "### Instruction:\n{prompt}\n\n### Response:\n",
    "oasst": "{prompt}",
    "orca": "### System:\nYou are an AI assistant that follows instruction extremely well. Help as much as you can.\n\n### User:\n{prompt}\n\n### Response:",
    "qlora": "### Human: {prompt}\n### Assistant:",
    "tulu": "\n{prompt}\n\n(include newline)",
    "vicuna": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\nUSER: {prompt}\nASSISTANT:",
    "wizardlm": "{prompt}\n\n### Response:",
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
    p = model_prompts.get(closest_key)  # TODO: provide a better default template
    return p.format(prompt=prompt)
