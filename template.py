from difflib import SequenceMatcher
import json

with open("model_prompts.json", "r") as f:
    model_prompts = json.load(f)


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
