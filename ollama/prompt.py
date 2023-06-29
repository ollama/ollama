import os
from difflib import SequenceMatcher
from jinja2 import Environment, PackageLoader


def template(model, prompt):
    best_ratio = 0
    best_template = ''

    environment = Environment(loader=PackageLoader(__name__, 'templates'))
    for template in environment.list_templates():
        base, _ = os.path.splitext(template)
        ratio = SequenceMatcher(None, os.path.basename(model.lower()), base).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_template = template

    template = environment.get_template(best_template)
    return template.render(prompt=prompt)
