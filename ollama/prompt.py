from os import path
from difflib import get_close_matches
from jinja2 import Environment, PackageLoader


def template(name, prompt):
    environment = Environment(loader=PackageLoader(__name__, 'templates'))
    best_templates = get_close_matches(
        path.basename(name), environment.list_templates(), n=1, cutoff=0
    )
    template = environment.get_template(best_templates.pop())
    return template.render(prompt=prompt)
