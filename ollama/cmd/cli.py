import json
from pathlib import Path
from argparse import ArgumentParser

from ollama import model, engine
from ollama.cmd import server


def main():
    parser = ArgumentParser()
    parser.add_argument('--models-home', default=Path.home() / '.ollama' / 'models')

    subparsers = parser.add_subparsers()

    server.set_parser(subparsers.add_parser('serve'))

    list_parser = subparsers.add_parser('list')
    list_parser.set_defaults(fn=list)

    generate_parser = subparsers.add_parser('generate')
    generate_parser.add_argument('model')
    generate_parser.add_argument('prompt')
    generate_parser.set_defaults(fn=generate)

    args = parser.parse_args()
    args = vars(args)

    fn = args.pop('fn')
    fn(**args)


def list(*args, **kwargs):
    for m in model.models(*args, **kwargs):
        print(m)


def generate(*args, **kwargs):
    for output in engine.generate(*args, **kwargs):
        output = json.loads(output)

        choices = output.get('choices', [])
        if len(choices) > 0:
            print(choices[0].get('text', ''), end='')
