from os import walk, path


def models(models_home='.', *args, **kwargs):
    for root, _, files in walk(models_home):
        for file in files:
            base, ext = path.splitext(file)
            if ext == '.bin':
                yield base, path.join(root, file)
