import requests
import validators
from pathlib import Path
from os import path, walk
from urllib.parse import urlsplit, urlunsplit
from tqdm import tqdm


models_endpoint_url = 'https://ollama.ai/api/models'
models_home = path.join(Path.home(), '.ollama', 'models')


def models(*args, **kwargs):
    for _, _, files in walk(models_home):
        for file in files:
            base, ext = path.splitext(file)
            if ext == '.bin':
                yield base


# get the url of the model from our curated directory
def get_url_from_directory(model):
    response = requests.get(models_endpoint_url)
    response.raise_for_status()
    directory = response.json()
    for model_info in directory:
        if model_info.get('name') == model:
            return model_info.get('url')
    return model


def download_from_repo(url, file_name):
    parts = urlsplit(url)
    path_parts = parts.path.split('/tree/')

    if len(path_parts) == 1:
        location = path_parts[0]
        branch = 'main'
    else:
        location, branch = path_parts

    location = location.strip('/')
    if file_name == '':
        file_name = path.basename(location).lower()

    download_url = urlunsplit(
        (
            'https',
            parts.netloc,
            f'/api/models/{location}/tree/{branch}',
            parts.query,
            parts.fragment,
        )
    )
    response = requests.get(download_url)
    response.raise_for_status()
    json_response = response.json()

    download_url, file_size = find_bin_file(json_response, location, branch)
    return download_file(download_url, file_name, file_size)


def find_bin_file(json_response, location, branch):
    download_url = None
    file_size = 0
    for file_info in json_response:
        if file_info.get('type') == 'file' and file_info.get('path').endswith('.bin'):
            f_path = file_info.get('path')
            download_url = (
                f'https://huggingface.co/{location}/resolve/{branch}/{f_path}'
            )
            file_size = file_info.get('size')

    if download_url is None:
        raise Exception('No model found')

    return download_url, file_size


def download_file(download_url, file_name, file_size):
    local_filename = path.join(models_home, file_name) + '.bin'

    first_byte = path.getsize(local_filename) if path.exists(local_filename) else 0

    if first_byte >= file_size:
        return local_filename

    print(f'Pulling {file_name}...')

    header = {'Range': f'bytes={first_byte}-'} if first_byte != 0 else {}

    response = requests.get(download_url, headers=header, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0)) + first_byte

    with open(local_filename, 'ab' if first_byte else 'wb') as file, tqdm(
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
        initial=first_byte,
        ascii=' ==',
        bar_format='Downloading [{bar}] {percentage:3.2f}% {rate_fmt}{postfix}',
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

    return local_filename


def pull(model, *args, **kwargs):
    if path.exists(model):
        # a file on the filesystem is being specified
        return model
    # check the remote model location and see if it needs to be downloaded
    url = model
    file_name = ""
    if not validators.url(url) and not url.startswith('huggingface.co'):
        url = get_url_from_directory(model)
        file_name = model

    if not (url.startswith('http://') or url.startswith('https://')):
        url = f'https://{url}'

    if not validators.url(url):
        if model in models(models_home):
            # the model is already downloaded, and specified by name
            return model
        raise Exception(f'Unknown model {model}')

    local_filename = download_from_repo(url, file_name)

    return local_filename
