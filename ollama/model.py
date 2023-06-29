import os
import requests
import validators
from urllib.parse import urlsplit, urlunsplit
from tqdm import tqdm


models_endpoint_url = 'https://ollama.ai/api/models'


def models(models_home='.', *args, **kwargs):
    for _, _, files in os.walk(models_home):
        for file in files:
            base, ext = os.path.splitext(file)
            if ext == '.bin':
                yield base


def pull(model, models_home='.', *args, **kwargs):
    url = model
    if not validators.url(url) and not url.startswith('huggingface.co'):
        # this may just be a local model location
        if model in models(models_home):
            return model
        # see if we have this model in our directory
        response = requests.get(models_endpoint_url)
        response.raise_for_status()
        directory = response.json()
        for model_info in directory:
            if model_info.get('name') == model:
                url = f"https://{model_info.get('url')}"
                break
        if not validators.url(url):
            raise Exception(f'Unknown model {model}')

    if not (url.startswith('http://') or url.startswith('https://')):
        url = f'https://{url}'

    parts = urlsplit(url)
    path_parts = parts.path.split('/tree/')

    if len(path_parts) == 1:
        location = path_parts[0]
        branch = 'main'
    else:
        location, branch = path_parts

    location = location.strip('/')

    # Reconstruct the URL
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
    response.raise_for_status()  # Raises stored HTTPError, if one occurred

    json_response = response.json()

    # get the last bin file we find, this is probably the most up to date
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

    local_filename = os.path.join(models_home, os.path.basename(url)) + '.bin'

    # Check if file already exists
    first_byte = 0
    if os.path.exists(local_filename):
        # TODO: check if the file is the same SHA
        first_byte = os.path.getsize(local_filename)

    if first_byte >= file_size:
        return local_filename

    print(f'Pulling {model}...')

    # If file size is non-zero, resume download
    if first_byte != 0:
        header = {'Range': f'bytes={first_byte}-'}
    else:
        header = {}

    response = requests.get(download_url, headers=header, stream=True)
    response.raise_for_status()  # Raises stored HTTPError, if one occurred

    total_size = int(response.headers.get('content-length', 0))

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
