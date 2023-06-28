import os
import requests
from urllib.parse import urlsplit, urlunsplit
from tqdm import tqdm


def models(models_home=".", *args, **kwargs):
    for root, _, files in os.walk(models_home):
        for file in files:
            base, ext = os.path.splitext(file)
            if ext == ".bin":
                yield base, os.path.join(root, file)


def pull(remote, models_home=".", *args, **kwargs):
    if not (remote.startswith("http://") or remote.startswith("https://")):
        remote = f"https://{remote}"

    parts = urlsplit(remote)
    path_parts = parts.path.split("/tree/")

    if len(path_parts) == 1:
        model = path_parts[0]
        branch = "main"
    else:
        model, branch = path_parts

    model = model.strip("/")

    # Reconstruct the URL
    new_url = urlunsplit(
        (
            "https",
            parts.netloc,
            f"/api/models/{model}/tree/{branch}",
            parts.query,
            parts.fragment,
        )
    )

    print(f"Pulling {parts.netloc}/{model}...")

    response = requests.get(new_url)
    response.raise_for_status()  # Raises stored HTTPError, if one occurred

    json_response = response.json()

    # get the last bin file we find, this is probably the most up to date
    download_url = None
    for file_info in json_response:
        if file_info.get("type") == "file" and file_info.get("path").endswith(".bin"):
            f_path = file_info.get("path")
            download_url = f"https://huggingface.co/{model}/resolve/{branch}/{f_path}"

    if download_url is None:
        raise Exception("No model found")

    local_filename = os.path.join(models_home, os.path.basename(model)) + ".bin"

    # Check if file already exists
    if os.path.exists(local_filename):
        # TODO: check if the file is the same SHA
        first_byte = os.path.getsize(local_filename)
    else:
        first_byte = 0

    # If file size is non-zero, resume download
    if first_byte != 0:
        header = {"Range": f"bytes={first_byte}-"}
    else:
        header = {}

    response = requests.get(download_url, headers=header, stream=True)
    response.raise_for_status()  # Raises stored HTTPError, if one occurred

    total_size = int(response.headers.get("content-length", 0))

    with open(local_filename, "ab" if first_byte else "wb") as file, tqdm(
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
        initial=first_byte,
        ascii=" ==",
        bar_format="Downloading [{bar}] {percentage:3.2f}% {rate_fmt}{postfix}",
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)
