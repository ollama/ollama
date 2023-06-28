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

    print(f"Fetching model from {new_url}")

    response = requests.get(new_url)
    response.raise_for_status()  # Raises stored HTTPError, if one occurred

    json_response = response.json()

    for file_info in json_response:
        if file_info.get("type") == "file" and file_info.get("path").endswith(".bin"):
            f_path = file_info.get("path")
            download_url = f"https://huggingface.co/{model}/resolve/{branch}/{f_path}"
            local_filename = os.path.join(models_home, os.path.basename(model)) + ".bin"

            if os.path.exists(local_filename):
                # TODO: check if the file is the same SHA
                break

            response = requests.get(download_url, stream=True)
            response.raise_for_status()  # Raises stored HTTPError, if one occurred

            total_size = int(response.headers.get("content-length", 0))

            with open(local_filename, "wb") as file, tqdm(
                desc=local_filename,
                total=total_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(chunk_size=1024):
                    size = file.write(data)
                    bar.update(size)

            break  # Stop after downloading the first .bin file
