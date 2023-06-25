import site
import os
from PyInstaller.__main__ import run as pyi_run

# the llama_cpp directory is not included if not explicitly added
site_packages_dir = site.getsitepackages()[0]
llama_cpp_dir = os.path.join(site_packages_dir, "llama_cpp")

args = [
    "proto.py",
    "--paths",
    site_packages_dir,
    "--add-data",
    f"{llama_cpp_dir}{os.pathsep}llama_cpp",
    "--onefile"
]

# generate the .spec file and run PyInstaller
pyi_run(args)
