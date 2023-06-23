import site
import os
from PyInstaller.__main__ import run as pyi_run

# Get the directory of site-packages and llama_cpp
site_packages_dir = site.getsitepackages()[0]
llama_cpp_dir = os.path.join(site_packages_dir, "llama_cpp")

# Prepare the arguments for PyInstaller
args = [
    "server.py",
    "--paths",
    site_packages_dir,
    "--add-data",
    f"{llama_cpp_dir}{os.pathsep}llama_cpp",
    "--onefile",
]

# Generate the .spec file and run PyInstaller
pyi_run(args)
