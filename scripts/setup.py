#!/usr/bin/env python
"""Setup configuration for Ollama package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements/core.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip()
        for line in fh
        if line.strip()
        and not line.startswith("#")
        and not line.startswith("--")
    ]

setup(
    name="ollama",
    version="1.0.0",
    author="kushin77",
    author_email="",
    description="Elite local AI development platform for LLM inference",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kushin77/ollama",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "dev": ["black", "isort", "mypy", "pytest", "pytest-cov"],
        "docs": ["sphinx", "sphinx-rtd-theme", "myst-parser"],
    },
    entry_points={
        "console_scripts": [
            "ollama=ollama.cli:main",
        ],
    },
)
