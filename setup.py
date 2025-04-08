from setuptools import setup, find_packages
import os
import re

def read_requirements():
    with open("requirements.txt") as req:
        return req.read().splitlines()

def read_version():
    version_file = os.path.join("open_rag_eval", "_version.py")
    with open(version_file, "r") as vf:
        content = vf.read()
    return re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content).group(1)


setup(
    name="open-rag-eval",
    version=read_version(),
    author="Suleman Kazi",
    author_email="suleman@vectara.com",
    description="A Python package for RAG Evaluation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/vectara/open-rag-eval",
    packages=find_packages(),
    install_requires=read_requirements(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords=["RAG", "Evaluation", "RAG evaluation", "Vectara"],
    project_urls={
        "Documentation": "https://vectara.github.io/open-rag-eval/",
    },
    python_requires=">=3.9",
)
