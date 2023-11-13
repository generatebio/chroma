import codecs
import os
import time

from setuptools import find_packages, setup

with open("requirements.txt", "r") as req_file:
    requirements = [line.split("#")[0].strip() for line in req_file]
    requirements = [line for line in requirements if line]


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


version = get_version("chroma/__init__.py")

# During CICD, append "-dev" and unix timestamp to version
if os.environ.get("CI_COMMIT_BRANCH") == "develop":
    version += f".dev{int(time.time())}"

setup(
    name="generate-chroma",
    version=version,
    url="https://github.com/generatebio/chroma",
    packages=find_packages(),
    description="Chroma is a generative model for designing proteins programmatically",
    include_package_data=True,
    author="Generate Biomedicines",
    license="Apache 2.0",
    install_requires=requirements,
)
