from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

with open(".version") as f:
    version = f.read()

setup(
    name="torchplus",
    version=version,
    packages=find_packages(),
    description="Useful extras when working with pytorch",
    install_requires=requirements,
)
