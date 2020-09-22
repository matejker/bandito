from setuptools import find_packages, setup

with open("requirements/base.txt") as f:
    required = f.read().splitlines()

setup(
    name="bandito",
    version="0.1.0",
    description="A fairly simple package playing with Stochastic Multi-Armed Bandits 🎰",
    author="Matej Kerekrety",
    author_email="matej.kerekrety@gmail.com",
    packages=find_packages(exclude=("tests", "notebooks",)),
    include_package_data=True,
    zip_safe=True,
    install_requires=required,
)
