from setuptools import setup, find_packages


setup (

    packages=find_packages(),
    install_requires = open("requirements.txt").read().splitlines(),


)