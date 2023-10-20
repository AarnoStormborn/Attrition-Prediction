from setuptools import find_packages, setup
from typing import List

def get_requirements(filepath:str)->List[str]:
    with open(filepath) as f:
        requirements = [req.strip() for req in f.readlines()]

        if '-e .' in requirements:
            requirements.remove('-e .')

    return requirements

setup(
    name='attrition-prediction',
    version='0.0',
    author='Harsh',
    author_email='harsh220902@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)