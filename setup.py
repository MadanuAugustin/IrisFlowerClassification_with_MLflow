
from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    description = f.read()



setup(
    name='IrisFlowerClassification_with_MLflow',
    version='0.0.0.0',
    author='augustin',
    author_email='augustin7766@gmail.com',
    description='End-End_MLflow-project',
    long_description=description,
    url = 'https://github.com/MadanuAugustin/IrisFlowerClassification_with_MLflow.git',
    packages = find_packages()
)