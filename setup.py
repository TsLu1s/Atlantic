from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    #long_description = f.read()

setup(
    name="atlantic",
    version="0.0.1",
    description="Atlantic if automated preprocessing framework for supervised machine learning",
    long_description= file: README.md #### 
    long_description_content_type="text/markdown",
    url="https://github.com/TsLu1s/Atlantic",
    author="Luís Santos",
    author_email="luisfssantos98@hotmail.com",
    license="MIT",
    #packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    python_requires='>=3.7.1',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "atlantic"},  
    keywords=[
        "data science",
        "machine learning",
        "data processing",
        "data preprocessing",
        "automated machine learning",
        "automl",
    ],
  install_requires=open("requirements.txt").readlines(),
)
 
