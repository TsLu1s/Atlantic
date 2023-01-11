import setuptools
# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setuptools.setup(
    name="atlantic",
    version="1.0.11",
    description="Atlantic is an automated preprocessing framework for supervised machine learning",
    long_description=long_description,      
    long_description_content_type="text/markdown",
    url="https://github.com/TsLu1s/Atlantic",
    author="Luís Santos",
    author_email="luisf_ssantos@hotmail.com",
    license="MIT",
    packages=setuptools.find_packages(),
    python_requires='>=3.7.1',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    py_modules=["atlantic"],
    package_dir={"": "src/atlantic"},  
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


  
