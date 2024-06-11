from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="onnxruntime_transformers",
    version="0.1.1",
    author="chuboy",
    author_email="billju666@gmail.com",
    description="transformers for production runtime, 3x faster on cpu, no pytorch nor tensorflow included",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/billju/onnxruntime_transformers",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=[
        "tokenizers",
        "onnxruntime",
    ],
)
# pip install setuptools wheel twine
# python setup.py sdist bdist_wheel
# twine upload dist/*