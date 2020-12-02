import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hpc-generative-models",
    version="0.0.1",
    author="Robert Jan Schlimbach",
    description="A collection of hpc generative models",
    long_description=long_description,
    url="https://github.com/sara-nl/vq-vae-2-pytorch/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
