import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="3D-VQ-VAE-2",
    version="1.0.0",
    author="Robert Jan Schlimbach",
    description="3D VQ-VAE-2 for high-resolution CT scan synthesis",
    long_description=long_description,
    url="https://github.com/sara-nl/3D-VQ-VAE-2/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
