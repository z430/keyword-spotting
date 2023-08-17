from setuptools import setup, find_packages

setup(
    name="keyword-spotting",
    packages=find_packages(exclude=["examples"]),
    version="0.0.1",
    license="MIT",
    description="Keyword Spotting Implementation",
    author="Phil Wang",
    author_email="maystya@gmail.com",
    url="https://github.com/z430/keyword-spotting",
    long_description_content_type="text/markdown",
    keywords=["artificial intelligence", "speech recognition", "audio"],
    install_requires=["torch>=2.0.1"],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
    ],
)
