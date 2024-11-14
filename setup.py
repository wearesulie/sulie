from setuptools import setup, find_packages


setup(
    name="sulie",
    version="1.0.4",
    packages=find_packages(),
    install_requires=[
        "requests",
        "pandas",
        "pyarrow",
        "tqdm"
    ],
    extras_require={},
    author="Dominik Safaric",
    author_email="hello@sulie.co",
    description="A Python package for interacting with Sulie API",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/wearesulie/sulie",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    project_urls={
        "Documentation": "https://docs.sulie.co",
        "Source": "https://github.com/wearesulie/sulie",
        "Tracker": "https://github.com/wearesulie/sulie/issues"
    },
    python_requires='>=3.8',
)