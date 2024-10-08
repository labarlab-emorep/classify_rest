from setuptools import setup, find_packages

exec(open("classify_rest/_version.py").read())

setup(
    name="classify_rest",
    version=__version__,  # noqa: F821
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "classify_rest=classify_rest.cli:main",
        ]
    },
    install_requires=[
        "nibabel>=5.1.0",
        "pandas>=1.5.2",
        "paramiko>=3.3.1",
        "PyMySQL>=1.1.0",
        "setuptools>=65.5.1",
        "sshtunnel>=0.4.0",
    ],
)
