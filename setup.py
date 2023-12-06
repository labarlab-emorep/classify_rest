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
        "pandas>=2.1.3",
        "setuptools>=65.5.1"
    ],
)
