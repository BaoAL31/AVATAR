from setuptools import setup, find_packages

setup(
    name="AVATAR",          # name of your project
    packages=find_packages(),  # automatically finds all folders with __init__.py
    extras_require={
        "eval": [
            "pyannote.metrics>=3.2",
            "pyannote.core>=4.5",
            "pyannote.database>=4.1",
            "PyYAML>=6.0",
        ],
    },
)
