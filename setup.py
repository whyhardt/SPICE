from setuptools import setup
import os

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Read dev requirements if file exists
dev_requirements = []
if os.path.exists("requirements-dev.txt"):
    with open("requirements-dev.txt", "r", encoding="utf-8") as fh:
        dev_requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]
else:
    dev_requirements = [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "black>=23.0.0",
        "flake8>=6.0.0",
        "mypy>=1.0.0",
        "isort>=5.0.0",
    ]

setup(
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
    },
) 