"""
setup.py
========
PlantSwarm package setup.
"""

from setuptools import find_packages, setup

setup(
    name="plantswarm",
    version="0.3.0",
    description=(
        "PlantSwarm: Qwen-swarm regional delta extraction for PathomeDB"
    ),
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "anthropic>=0.40.0",
        "httpx>=0.27.0",
        "openpyxl>=3.1.0",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0",
        "requests>=2.31.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "dev": ["pytest", "black", "flake8"],
    },
)
