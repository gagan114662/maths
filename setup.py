"""
Setup script for Enhanced Trading Strategy System.
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

with open("tests/requirements-test.txt", "r", encoding="utf-8") as f:
    test_requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="trading-system",
    version="1.0.0",
    author="Trading System Team",
    author_email="team@tradingsystem.com",
    description="Enhanced Trading Strategy System with LLM Integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/trading-system",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    tests_require=test_requirements,
    extras_require={
        "dev": [
            "black>=22.0.0",
            "isort>=5.10.0",
            "mypy>=0.910",
            "pylint>=2.12.0",
            "flake8>=4.0.0"
        ],
        "docs": [
            "sphinx>=4.4.0",
            "sphinx-rtd-theme>=1.0.0",
            "sphinx-autodoc-typehints>=1.12.0"
        ],
        "test": test_requirements
    },
    entry_points={
        "console_scripts": [
            "trading-system=src.web.app:main",
            "trading-monitor=src.monitoring.dashboard:main",
            "run-tests=run_tests:main"
        ]
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.json", "*.html", "*.css", "*.js"]
    },
    data_files=[
        ("config", ["config/example_config.yaml"]),
        ("static/styles", ["static/styles/main.css"]),
        ("templates", [
            "templates/base.html",
            "templates/index.html"
        ])
    ]
)