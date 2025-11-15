"""Setup configuration for the RFP Agent package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="rfp-agent",
    version="0.1.0",
    author="RFP Agent Contributors",
    description="Semi-automated RFP answer engine using Gemini AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rdwornik/rfp-agent-cognitive-planning",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "google-generativeai>=0.3.0",
        "pandas>=2.0.0",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0",
    ],
    entry_points={
        "console_scripts": [
            "rfp-agent=rfp_agent.main:main",
        ],
    },
)
