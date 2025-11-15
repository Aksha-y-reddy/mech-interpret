from setuptools import setup, find_packages

setup(
    name="mech-interpret-bias-detection",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Detecting Semantic Data Poisoning with Mechanistic Interpretability",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mech-interpret",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
        line.strip()
        for line in open("requirements.txt").readlines()
        if line.strip() and not line.startswith("#")
    ],
)

