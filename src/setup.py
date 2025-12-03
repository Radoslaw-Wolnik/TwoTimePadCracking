from setuptools import setup, find_packages

setup(
    name="two-time-pad-decrypt",
    version="2.0.0",
    description="Two-Time Pad Decryption System using Language Models",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "requests>=2.25.1",
        "beautifulsoup4>=4.9.3",
        "tqdm>=4.60.0",
        "numpy>=1.20.0",
        "matplotlib>=3.7",
        "pytest>=7.2",
    ],
    python_requires=">=3.8",
    author="Radoslaw Wolnik",
    author_email="radoslaw.m.wolnik@gmail.com",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Security :: Cryptography",
    ],
)