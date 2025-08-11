# setup.py

from setuptools import setup, find_packages

setup(
    name="vlm_inspector",
    version="0.1.0",
    author="",
    author_email="",
    description="A package for vehicle detection and fine-grained analysis using Qwen and Gemma VLMs.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/your_username/vlm_vehicle_inspector", 
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=open('requirements.txt').read().splitlines(),
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'vlm-inspect=vlm_inspector.cli:main',
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License", 
        "Operating System :: OS Independent",
    ],
)