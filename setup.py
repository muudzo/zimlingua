from setuptools import setup, find_packages

setup(
    name="zimlingua",
    version="0.1.0",
    description="Offline NMT for Shona/Ndebele/English using NLLB-200",
    author="ZimLingua Team",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "ctranslate2>=3.0.0",
        "sentencepiece>=0.1.99",
        "transformers>=4.30.0",
        "datasets>=2.14.0",
        "pyyaml",
        "pandas",
        "peft",
        # "streamlit" # Optional
    ],
    entry_points={
        "console_scripts": [
            "zimlingua=src.cli:main",
        ],
    },
)
