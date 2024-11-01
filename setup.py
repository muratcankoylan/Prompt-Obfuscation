from setuptools import setup, find_packages

setup(
    name="prompt-obfuscation",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'torch',
        'ollama',
        'numpy',
        'transformers',
        'psutil',
        'nltk',
        'sentencepiece',
        'cryptography',
        'fasttext-wheel',
        'fasttext'
    ],
    python_requires='>=3.8',
    author="Prompt Obfuscation Team",
    description="A secure prompt obfuscation system for LLaMA 3.1 8B",
    license="MIT"
) 