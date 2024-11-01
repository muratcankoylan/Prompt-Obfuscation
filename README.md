# Prompt Obfuscation Implementation Attempt

An attempt to reproduce the paper "Prompt Obfuscation for Large Language Models" by David Pape, Thorsten Eisenhofer, and Lea Schönherr.

## Overview

This repository contains an implementation attempt of the prompt obfuscation methodology described in the paper. The core idea is to find a representation of the original system prompt that leads to the same functionality while being obfuscated.

### Key Features
- BERT embedding space manipulation
- Functional collision finding
- Task semantics preservation
- LLaMA 3.1 integration

### Current Status
This is an **unsuccessful attempt** at reproducing the paper's results. Key issues:
1. Embedding space manipulation not achieving desired obfuscation
2. Task semantics not fully preserved
3. Generated prompts often nonsensical

## Installation

```bash
# Clone repository
git clone https://github.com/muratcankoylan/prompt-obfuscation-attempt.git
cd prompt-obfuscation-attempt

# Create virtual environment
python -m venv prompt_obf_env
source prompt_obf_env/bin/activate  # Linux/Mac
# or
prompt_obf_env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

```python
from obfuscation import PromptObfuscator

# Initialize obfuscator
obfuscator = PromptObfuscator()

# Obfuscate a prompt
original_prompt = "Write a hello world program"
obfuscated_prompt = obfuscator.obfuscate_prompt(original_prompt)
```

## Project Structure
```
prompt-obfuscation/
├── src/
│   ├── obfuscation/
│   │   ├── __init__.py
│   │   └── prompt_handler.py
│   ├── monitoring/
│   │   └── performance_tracker.py
│   ├── tests/
│   │   └── test_obfuscation.py
│   └── main.py
├── setup.py
└── README.md
```

## Paper Reference
Pape, D., Eisenhofer, T., & Schönherr, L. (2024). Prompt Obfuscation for Large Language Models. arXiv preprint arXiv:2409.11026.
https://arxiv.org/abs/2409.11026 

## License
MIT License

## Disclaimer
This is an unsuccessful attempt at reproducing the paper's results. The implementation falls short of achieving the paper's reported performance and should not be considered a faithful reproduction. 
