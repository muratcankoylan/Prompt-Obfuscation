import torch
import ollama
import numpy as np
from transformers import AutoTokenizer

def test_environment():
    # Test MPS availability
    print(f"MPS available: {torch.backends.mps.is_available()}")
    
    # Test Ollama connection
    try:
        models = ollama.list()
        print(f"Available Ollama models: {models}")
    except Exception as e:
        print(f"Ollama connection error: {e}")
    
    # Test basic model response
    try:
        response = ollama.generate(model='llama3.1:8b', 
                                 prompt='Say "Hello, World!"')
        print(f"Model response: {response}")
    except Exception as e:
        print(f"Model generation error: {e}")

if __name__ == "__main__":
    test_environment() 