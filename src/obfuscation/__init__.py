from .prompt_handler import PromptObfuscator
import torch

__version__ = "0.1.0"
__author__ = "Prompt Obfuscation Team"
__description__ = "A secure prompt obfuscation system for LLaMA 3.1 8B"
__license__ = "MIT"

# Configuration defaults
DEFAULT_CONFIG = {
    # BERT Configuration
    'bert_model': 'bert-base-uncased',
    'bert_dim': 768,
    'top_k': 5,
    
    # General Configuration
    'device': 'mps' if torch.backends.mps.is_available() else 'cpu',
    'learning_rate': 0.01,
    'max_iterations': 5,
    'similarity_threshold': 0.85,
    
    # LLaMA Configuration
    'vocab_size': 128256,  # LLaMA 3.1 vocabulary size
    'embedding_dim': 4096,  # LLaMA 3.1 8B embedding dimension
    'hidden_dim': 14336,   # MLP intermediate dimension
    'num_attention_heads': 32,
    'head_dim': 128,      # 4096/32
    'rope_base': 500000,  # LLaMA 3.1 specific base value
    'max_position': 128000,
    
    # Security Settings
    'noise_factor': 0.05,
    'obfuscation_level': 'high',  # low, medium, high
    'techniques': ['semantic', 'syntactic', 'structural'],
    'security': {
        'min_transform_ratio': 0.7,
        'max_original_words': 0.3,
        'force_context_shift': True
    }
}

# Export the main class and configuration
__all__ = [
    "PromptObfuscator",
    "DEFAULT_CONFIG"
]
