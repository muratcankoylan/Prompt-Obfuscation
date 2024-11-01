import unittest
import torch
import ollama
from obfuscation import PromptObfuscator, DEFAULT_CONFIG
from main import LlamaObfuscationSystem

class TestObfuscationSystem(unittest.TestCase):
    def setUp(self):
        """Initialize test environment"""
        self.obfuscator = PromptObfuscator()
        self.llama_system = LlamaObfuscationSystem()
        self.test_prompt = "Write a hello world program"
        
    def test_obfuscator_initialization(self):
        """Test if obfuscator initializes correctly"""
        self.assertIsNotNone(self.obfuscator)
        self.assertIsNotNone(self.obfuscator.config)
        self.assertEqual(self.obfuscator.config['noise_factor'], DEFAULT_CONFIG['noise_factor'])
        self.assertIn('device', self.obfuscator.config)
        self.assertIsInstance(self.obfuscator.device, torch.device)
        
    def test_prompt_obfuscation(self):
        """Test if prompt obfuscation works"""
        obfuscated = self.obfuscator.obfuscate_prompt(self.test_prompt)
        self.assertIsNotNone(obfuscated)
        self.assertNotEqual(obfuscated, self.test_prompt)
        self.assertIn("Please answer this question:", obfuscated)
        
    def test_llama_system(self):
        """Test if LlamaObfuscationSystem works end-to-end"""
        result = self.llama_system.process_prompt(self.test_prompt)
        self.assertEqual(result['status'], 'success')
        self.assertIsNotNone(result['obfuscated_prompt'])
        self.assertIsNotNone(result['response'])
        
    def test_error_handling(self):
        """Test error handling with invalid inputs"""
        result = self.llama_system.process_prompt("")
        self.assertIn('status', result)
        
    def test_mps_availability(self):
        """Test MPS (Metal Performance Shaders) availability"""
        self.assertTrue(hasattr(torch.backends, 'mps'))
        
    def test_model_connection(self):
        """Test Ollama model connection"""
        try:
            models = ollama.list()
            self.assertIn('llama3.1:8b', str(models))
        except Exception as e:
            self.fail(f"Model connection failed: {e}")
        
    def test_embedding_dimensions(self):
        """Test if embeddings maintain correct dimensions"""
        test_prompt = "Write a hello world program"
        original_embeddings = self.obfuscator._text_to_embeddings(test_prompt)
        
        # Check dimensions
        self.assertEqual(len(original_embeddings), len(test_prompt.split()))
        for emb_type, emb in original_embeddings:
            if emb_type == "EMB":
                self.assertEqual(emb.shape[-1], 4096)  # LLaMA's embedding dimension

def run_tests():
    unittest.main()

if __name__ == '__main__':
    run_tests() 