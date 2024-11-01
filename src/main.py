from typing import Dict, Any
import ollama
from obfuscation import PromptObfuscator, DEFAULT_CONFIG
from monitoring.performance_tracker import PerformanceTracker

class LlamaObfuscationSystem:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {
            'model': 'llama3.1:8b',
            'options': {
                'temperature': 0.2,
                'num_ctx': 2048,
                'num_gpu': 1,
                'num_thread': 4,
                'seed': 42,
                'num_predict': 256,
                'top_k': 40,
                'top_p': 0.9,
                'repeat_penalty': 1.1
            }
        }
        # Use DEFAULT_CONFIG from obfuscation module
        obfuscator_config = DEFAULT_CONFIG.copy()
        self.obfuscator = PromptObfuscator(config=obfuscator_config)
        self.performance_tracker = PerformanceTracker()
        
    @property
    def metrics(self):
        """Get current performance metrics"""
        return self.performance_tracker.get_metrics()
    
    def process_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Process a prompt through obfuscation and model generation
        """
        @self.performance_tracker.track_request
        def _process(p: str):
            try:
                # Obfuscate the prompt
                obfuscated_prompt = self.obfuscator.obfuscate_prompt(p)
                
                # Generate response using Ollama with correct parameters
                response = ollama.generate(
                    model=self.config['model'],
                    prompt=obfuscated_prompt,
                    options=self.config['options']
                )
                
                return {
                    'original_prompt': p,
                    'obfuscated_prompt': obfuscated_prompt,
                    'response': response,
                    'status': 'success'
                }
                
            except Exception as e:
                return {
                    'status': 'error',
                    'error': str(e),
                    'original_prompt': p,
                    'obfuscated_prompt': None,
                    'response': None
                }
        
        return _process(prompt)
    
    def validate_response(self, response: Dict[str, Any]) -> bool:
        """
        Validate the response quality and security
        """
        if response['status'] == 'error':
            return False
            
        # Add validation logic here
        return True

def test_system():
    """
    Test the complete obfuscation system
    """
    system = LlamaObfuscationSystem()
    test_prompts = [
        "Write a very short hello world program.",
        "What is 2+2?",
        "Name three colors."
    ]
    
    for prompt in test_prompts:
        print(f"\nProcessing prompt: {prompt}")
        result = system.process_prompt(prompt)
        print(f"Status: {result['status']}")
        
        if result['status'] == 'success':
            print(f"Obfuscated prompt: {result['obfuscated_prompt']}")
            print(f"Model response: {result['response']['response']}")
        else:
            print(f"Error: {result['error']}")
            print("Trying to recover...")
    
    # Print performance metrics after all tests
    print("\nPerformance Summary:")
    system.performance_tracker.log_metrics()

if __name__ == "__main__":
    test_system()