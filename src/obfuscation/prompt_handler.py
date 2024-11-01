import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple
import torch.optim as optim
from transformers import BertModel, BertTokenizer
import random

class PromptObfuscator(torch.nn.Module):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        
        self.config = config or {
            'bert_model': 'bert-base-uncased',
            'bert_dim': 768,  # BERT's hidden size
            'device': 'cpu',
            'learning_rate': 0.01,
            'max_iterations': 5,
            'similarity_threshold': 0.85,
            'top_k': 5  # Number of similar tokens to consider
        }
        
        self.device = torch.device(self.config['device'])
        
        try:
            # Try to load BERT
            self.bert = BertModel.from_pretrained(
                self.config['bert_model'],
                local_files_only=True  # Try local first
            )
        except:
            print("Downloading BERT model (one-time setup)...")
            # Download and cache the model
            self.bert = BertModel.from_pretrained(
                self.config['bert_model'],
                token="hf_..."  # Your HuggingFace token
            )
            # Save model locally
            self.bert.save_pretrained("./models/bert")
            
        try:
            self.tokenizer = BertTokenizer.from_pretrained(
                self.config['bert_model'],
                local_files_only=True
            )
        except:
            print("Downloading BERT tokenizer...")
            self.tokenizer = BertTokenizer.from_pretrained(
                self.config['bert_model'],
                token="hf_..."  # Your HuggingFace token
            )
            self.tokenizer.save_pretrained("./models/bert")
            
        self.bert.eval()
        self.bert = self.bert.to(self.device)
        
        # Task-specific templates
        self.task_templates = {
            'code': {
                'prefix': 'implement',
                'suffix': 'output'
            },
            'math': {
                'prefix': 'calculate',
                'suffix': 'result'
            },
            'list': {
                'prefix': 'enumerate',
                'suffix': 'items'
            }
        }
        
    def _detect_task_type(self, prompt: str) -> Dict[str, Any]:
        """Detect task type and constraints"""
        prompt_lower = prompt.lower()
        task_info = {
            'type': 'general',
            'count': None,
            'constraints': []
        }
        
        # Detect task type
        if any(word in prompt_lower for word in ['write', 'program', 'code']):
            task_info['type'] = 'code'
        elif any(word in prompt_lower for word in ['what', 'calculate', '+', '-', '*', '/']):
            task_info['type'] = 'math'
        elif any(word in prompt_lower for word in ['name', 'list', 'enumerate']):
            task_info['type'] = 'list'
            
        # Extract count if present
        numbers = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5}
        for word, num in numbers.items():
            if word in prompt_lower:
                task_info['count'] = num
                break
                
        return task_info
        
    def _collision_to_text(self, collision: torch.Tensor, original_prompt: str) -> str:
        """Convert collision embeddings back to text while preserving task semantics"""
        # Get task type
        task_info = self._detect_task_type(original_prompt)
        
        # Get similar tokens for each position
        result_tokens = []
        
        for pos_embedding in collision:
            # Compute similarity with BERT vocabulary
            similarity = torch.matmul(
                pos_embedding,
                self.bert.embeddings.word_embeddings.weight.t()
            )
            
            # Get top-k similar tokens
            _, indices = similarity.topk(self.config['top_k'])
            tokens = [self.tokenizer.convert_ids_to_tokens(idx.item()) for idx in indices]
            
            # Filter and clean tokens
            valid_tokens = []
            for token in tokens:
                token = token.replace('##', '')
                if len(token) > 1 and not token.startswith('['):
                    valid_tokens.append(token)
                    
            if valid_tokens:
                result_tokens.append(random.choice(valid_tokens))
                
        # Preserve task-specific tokens
        if task_info['type'] == 'math':
            # Keep mathematical operations
            for i, token in enumerate(original_prompt.split()):
                if any(c in '0123456789+-*/' for c in token):
                    if i < len(result_tokens):
                        result_tokens[i] = token
                        
        return ' '.join(result_tokens)
        
    def obfuscate_prompt(self, prompt: str) -> str:
        """Implement soft prompt obfuscation using BERT's embedding space"""
        try:
            # 1. Get BERT embeddings
            original_embeddings = self._text_to_embeddings(prompt)
            
            # 2. Find functional collision in BERT's embedding space
            collision = self._find_functional_collision(original_embeddings)
            
            # 3. Convert back to text while preserving task semantics
            obfuscated = self._collision_to_text(collision, prompt)
            
            return obfuscated
            
        except Exception as e:
            print(f"Error in prompt obfuscation: {e}")
            return prompt
            
    def _text_to_embeddings(self, text: str) -> torch.Tensor:
        """Convert text to BERT embedding space"""
        # Tokenize with attention mask
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_attention_mask=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get BERT embeddings with attention
        with torch.no_grad():
            outputs = self.bert(**inputs)
            # Use attention mask to get meaningful token embeddings
            embeddings = outputs.last_hidden_state * inputs['attention_mask'].unsqueeze(-1)
            
        return embeddings.squeeze(0)  # [seq_len, hidden_dim]
        
    def _find_functional_collision(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Find functional collision in BERT embedding space"""
        # Initialize with noise
        collision = embeddings.clone() + torch.randn_like(embeddings) * 0.1
        collision.requires_grad_(True)
        
        # Track best result
        best_collision = collision.clone().detach()
        best_loss = float('inf')
        
        # Optimize to find collision
        optimizer = optim.Adam([collision], lr=self.config['learning_rate'])
        
        for _ in range(self.config['max_iterations']):
            optimizer.zero_grad()
            
            # 1. Functional similarity loss (maintain task)
            # Compare sequence-level embeddings
            seq_sim = F.cosine_similarity(
                embeddings.mean(dim=0),
                collision.mean(dim=0),
                dim=0
            )
            sim_loss = 1 - seq_sim
            
            # 2. Diversity loss (ensure different surface form)
            # Encourage different token-level embeddings
            token_div = F.cosine_similarity(
                embeddings.view(-1),
                collision.view(-1),
                dim=0
            )
            div_loss = token_div
            
            # Total loss with careful weighting
            loss = 0.7 * sim_loss + 0.3 * div_loss
            
            # Backward pass
            loss.backward(retain_graph=True)
            optimizer.step()
            
            # Store best result
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_collision = collision.clone().detach()
                
            # Clear computation graph
            optimizer.zero_grad(set_to_none=True)
            
        return best_collision