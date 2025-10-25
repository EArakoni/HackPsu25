from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

class ModelManager:
    def __init__(self):
        self.available_models = {
            'phi3-mini': {
                'name': 'microsoft/Phi-3-mini-4k-instruct',
                'size': '2.3GB',
                'speed': 'fast',
                'quality': 'excellent',
                'description': 'Best for detailed study plans'
            },
            'tinyllama': {
                'name': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
                'size': '600MB',
                'speed': 'very fast',
                'quality': 'good',
                'description': 'Fast and lightweight'
            }
        }
        self.current_model = None
        self.current_tokenizer = None
        self.model_name = None
        
        # Check if GPU is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
    
    def load_model(self, model_key='tinyllama'):
        """Load a model by key"""
        if model_key not in self.available_models:
            raise ValueError(f"Model {model_key} not available")
        
        model_info = self.available_models[model_key]
        model_name = model_info['name']
        
        print(f"Loading model: {model_name}...")
        print(f"This may take a few minutes on first run (downloading {model_info['size']})...")
        
        try:
            # Load tokenizer
            self.current_tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            # Load model
            self.current_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            if self.device == "cpu":
                self.current_model = self.current_model.to(self.device)
            
            self.model_name = model_key
            print(f"✓ Model {model_name} loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def generate(self, prompt, max_length=800, temperature=0.7):
        """Generate text from prompt"""
        if self.current_model is None or self.current_tokenizer is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        try:
            # Prepare input
            inputs = self.current_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.current_model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.current_tokenizer.eos_token_id
                )
            
            # Decode output
            generated_text = self.current_tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )
            
            # Remove the input prompt from output (some models include it)
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return generated_text
            
        except Exception as e:
            print(f"Error generating text: {e}")
            return None
    
    def get_available_models(self):
        """Return list of available models"""
        return self.available_models
    
    def get_current_model(self):
        """Return current model name"""
        return self.model_name