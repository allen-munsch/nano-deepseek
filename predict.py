import os
import pickle
import torch
import numpy as np
from train import ModelWrapper, create_config, get_device

def load_model(checkpoint_path='out/best_ckpt.pt'):
    """Load the trained model from checkpoint"""
    print(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Create model with same config
    config = checkpoint['config']
    model = ModelWrapper(config)

    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def load_encoder():
    """Load the tokenizer from meta.pkl"""
    with open('data/openwebtext/meta.pkl', 'rb') as f:
        meta = pickle.load(f)
    return tiktoken.get_encoding(meta['encoding_name'])

def generate(model, start_text, max_tokens=100, temperature=0.8):
    """Generate text starting from start_text"""
    # Load tokenizer
    enc = load_encoder()

    # Convert start text to tokens
    context = torch.tensor(enc.encode_ordinary(start_text), dtype=torch.long).unsqueeze(0)

    # Move model and context to device
    device = get_device()
    model = model.to(device)
    context = context.to(device)

    # Generate
    model.eval()
    generated = []
    with torch.no_grad():
        for _ in range(max_tokens):
            # Get predictions
            logits, _ = model(context, None)
            logits = logits[:, -1, :] / temperature
            
            # Apply sophisticated sampling
            # Filter with top-k
            top_k = 40
            top_k_logits, top_k_indices = torch.topk(logits, top_k)
            
            # Apply softmax with temperature
            probs = F.softmax(top_k_logits, dim=-1)
            
            # Sample from filtered distribution
            next_token_idx = torch.multinomial(probs, num_samples=1)
            next_token = top_k_indices[0, next_token_idx[0]]

            # Stop if we generate end token
            if next_token.item() == enc.eot_token:
                break

            # Append to generated sequence
            generated.append(next_token.item())
            context = torch.cat([context, next_token.unsqueeze(0).unsqueeze(0)], dim=1)

            # Check generated text quality
            if len(generated) > 20:  # Check larger chunks
                current_text = enc.decode(generated[-20:])
                # Stop if generating repetitive or garbage text
                if len(set(current_text)) < 5 or current_text.count(current_text[:3]) > 2:
                    break

    # Convert tokens back to text
    generated_tokens = generated
    generated_text = enc.decode(generated_tokens)
    return start_text + generated_text

if __name__ == '__main__':
    # Load model
    model = load_model()

    # Generate text
    prompt = "The quick brown fox"
    print(f"\nPrompt: {prompt}")
    print("\nGenerated text:")
    print(generate(model, prompt, max_tokens=200, temperature=0.8))
