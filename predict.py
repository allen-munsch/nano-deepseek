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
    """Load the character encoder from meta.pkl"""
    with open('data/openwebtext/meta.pkl', 'rb') as f:
        meta = pickle.load(f)
    return meta['stoi'], meta['itos']

def generate(model, start_text, max_tokens=100, temperature=0.8):
    """Generate text starting from start_text"""
    # Load encoder
    stoi, itos = load_encoder()

    # Convert start text to tensor
    context = torch.tensor([stoi[c] for c in start_text], dtype=torch.long).unsqueeze(0)

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

            # Apply softmax with temperature
            probs = torch.softmax(logits, dim=-1)

            # Filter unlikely tokens (optional)
            top_k = 40  # Keep only top k tokens
            top_k_probs, top_k_indices = torch.topk(probs, top_k)
            
            # Renormalize probabilities for top-k
            top_k_probs = top_k_probs / top_k_probs.sum()

            # Sample from filtered distribution
            next_token_idx = torch.multinomial(top_k_probs, num_samples=1)
            next_token = top_k_indices[0, next_token_idx[0]]

            # Stop if we generate a newline or special token
            if itos[next_token.item()] in ['\n', '<|endoftext|>']:
                break

            # Append to generated sequence
            generated.append(next_token.item())
            context = torch.cat([context, next_token.unsqueeze(0).unsqueeze(0)], dim=1)

            # Optional: Stop if we generate too many special characters in a row
            if len(generated) > 3:
                last_chars = ''.join([itos[i] for i in generated[-3:]])
                if not any(c.isalnum() or c.isspace() for c in last_chars):
                    break

    # Convert back to text
    generated_text = ''.join([itos[i] for i in generated])
    return start_text + generated_text

if __name__ == '__main__':
    # Load model
    model = load_model()

    # Generate text
    prompt = "The quick brown fox"
    print(f"\nPrompt: {prompt}")
    print("\nGenerated text:")
    print(generate(model, prompt, max_tokens=200, temperature=0.8))
