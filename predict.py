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

            # Sample from the distribution
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            # Ensure the token is valid (in the vocabulary range)
            if next_token < len(itos):
                generated.append(next_token)
                context = torch.cat([context, torch.tensor([[next_token]], device=device)], dim=1)
            else:
                print(f"Invalid token index: {next_token}. Skipping.")
                break

    # Convert back to text
    try:
        generated_text = ''.join([itos[i] for i in generated])
    except KeyError as e:
        print(f"KeyError: {e} while generating text.")
        return start_text  # Return the start_text in case of an error
    return start_text + generated_text

if __name__ == '__main__':
    # Load model
    model = load_model('out/ckpt_0000080.pt')

    # Generate text
    prompt = "The quick brown fox"
    print(f"\nPrompt: {prompt}")
    print("\nGenerated text:")
    print(generate(model, prompt, max_tokens=200, temperature=0.8))

