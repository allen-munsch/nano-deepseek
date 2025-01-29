"""
Prepare the dataset using GPT-2 BPE tokenization.
Will save train.bin, val.bin containing the token ids, and meta.pkl containing the
encoder information and other metadata.
"""
import os
import pickle
import requests
import numpy as np
import tiktoken

def download_data():
    """Download the dataset if not present"""
    input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
    if not os.path.exists(input_file_path):
        data_url = 'https://gist.githubusercontent.com/allen-munsch/2971dbe7fd31d497e60550f58b377491/raw/d85ed486d0b6046a0b05773ef252c1c2e4153781/input.txt'
        with open(input_file_path, 'w') as f:
            f.write(requests.get(data_url).text)
    return input_file_path

def load_tokenizer():
    """Load the GPT-2 tokenizer"""
    return tiktoken.get_encoding("gpt2")

def process_data():
    """Process the data using GPT-2 tokenizer"""
    # Load data
    input_file_path = download_data()
    with open(input_file_path, 'r') as f:
        data = f.read()
    print(f"Length of dataset in characters: {len(data):,}")
    
    # Initialize tokenizer
    enc = load_tokenizer()
    print(f"Vocabulary size: {enc.n_vocab:,}")
    
    # Tokenize the entire text
    tokens = enc.encode_ordinary(data)
    print(f"Number of tokens: {len(tokens):,}")
    
    # Create train/val split
    n = len(tokens)
    train_tokens = tokens[:int(n*0.9)]
    val_tokens = tokens[int(n*0.9):]
    
    print(f"Train has {len(train_tokens):,} tokens")
    print(f"Val has {len(val_tokens):,} tokens")
    
    # Save as binary files
    train_ids = np.array(train_tokens, dtype=np.uint16)
    val_ids = np.array(val_tokens, dtype=np.uint16)
    
    train_path = os.path.join(os.path.dirname(__file__), 'train.bin')
    val_path = os.path.join(os.path.dirname(__file__), 'val.bin')
    meta_path = os.path.join(os.path.dirname(__file__), 'meta.pkl')
    
    train_ids.tofile(train_path)
    val_ids.tofile(val_path)
    
    # Save metadata
    meta = {
        'vocab_size': enc.n_vocab,
        'encoding_name': "gpt2",
    }
    
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)

if __name__ == '__main__':
    process_data()
