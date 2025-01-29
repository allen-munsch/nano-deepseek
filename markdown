# Data Preparation Guide

This guide explains how to prepare training data using `prepare.py` and use it with `train.py`.

## Overview

The training pipeline consists of two main steps:
1. Data preparation using `prepare.py`
2. Model training using `train.py`

## Using prepare.py

The `prepare.py` script converts raw text data into binary format for efficient training:

```bash
python data/openwebtext/prepare.py
```

This will:
1. Download sample text data if not present
2. Convert text to integer tokens
3. Create training/validation splits
4. Save binary files:
   - `train.bin`: Training dataset
   - `val.bin`: Validation dataset 
   - `meta.pkl`: Vocabulary and encoding information

### Input Format

The script expects raw text input in `data/openwebtext/input.txt`. You can either:
- Use the default Shakespeare dataset that gets downloaded automatically
- Replace with your own text file before running

### Output Files

The script generates:
- `train.bin`: ~90% of data for training
- `val.bin`: ~10% of data for validation
- `meta.pkl`: Contains:
  - `vocab_size`: Number of unique tokens
  - `itos`: Integer-to-string mapping
  - `stoi`: String-to-integer mapping

## Using with train.py

1. First prepare your data:
```bash
python data/openwebtext/prepare.py
```

2. Then start training:
```bash
python train.py
```

The training script automatically:
- Loads the binary data files
- Uses memory mapping for efficient data loading
- Handles batching and shuffling

## Customization

To use your own dataset:

1. Place your text file at `data/openwebtext/input.txt`
2. Run prepare.py to convert it
3. The resulting binary files will be ready for training

## Monitoring

During preparation you'll see:
- Dataset size in characters
- Vocabulary size
- Number of training/validation tokens

During training you'll see:
- Loss metrics
- Training speed
- Memory usage
