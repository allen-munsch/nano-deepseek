# Getting Started

## Prerequisites

- Python 3.12+
- CUDA capable GPU (for GPU training) or Apple Silicon Mac
- 16GB+ RAM recommended

## Installation

1. Install PyTorch and dependencies:

```bash
# For CUDA support:
pip install torch torchvision torchaudio

# For Apple Silicon:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
```

2. Clone the repository:

```bash
git clone <repository-url>
cd <repository-name>
```

## Quick Start

1. Prepare your training data in the `data` directory
2. Run basic training:

```bash
python train.py
```

See [Training Guide](training_guide.md) for detailed instructions.
