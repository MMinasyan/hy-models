# HY-Models

A Python module for transformer-based models implementation with PyTorch.

## Overview

HY-Models provides implementations of custom transformer-based models for various machine learning tasks. The package includes:

- **ArtForCausalLM**: A causal language model implementation
- **AutoEditForConditionalGeneration**: A sequence-to-sequence model for conditional text generation

## Key Features

- **HuggingFace Compatible**: Built on the transformers library architecture
- **Modern Architectures**: Implements state-of-the-art transformer designs
- **Flexible Configuration**: Highly customizable model parameters
- **Multiple Models**: Causal LM and sequence-to-sequence implementations

## Installation

```bash
pip install git+https://github.com/MMinasyan/hy-models.git
```

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.5.0
- Transformers ≥ 4.0.0

## Usage

[Usage documentation will be added later]

## Documentation

Detailed documentation for each model and component:

- **ArtForCausalLM**: A decoder-only architecture for text generation
  - Suitable for language modeling and text completion
  - Follows modern LLM design patterns

- **AutoEditForConditionalGeneration**: An encoder-decoder architecture
  - Designed for text editing and transformation tasks
  - Supports both token and convolutional embeddings for encoder

## Testing

Run the test suite:

```bash
python -m pytest tests/test_models.py -v
```

## License

This project is licensed under the terms included in the LICENSE file.

## Author

- Mkrtich Minasyan (mkrtichm@outlook.com)
