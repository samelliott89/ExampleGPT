# ExampleGPT

A clean, educational implementation of a GPT (Generative Pre-trained Transformer) model built from scratch in PyTorch. This project demonstrates the core concepts of transformer-based language models with a complete training pipeline.

## Features

- **Complete Transformer Architecture**
  - Multi-head self-attention mechanism
  - Position-wise feedforward networks
  - Layer normalization and residual connections
  - Positional embeddings

- **Production-Ready Training**
  - Mixed precision training (AMP) for faster training
  - Gradient checkpointing for memory efficiency
  - Learning rate warmup and cosine decay scheduling
  - WandB integration for experiment tracking
  - Periodic checkpoint saving

- **Text Generation**
  - Temperature-based sampling
  - Nucleus (top-p) sampling
  - Customizable generation parameters

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

Dependencies:
- `torch` - PyTorch framework
- `tiktoken` - GPT-2 tokenizer
- `datasets` - HuggingFace datasets library
- `wandb` - Experiment tracking
- `numpy` - Numerical operations

## Model Architecture

The model implements the GPT architecture with the following default configuration:

- **Vocabulary Size**: 50,257 tokens (GPT-2 tokenizer)
- **Embedding Dimension**: 768
- **Attention Heads**: 12
- **Transformer Layers**: 12
- **Feedforward Dimension**: 3,072
- **Max Sequence Length**: 1,024 tokens
- **Total Parameters**: ~124M parameters

## Usage

### Training

Train the model from scratch on WikiText-103:

```bash
python example.py train
```

Training features:
- Automatic dataset download and tokenization (cached for subsequent runs)
- Mixed precision training for faster computation
- Gradient checkpointing to reduce memory usage
- WandB logging (optional, configure in `GPTConfig`)
- Checkpoints saved every 500 batches to `checkpoints/`

### Text Generation

Generate text using a trained model:

```bash
# Using the main saved model
python example.py generate --prompt "The history of" --max-tokens 100

# Using a specific checkpoint
python example.py generate --prompt "Once upon a time" --checkpoint checkpoints/checkpoint_epoch1_batch500.pt

# With custom sampling parameters
python example.py generate \
  --prompt "In the future" \
  --max-tokens 200 \
  --temperature 0.7 \
  --top-p 0.9
```

**Generation Parameters:**
- `--prompt, -p`: Starting text for generation
- `--max-tokens, -n`: Number of tokens to generate (default: 50)
- `--temperature, -t`: Sampling temperature (default: 0.8, lower = more focused)
- `--top-p`: Nucleus sampling threshold (default: 0.9)
- `--checkpoint, -c`: Load from specific checkpoint file

## Configuration

Modify `GPTConfig` class in `example.py` to customize:

```python
@dataclass
class GPTConfig:
    # Model architecture
    d_model = 768           # Embedding dimension
    num_heads = 12          # Number of attention heads
    num_layers = 12         # Number of transformer blocks
    
    # Training
    num_epochs = 3
    batch_size = 64
    learning_rate = 1e-4
    use_amp = True          # Mixed precision
    use_checkpointing = True # Gradient checkpointing
    
    # Logging
    use_wandb = True
    wandb_project = "gpt-training"
```

## Project Structure

```
ExampleGPT/
├── example.py              # Main implementation
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── gpt_model.pt           # Saved model weights (after training)
├── wikitext103_tokens.pt  # Cached tokenized dataset
└── checkpoints/           # Training checkpoints
```

## Implementation Details

### Multi-Head Attention
The attention mechanism splits the embedding into multiple heads, allowing the model to attend to different aspects of the input simultaneously.

### Gradient Checkpointing
Trades computation for memory by recomputing activations during the backward pass instead of storing them.

### Mixed Precision Training
Uses float16 for faster computation while maintaining float32 for critical operations, speeding up training on modern GPUs.

### Learning Rate Schedule
- Linear warmup for the first 100 steps
- Cosine decay from peak LR to minimum LR over remaining steps
- Helps with training stability and convergence

## Dataset

The model is trained on **WikiText-103**, a high-quality language modeling dataset containing ~103M tokens from Wikipedia articles. The dataset is automatically downloaded and tokenized on first run, then cached for subsequent training sessions.

## Notes

- **First Run**: Dataset download and tokenization takes a few minutes
- **GPU Recommended**: Training is significantly faster with CUDA-capable GPU
- **Memory**: ~8GB VRAM recommended for default configuration
- **Checkpoints**: Saved to `checkpoints/` directory during training (every 500 batches)

## Example Output

After training, you can generate text:

```
$ python example.py generate --prompt "The theory of relativity"
The theory of relativity was developed by Albert Einstein in the early 20th century...
```

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - GPT-2 paper
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Visual guide

## License

This is an educational project for learning transformer architectures.
