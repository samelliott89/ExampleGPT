# ExampleGPT

A clean, educational GPT-style language model built from scratch in PyTorch. Contains lots of comments for explanation.

## Features

- GPT-style Transformer (causal self-attention)
- Training loop with warmup + cosine LR schedule
- Checkpointing + sampling (temperature / top-p / top-k)
 
## Train/Loss

<img width="3696" height="1594" alt="simplesnap-1-15-2026_at_10-15-10" src="https://github.com/user-attachments/assets/85c0a7a2-289e-44dd-b676-f0e837216f36" />

## Requirements

```bash
pip install -r requirements.txt
```

### Training

```bash
python3 example.py train
```

### Text Generation

```bash
# Using the main saved model
python3 example.py generate --prompt "The history of" --max-tokens 100

# Using a specific checkpoint
python3 example.py generate --prompt "Once upon a time" --checkpoint checkpoints/checkpoint_epoch1_batch500.pt

# With custom sampling parameters
python3 example.py generate \
  --prompt "In the future" \
  --max-tokens 200 \
  --temperature 0.7 \
  --top-p 0.9 \
  --top-k 40
```

**Generation Parameters:**
- `--prompt, -p`: Starting text for generation
- `--max-tokens, -n`: Number of tokens to generate (default: 50)
- `--temperature, -t`: Sampling temperature (default: 0.8, lower = more focused)
- `--top-p`: Nucleus sampling threshold (default: 0.9)
- `--top-k`: Top-k sampling (default: 40, 0 disables)
- `--checkpoint, -c`: Load from specific checkpoint file

## Example Output

After training, you can generate text:

```
$ python3 example.py generate --prompt "The theory of relativity"
The theory of relativity was developed by Albert Einstein in the early 20th century...
```

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - GPT-2 paper
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Visual guide

## License

This is an educational project for learning transformer architectures.
