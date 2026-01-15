"""
Basic GPT Model with Training Loop

This file implements a simplified GPT (Generative Pre-trained Transformer) model
with a complete training loop. It demonstrates:
1. Token embeddings
2. Positional embeddings
3. Transformer blocks (pre-LN, GPT-2 style)
4. Causal self-attention mask (cached as a buffer)
5. Weight tying (token embedding <-> output head)
6. Training loop (AdamW + warmup/cosine LR + periodic checkpoints)
7. Text generation (temperature + top-k + top-p sampling)
"""

from dataclasses import dataclass
import argparse
from datetime import datetime
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint
from torch.amp import GradScaler
import math

import tiktoken
import wandb
from datasets import load_dataset

MODEL_PATH = "gpt_model.pt"
DATASET_CACHE = "wikitext103_tokens.pt"
RUNS_DIR = "runs"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
tokenizer = tiktoken.get_encoding("gpt2")


@dataclass
class GPTConfig:
    # Model architecture (scaled up for 94GB VRAM)
    vocab_size = 50257  # GPT-2 vocabulary size
    d_model = 768  # Embedding dimension (GPT-2 small size)
    num_heads = 12  # Number of attention heads
    num_layers = 12  # Number of transformer blocks
    d_ff = 3072  # Feedforward dimension (4 * d_model)
    max_seq_len = 1024  # Maximum sequence length
    dropout = 0.1  # Dropout rate

    # Training (with mixed precision + checkpointing)
    num_epochs = 3  # Number of training epochs
    learning_rate = 1e-4  # Peak learning rate (conservative)
    min_lr = 1e-5  # Minimum LR (10% of peak)
    warmup_steps = 100  # Linear warmup steps
    batch_size = 64  # Batch size (higher with memory optimizations)
    seq_len = 1024  # Sequence length (full context)
    num_workers = 4  # DataLoader workers
    use_amp = True  # Mixed precision training
    use_checkpointing = True  # Gradient checkpointing

    # Logging & checkpoints
    use_wandb = True  # Enable WandB logging
    wandb_project = "gpt-training"  # WandB project name
    checkpoint_interval = 500  # Save checkpoint every N batches

    # Generation
    temperature = 1.0  # Sampling temperature
    top_p = 0.9  # Nucleus sampling threshold
    top_k = 40  # Top-k sampling threshold


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""

    def __init__(
        self,
        d_model=GPTConfig.d_model,
        num_heads=GPTConfig.num_heads,
        dropout=GPTConfig.dropout,
    ):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.attn_dropout = nn.Dropout(dropout)

        # Query, Key, Value projections
        self.W_q = nn.Linear(d_model, d_model)  # This calculates the query vector
        self.W_k = nn.Linear(d_model, d_model)  # This calculates the key vector
        self.W_v = nn.Linear(d_model, d_model)  # This calculates the value vector
        self.W_o = nn.Linear(d_model, d_model)  # This calculates the output vector
        # Mark residual projection for GPT-2-style scaled init
        self.W_o._gpt2_residual_proj = True

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()

        # Project to Q, K, V
        Q = self.W_q(x)  # [batch, seq_len, d_model]
        K = self.W_k(x)
        V = self.W_v(x)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply mask (causal mask for GPT)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, V)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, d_model)

        # Output projection
        output = self.W_o(attn_output)
        return output


# Feedforward (MLP)
class FeedForward(nn.Module):
    """Position-wise feedforward network"""

    def __init__(self, d_model=GPTConfig.d_model, d_ff=GPTConfig.d_ff, dropout=GPTConfig.dropout):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        # Mark residual projection for GPT-2-style scaled init
        self.linear2._gpt2_residual_proj = True

    def forward(self, x):
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block: self-attention + feedforward"""

    def __init__(
        self,
        d_model=GPTConfig.d_model,
        num_heads=GPTConfig.num_heads,
        d_ff=GPTConfig.d_ff,
        dropout=GPTConfig.dropout,
    ):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.feedforward = FeedForward(d_model, d_ff, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    # Pre-LN transformer block (GPT-2 style)
    def forward(self, x, mask=None):
        # Attention sub-layer
        x = x + self.dropout(self.attention(self.norm1(x), mask))

        # MLP sub-layer
        x = x + self.dropout(self.feedforward(self.norm2(x)))

        return x


class SimpleGPT(nn.Module):
    """Simplified GPT model (GPT-2-ish, pre-LN) with optional gradient checkpointing"""

    def __init__(
        self,
        vocab_size=GPTConfig.vocab_size,
        d_model=GPTConfig.d_model,
        num_heads=GPTConfig.num_heads,
        num_layers=GPTConfig.num_layers,
        d_ff=GPTConfig.d_ff,
        max_seq_len=GPTConfig.max_seq_len,
        dropout=GPTConfig.dropout,
        use_checkpointing=False,
    ):
        super().__init__()
        self.use_checkpointing = use_checkpointing
        self.num_layers = num_layers
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # Causal mask buffer (avoid re-allocating every forward pass)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool)).view(
                1, 1, max_seq_len, max_seq_len
            ),
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.apply(self._init_weights)

        # Weight tying (GPT-2 style): share token embedding weights with output head
        # token_embedding.weight and head.weight are both [vocab_size, d_model]
        self.head.weight = self.token_embedding.weight

    # Initialize weights
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # GPT-2 scales residual projections by 1/sqrt(2 * num_layers)
            # to prevent activation growth in deep networks.
            std = 0.02
            if getattr(module, "_gpt2_residual_proj", False):
                std = 0.02 / math.sqrt(2 * self.num_layers)

            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)  # [out_features]
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=0.02
            )  # [num_embeddings, embedding_dim]
            # Note: Embedding layers don't have bias, only weight

    # Forward pass through the model
    def forward(self, idx, targets=None):
        """
        Args:
            idx: token indices [batch_size, seq_len]
            targets: target token indices [batch_size, seq_len] (for training)
        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        # GPT expects 2D input: [batch_size, seq_len]
        _, seq_len = idx.shape

        # Slice the cached causal mask to current sequence length
        mask = self.mask[:, :, :seq_len, :seq_len]

        # Token embeddings
        tok_emb = self.token_embedding(idx)  # [batch, seq_len, d_model]

        # Positional embeddings
        positions = torch.arange(0, seq_len, device=idx.device).unsqueeze(0)
        pos_emb = self.position_embedding(positions)  # [1, seq_len, d_model]

        # Combine embeddings
        x = self.dropout(tok_emb + pos_emb)  # [batch, seq_len, d_model]

        # Pass through transformer blocks (with optional checkpointing)
        for block in self.blocks:
            if self.use_checkpointing and self.training:
                # Checkpoint: recompute activations during backward pass to save memory
                x = checkpoint(block, x, mask, use_reentrant=False)
            else:
                x = block(x, mask)

        # Final layer norm
        x = self.ln_f(x)

        # Language modeling head
        logits = self.head(x)  # [batch, seq_len, vocab_size]

        return logits

    def generate(self, idx, max_new_tokens, temperature=1.0, top_p=0.9, top_k=0):
        """
        Generate tokens with temperature + (optional) top-k + top-p sampling.

        Args:
            idx: Starting token indices [batch, seq_len]
            max_new_tokens: Number of tokens to generate
            temperature: Controls randomness (lower = more focused)
            top_p: Nucleus sampling threshold (0.9 = keep tokens summing to 90% prob)
            top_k: Keep only the top-k tokens (0 disables)
        """
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Crop context if too long
                idx_cond = idx[:, -GPTConfig.max_seq_len :]

                # Get predictions
                logits = self(idx_cond)  # [batch, seq_len, vocab_size]

                # Focus on last time step, apply temperature
                logits = logits[:, -1, :] / temperature  # [batch, vocab_size]

                # Apply top-k sampling (optional)
                if top_k and top_k > 0:
                    top_k = min(top_k, logits.size(-1))
                    kth_values = (
                        torch.topk(logits, k=top_k, dim=-1).values[:, -1].unsqueeze(-1)
                    )
                    logits = logits.masked_fill(logits < kth_values, float("-inf"))

                # Apply top-p (nucleus) sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(
                        F.softmax(sorted_logits, dim=-1), dim=-1
                    )

                    # Remove tokens with cumulative prob above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Keep at least one token
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[
                        :, :-1
                    ].clone()
                    sorted_indices_to_remove[:, 0] = False

                    # Scatter back to original indices
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float("-inf")

                # Apply softmax to get probabilities
                probs = F.softmax(logits, dim=-1)

                # Sample next token
                idx_next = torch.multinomial(probs, num_samples=1)

                # Append to sequence
                idx = torch.cat([idx, idx_next], dim=1)

        return idx


# Training Dataset
class TextDataset(Dataset):
    """Dataset for language modeling with non-overlapping chunks"""

    def __init__(self, data, seq_len, stride=None):
        self.data = data
        self.seq_len = seq_len
        self.stride = stride or seq_len  # Default: no overlap

    def __len__(self):
        return (len(self.data) - self.seq_len) // self.stride

    def __getitem__(self, idx):
        start = idx * self.stride
        x = self.data[start : start + self.seq_len]
        y = self.data[start + 1 : start + self.seq_len + 1]
        return x, y


def get_dataset():
    """Load WikiText-103 dataset, using cached tokenized version if available"""
    import os

    # Check for cached tokenized data
    if os.path.exists(DATASET_CACHE):
        print(f"  Loading cached tokens from {DATASET_CACHE}...")
        data = torch.load(DATASET_CACHE, weights_only=True)
        return data

    # Download and tokenize
    print("  Downloading WikiText-103 (first run only)...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

    print("  Tokenizing (~103M tokens, this takes a few minutes)...")
    text = "\n".join(dataset["text"])
    token_ids = tokenizer.encode(text)
    data = torch.tensor(token_ids, dtype=torch.long)

    # Cache to disk
    print(f"  Saving tokenized data to {DATASET_CACHE}...")
    torch.save(data, DATASET_CACHE)

    return data


def create_dataloader(data, batch_size, seq_len, num_workers):
    """Create DataLoader with parallel workers and non-overlapping chunks"""
    stride = seq_len  # No overlap
    dataset = TextDataset(data, seq_len, stride=stride)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,  # Faster GPU transfer
        drop_last=True,  # Avoid variable batch sizes
    )
    return loader


# Training Loop & Learning Rate Schedule


def get_lr(step, total_steps):
    """Learning rate schedule with warmup and cosine decay"""
    warmup_steps = GPTConfig.warmup_steps

    # Linear warmup
    if step < warmup_steps:
        return GPTConfig.learning_rate * (step / warmup_steps)

    # Cosine decay after warmup
    decay_steps = total_steps - warmup_steps
    progress = (step - warmup_steps) / decay_steps
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    return (
        GPTConfig.min_lr + (GPTConfig.learning_rate - GPTConfig.min_lr) * cosine_decay
    )


def train_epoch(
    model,
    dataloader,
    optimizer,
    scaler,
    device,
    epoch,
    global_step,
    total_steps,
    checkpoint_dir,
):
    """Train for one epoch with mixed precision and LR schedule"""
    model.train()
    total_loss = 0
    num_batches = 0

    for x, y in dataloader:
        # Update learning rate
        lr = get_lr(global_step, total_steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Mixed precision forward pass
        with torch.autocast(device_type="cuda", enabled=GPTConfig.use_amp):
            logits = model(x)
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = y.view(-1)
            loss = F.cross_entropy(logits_flat, targets_flat)

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()

        # Unscale before clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Optimizer step with scaler
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1
        global_step += 1

        # Logging
        if num_batches % 100 == 0:
            print(f"    Batch {num_batches}, Loss: {loss.item():.4f}, LR: {lr:.2e}")
            if GPTConfig.use_wandb:
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/lr": lr,
                        "train/step": global_step,
                    }
                )

        # Periodic checkpoint
        if num_batches % GPTConfig.checkpoint_interval == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(
                checkpoint_dir, f"checkpoint_epoch{epoch + 1}_batch{num_batches}.pt"
            )
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "loss": loss.item(),
                },
                checkpoint_path,
            )
            print(f"    Checkpoint saved: {checkpoint_path}")

    return total_loss / num_batches, global_step


# Training


def train(model, device, checkpoint_dir):
    """Train the model with mixed precision and WandB logging"""
    print("=" * 70)
    print("Training")
    print("=" * 70)

    # Initialize WandB
    if GPTConfig.use_wandb:
        wandb.init(
            project=GPTConfig.wandb_project,
            config={
                "d_model": GPTConfig.d_model,
                "num_heads": GPTConfig.num_heads,
                "num_layers": GPTConfig.num_layers,
                "batch_size": GPTConfig.batch_size,
                "seq_len": GPTConfig.seq_len,
                "learning_rate": GPTConfig.learning_rate,
                "min_lr": GPTConfig.min_lr,
                "warmup_steps": GPTConfig.warmup_steps,
                "num_epochs": GPTConfig.num_epochs,
                "use_amp": GPTConfig.use_amp,
                "use_checkpointing": GPTConfig.use_checkpointing,
            },
        )

    # Load dataset
    print("\nDownloading and preparing dataset...")
    data = get_dataset()
    print(f"  Dataset size: {len(data):,} tokens")

    # Create DataLoader
    print(
        f"  Creating DataLoader (batch_size={GPTConfig.batch_size}, seq_len={GPTConfig.seq_len}, workers={GPTConfig.num_workers})"
    )
    dataloader = create_dataloader(
        data, GPTConfig.batch_size, GPTConfig.seq_len, GPTConfig.num_workers
    )
    print(f"  Batches per epoch: {len(dataloader):,} (non-overlapping chunks)")

    # Optimizer and scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=GPTConfig.learning_rate)
    scaler = GradScaler("cuda", enabled=GPTConfig.use_amp)

    # Calculate total steps for LR schedule
    total_steps = len(dataloader) * GPTConfig.num_epochs

    print(f"  Mixed precision (AMP): {GPTConfig.use_amp}")
    print(f"  Gradient checkpointing: {GPTConfig.use_checkpointing}")
    print(f"  LR schedule: warmup {GPTConfig.warmup_steps} steps, then cosine decay")
    print(f"  Peak LR: {GPTConfig.learning_rate}, Min LR: {GPTConfig.min_lr}")

    # Training loop
    global_step = 0
    for epoch in range(GPTConfig.num_epochs):
        print(f"\nEpoch {epoch + 1}/{GPTConfig.num_epochs}")
        loss, global_step = train_epoch(
            model,
            dataloader,
            optimizer,
            scaler,
            device,
            epoch,
            global_step,
            total_steps,
            checkpoint_dir,
        )
        print(f"  Average Loss: {loss:.4f}")

        if GPTConfig.use_wandb:
            wandb.log({"train/epoch_loss": loss, "train/epoch": epoch + 1})

    if GPTConfig.use_wandb:
        wandb.finish()

    print("\nTraining Complete!")
    return model


# Generation
def generate_text(
    model,
    prompt,
    max_new_tokens=50,
    temperature=1.0,
    top_p=0.9,
    top_k=0,
    device="cpu",
):
    """Generate text from a prompt using temperature + top-k + top-p sampling."""
    model.eval()

    # Tokenize the prompt
    token_ids = tokenizer.encode(prompt)
    context = torch.tensor(token_ids).unsqueeze(0).to(device)  # [1, seq_len]

    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            context,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

    # Decode back to text
    generated_text = tokenizer.decode(generated_ids[0].tolist())

    return generated_text


def create_model(device):
    """Create and return a new GPT model"""
    model = SimpleGPT(
        vocab_size=GPTConfig.vocab_size,
        d_model=GPTConfig.d_model,
        num_heads=GPTConfig.num_heads,
        num_layers=GPTConfig.num_layers,
        d_ff=GPTConfig.d_ff,
        max_seq_len=GPTConfig.max_seq_len,
        dropout=GPTConfig.dropout,
        use_checkpointing=GPTConfig.use_checkpointing,
    ).to(device)
    return model


def save_model(model, path=MODEL_PATH):
    """Save model weights"""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(device, path=MODEL_PATH):
    """Load model weights"""
    model = create_model(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    print(f"Model loaded from {path}")
    return model


def load_from_checkpoint(device, checkpoint_path):
    """Load model from a training checkpoint"""
    model = create_model(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    print(f"Model loaded from checkpoint: {checkpoint_path}")
    print(
        f"  Epoch: {checkpoint['epoch'] + 1}, Step: {checkpoint['global_step']}, Loss: {checkpoint['loss']:.4f}"
    )
    return model


def run_train(device):
    """Train the model and save it"""

    # Create a unique run folder per training session
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join(RUNS_DIR, run_id)
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    print(f"\nRun dir: {run_dir}")

    print("=" * 70)
    print("Training GPT Model")
    print("=" * 70)

    print(f"\nUsing device: {device}")
    print("\nModel config:")
    print(f"  Vocabulary size: {GPTConfig.vocab_size}")
    print(f"  Embedding dimension: {GPTConfig.d_model}")
    print(f"  Number of heads: {GPTConfig.num_heads}")
    print(f"  Number of layers: {GPTConfig.num_layers}")
    print(f"  Max sequence length: {GPTConfig.max_seq_len}")

    model = create_model(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {num_params:,}")

    # Train, writing checkpoints into this run folder
    model = train(model, device, checkpoints_dir)

    # Save final model into the run folder
    save_model(model, path=os.path.join(run_dir, "gpt_model.pt"))

    # (Optional convenience) also overwrite top-level gpt_model.pt as "latest"
    save_model(model, path=MODEL_PATH)

    return model


def run_generate(
    prompt, device, max_tokens=50, temperature=0.8, top_p=0.9, top_k=0, checkpoint=None
):
    """Load model and generate text from prompt"""
    if checkpoint:
        model = load_from_checkpoint(device, checkpoint)
    else:
        model = load_model(device)
    generated = generate_text(
        model,
        prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        device=device,
    )
    print(f"\n{generated}")


def main():
    parser = argparse.ArgumentParser(description="GPT Training and Generation")
    parser.add_argument(
        "mode", choices=["train", "generate"], help="Mode: train or generate"
    )
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        default="The history of",
        help="Prompt for generation",
    )
    parser.add_argument(
        "--max-tokens", "-n", type=int, default=50, help="Max tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        default=GPTConfig.temperature,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=GPTConfig.top_p,
        help="Nucleus sampling threshold (0.9 = top 90%% prob mass)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=GPTConfig.top_k,
        help="Top-k sampling threshold (40 = keep top 40 tokens)",
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        default=None,
        help="Load from checkpoint file (for generate)",
    )

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "train":
        run_train(device)
    elif args.mode == "generate":
        run_generate(
            args.prompt,
            device,
            args.max_tokens,
            args.temperature,
            args.top_p,
            args.top_k,
            args.checkpoint,
        )


if __name__ == "__main__":
    main()
