"""
Checkpoint/resume training script for the small GPT model.

Designed to run inside a time-limited CI job (e.g. a GitHub Actions runner
that gets cut off every few hours) and pick up exactly where it left off
on the next run. Run it the same way locally too if you ever want to.
"""
import glob
import math
import os
import time

import numpy as np
import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

from model import GPT, ModelConfig

# ---- paths ----
DATA_DIR = "data"
CKPT_DIR = "checkpoints"
CKPT_PATH = os.path.join(CKPT_DIR, "checkpoint.pt")
TOKENIZER_PATH = os.path.join(CKPT_DIR, "tokenizer.json")

# ---- training settings ----
BATCH_SIZE = 16
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 200
MAX_GRAD_NORM = 1.0
LOG_EVERY = 20
SAVE_EVERY = 200  # also save on a step interval, whichever comes first
TIME_BUDGET_SECONDS = float(os.environ.get("TIME_BUDGET_SECONDS", 3.5 * 3600))
TOTAL_TARGET_STEPS = int(os.environ.get("TOTAL_TARGET_STEPS", 50000))

os.makedirs(CKPT_DIR, exist_ok=True)


def load_or_train_tokenizer():
    if os.path.exists(TOKENIZER_PATH):
        return Tokenizer.from_file(TOKENIZER_PATH)

    files = glob.glob(os.path.join(DATA_DIR, "*.txt"))
    if not files:
        raise SystemExit(f"No .txt files found in {DATA_DIR}/. Add your training text there first.")

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel()
    trainer = BpeTrainer(
        vocab_size=ModelConfig.vocab_size,
        special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"],
    )
    tokenizer.train(files, trainer)
    tokenizer.save(TOKENIZER_PATH)
    print(f"Trained new tokenizer, vocab size {tokenizer.get_vocab_size()}")
    return tokenizer


def load_dataset(tokenizer):
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.txt")))
    text = ""
    for f in files:
        with open(f, "r", encoding="utf-8", errors="ignore") as fh:
            text += fh.read() + "\n"
    ids = tokenizer.encode(text).ids
    return np.array(ids, dtype=np.int32)


def get_batch(data, context_length, batch_size, device):
    max_start = len(data) - context_length - 1
    if max_start < 1:
        raise SystemExit("Dataset is too small for the configured context length. Add more text to data/.")
    starts = np.random.randint(0, max_start, size=batch_size)
    x = np.stack([data[s:s + context_length] for s in starts])
    y = np.stack([data[s + 1:s + 1 + context_length] for s in starts])
    return (
        torch.tensor(x, dtype=torch.long, device=device),
        torch.tensor(y, dtype=torch.long, device=device),
    )


def lr_schedule(step):
    if step < WARMUP_STEPS:
        return LEARNING_RATE * step / max(1, WARMUP_STEPS)
    progress = (step - WARMUP_STEPS) / max(1, TOTAL_TARGET_STEPS - WARMUP_STEPS)
    progress = min(progress, 1.0)
    return 0.5 * LEARNING_RATE * (1 + math.cos(math.pi * progress))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    tokenizer = load_or_train_tokenizer()
    data = load_dataset(tokenizer)
    print(f"Dataset: {len(data):,} tokens, vocab={ModelConfig.vocab_size}, device={device}")

    cfg = ModelConfig()
    model = GPT(cfg).to(device)
    print(f"Model: {model.num_params():,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    step = 0
    if os.path.exists(CKPT_PATH):
        ckpt = torch.load(CKPT_PATH, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        step = ckpt["step"]
        print(f"Resumed from checkpoint at step {step:,}")
    else:
        print("No checkpoint found, starting fresh")

    def save_checkpoint():
        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step},
            CKPT_PATH,
        )
        print(f"Saved checkpoint at step {step:,}")

    if step >= TOTAL_TARGET_STEPS:
        print("Training already reached TOTAL_TARGET_STEPS. Nothing to do.")
        return

    start_time = time.time()
    model.train()

    while step < TOTAL_TARGET_STEPS:
        elapsed = time.time() - start_time
        if elapsed > TIME_BUDGET_SECONDS:
            print(f"Time budget reached ({elapsed / 3600:.2f}h). Saving and exiting.")
            break

        for g in optimizer.param_groups:
            g["lr"] = lr_schedule(step)

        x, y = get_batch(data, cfg.context_length, BATCH_SIZE, device)
        _, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()

        if step % LOG_EVERY == 0:
            print(f"step {step:,} | loss {loss.item():.4f} | elapsed {elapsed / 60:.1f}m")

        if step % SAVE_EVERY == 0 and step > 0:
            save_checkpoint()

        step += 1

    save_checkpoint()
    print("Done for this run. The next scheduled run will resume from here.")


if __name__ == "__main__":
    main()
