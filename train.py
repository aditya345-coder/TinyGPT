import os
import re
import math
import requests
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

from model import GPT, load_pretrained_weights, config


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, block_size):
        all_ids = []
        for text in texts:
            all_ids.extend(tokenizer.encode(text))

        self.inputs = []
        for i in range(0, len(all_ids) - block_size, block_size):
            chunk = all_ids[i:i + block_size + 1]
            self.inputs.append((
                torch.tensor(chunk[:-1], dtype=torch.long),
                torch.tensor(chunk[1:], dtype=torch.long),
            ))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx]


class GSM8KDataset(Dataset):
    def __init__(self, dataset, tokenizer, block_size):
        self.inputs = []
        for example in dataset:
            text = f"Question: {example['question']}\nAnswer: {example['answer']}"
            ids = tokenizer.encode(text)
            if len(ids) <= block_size + 1:
                padded = ids + [tokenizer.eos_token_id] * (block_size + 1 - len(ids))
                self.inputs.append((
                    torch.tensor(padded[:block_size], dtype=torch.long),
                    torch.tensor(padded[1:block_size + 1], dtype=torch.long),
                ))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx]


class MixedDataset(Dataset):
    def __init__(self, ds1, ds2, ratio, size=10000):
        self.ds1 = ds1
        self.ds2 = ds2
        self.ratio = ratio
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if torch.rand(1).item() < self.ratio:
            return self.ds1[torch.randint(0, len(self.ds1), (1,)).item()]
        else:
            return self.ds2[torch.randint(0, len(self.ds2), (1,)).item()]


def generate_sample(model, tokenizer, label, prompt=None, max_new=100):
    if prompt is None:
        prompt = "\n" if "shake" in label.lower() else "Question:"

    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=max_new, temperature=0.8)

    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"\n--- {label} sample ---")
    print(generated[:200])
    print("-" * 30)
    model.train()


def compute_perplexity(model, dataset, num_samples=100):
    model.eval()
    losses = []
    loader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=False)
    for i, (x, y) in enumerate(loader):
        if i >= num_samples:
            break
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            _, loss = model(x, y)
        losses.append(loss.item())
    avg_loss = sum(losses) / len(losses)
    perplexity = math.exp(avg_loss)
    model.train()
    return perplexity, avg_loss


def train(model, dataset, tokenizer, num_epochs, batch_size, lr, label):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(loader) * num_epochs
    )

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        pbar = tqdm(loader, desc=f"{label} epoch {epoch+1}/{num_epochs}")
        for step, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            _, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = epoch_loss / len(loader)
        print(f"{label} epoch {epoch+1}: avg loss = {avg_loss:.4f}")

        generate_sample(model, tokenizer, label)

    return model


def load_shakespeare_data(tokenizer):
    print("Downloading Shakespeare...")
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    response = requests.get(url)
    shakespeare_text = response.text
    print(f"Shakespeare: {len(shakespeare_text):,} characters")

    chunks = [c.strip() for c in shakespeare_text.split("\n\n") if len(c.strip()) > 50]
    print(f"{len(chunks)} chunks")

    dataset = TextDataset(chunks, tokenizer, config["block_size"])
    print(f"Shakespeare dataset: {len(dataset):,} samples")
    return dataset


def load_gsm8k_data(tokenizer):
    print("Loading GSM8K...")
    gsm8k = load_dataset("gsm8k", "main", split="train")
    print(f"GSM8K: {len(gsm8k):,} training problems")

    gsm8k = gsm8k.select(range(5000))
    print(f"Using subset: {len(gsm8k)} problems")

    dataset = GSM8KDataset(gsm8k, tokenizer, config["block_size"])
    print(f"GSM8K dataset: {len(dataset):,} samples")
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", type=str, default=None, help="Load checkpoint and skip training")
    parser.add_argument("--stage1-only", action="store_true", help="Only run Stage 1 (Shakespeare)")
    parser.add_argument("--stage2-only", action="store_true", help="Only run Stage 2 (GSM8K + mix)")
    parser.add_argument("--stage1-epochs", type=int, default=20, help="Stage 1 epochs")
    parser.add_argument("--stage2-epochs", type=int, default=10, help="Stage 2 epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    args = parser.parse_args()

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    model = GPT(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    if args.load:
        model.load_state_dict(torch.load(args.load, map_location=device))
        print(f"Loaded checkpoint: {args.load}")
    else:
        model = load_pretrained_weights(model)

    if args.load:
        print("Checkpoint loaded. Skipping training.")
    elif args.stage2_only:
        shakespeare_data = load_shakespeare_data(tokenizer)

        shake_ppl, shake_loss = compute_perplexity(model, shakespeare_data, num_samples=50)
        print(f"Shakespeare before Stage 2: PPL={shake_ppl:.2f}, Loss={shake_loss:.4f}")

        gsm8k_data = load_gsm8k_data(tokenizer)
        mixed_data = MixedDataset(gsm8k_data, shakespeare_data, ratio=0.9, size=10000)

        print(f"\n{'='*50}")
        print("Stage 2: GSM8K + Shakespeare mix")
        print(f"{'='*50}")
        model = train(
            model, mixed_data, tokenizer,
            num_epochs=args.stage2_epochs,
            batch_size=args.batch_size,
            lr=3e-5,
            label="GSM8K+Shakespeare",
        )
        torch.save(model.state_dict(), "checkpoints/tinygpt_shake_gsm.pt")
        print("Final checkpoint saved!")
    else:
        shakespeare_data = load_shakespeare_data(tokenizer)

        if not args.stage1_only:
            gsm8k_data = load_gsm8k_data(tokenizer)

        print(f"\n{'='*50}")
        print("Stage 1: Shakespeare")
        print(f"{'='*50}")
        model = train(
            model, shakespeare_data, tokenizer,
            num_epochs=args.stage1_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            label="Shakespeare",
        )
        torch.save(model.state_dict(), "checkpoints/shakespeare.pt")
        print("Shakespeare checkpoint saved!")

        if not args.stage1_only:
            shake_ppl, shake_loss = compute_perplexity(model, shakespeare_data, num_samples=50)
            print(f"Shakespeare before Stage 2: PPL={shake_ppl:.2f}, Loss={shake_loss:.4f}")

            mixed_data = MixedDataset(gsm8k_data, shakespeare_data, ratio=0.9, size=10000)

            print(f"\n{'='*50}")
            print("Stage 2: GSM8K + Shakespeare mix")
            print(f"{'='*50}")
            model = train(
                model, mixed_data, tokenizer,
                num_epochs=args.stage2_epochs,
                batch_size=args.batch_size,
                lr=3e-5,
                label="GSM8K+Shakespeare",
            )
            torch.save(model.state_dict(), "checkpoints/tinygpt_shake_gsm.pt")
            print("Final checkpoint saved!")

    print("Training complete.")


if __name__ == "__main__":
    main()
