# TinyGPT

GPT-2 (124M parameters) built from scratch in PyTorch, fine-tuned on Shakespeare + GSM8K math word problems.

## Pipeline

```
GPT-2 (124M) ──► Stage 1: Shakespeare ──► Stage 2: GSM8K + 10% Shakespeare ──► Gradio Demo
```

## Files

| File | What it does |
|------|-------------|
| `model.py` | GPT-2 architecture: 12 layers, 12 heads, 768 hidden dim, weight-tied embeddings |
| `train.py` | Two-stage training: Shakespeare first, then GSM8K with 10% Shakespeare mix |
| `evaluate.py` | Perplexity, sample generation, answer accuracy, Gradio web demo |
| `requirements.txt` | Python dependencies |
| `tinygpt_kaggle.ipynb` | Kaggle notebook (run on T4 GPU) |

## Quick Start

```bash
pip install -r requirements.txt

# Train from scratch (requires GPU, ~4-5 hours on T4)
python train.py

# Or load existing checkpoint and evaluate
python evaluate.py --checkpoint checkpoints/tinygpt_shake_gsm.pt --eval

# Launch Gradio demo
python evaluate.py --checkpoint checkpoints/tinygpt_shake_gsm.pt --gradio

# Generate Shakespeare samples
python evaluate.py --checkpoint checkpoints/tinygpt_shake_gsm.pt --shakespeare-samples
```

## Training Options

```bash
# Stage 1 only
python train.py --stage1-only

# Stage 2 only (requires shakespeare.pt checkpoint)
python train.py --stage2-only

# Custom parameters
python train.py --stage1-epochs 30 --stage2-epochs 8 --batch-size 4 --lr 5e-5
```

## Results

| Metric | Value |
|--------|-------|
| GSM8K Perplexity | 2.48 |
| Contains-answer rate (200 problems) | 13.5% |
| Exact-match accuracy | 4.5% |
| Shakespeare PPL (before Stage 2) | 386.39 |
| Shakespeare PPL (after Stage 2) | 124.92 |

A 124M model cannot do multi-step math reasoning (GPT-3 175B gets 58%). The evaluation focuses on format learning, structure, and answer approximation.

## Architecture

| Component | Purpose |
|-----------|---------|
| Embedding | Maps token IDs to 768-dim vectors |
| Position encoding | Adds position info to each token |
| Multi-head attention (12 heads) | Each token attends to previous tokens |
| MLP (3072 hidden) | Processes each position independently |
| LayerNorm | Stabilizes gradients |
| Weight tying | Input and output embeddings share weights |

## Interview Q&A

**Q: Why build from scratch instead of using HuggingFace?**
> I wanted to understand every layer — attention, residual connections, weight tying. Building from scratch forced me to learn the internals.

**Q: How do you load pretrained weights?**
> Download GPT-2 via HuggingFace, then map parameter names by stripping the `transformer.` prefix. Only matching shapes are copied.

**Q: Why does Stage 2 mix in Shakespeare data?**
> Without it, the model forgets Shakespeare (catastrophic forgetting). Mixing 10% Shakespeare during math training preserves both skills.

**Q: Why is math accuracy low?**
> GPT-2 has 124M parameters — too small for true reasoning. It learns patterns and memorizes solutions. This is a known limitation of small LMs.
