import os
import re
import math
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

from model import GPT, config


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


def generate(model, tokenizer, prompt, max_new_tokens=150, temperature=0.8):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=temperature)
    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated


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


def extract_answer(text):
    match = re.search(r'####\s*(-?\d+\.?\d*)', text)
    return match.group(1) if match else None


def run_evaluation(model, tokenizer, shakespeare_data=None, gsm8k_data=None, num_math=200):
    os.makedirs("outputs", exist_ok=True)

    print("=" * 50)
    print("FINAL EVALUATION")
    print("=" * 50)

    results = []

    if gsm8k_data is not None:
        ppl, loss_val = compute_perplexity(model, gsm8k_data, num_samples=50)
        print(f"\nGSM8K Perplexity: {ppl:.2f}")
        print(f"Cross-Entropy Loss: {loss_val:.4f}")
        results.append(("GSM8K Perplexity", f"{ppl:.2f}"))

    print(f"\nSample Generations:")
    test_data = load_dataset("gsm8k", "main", split="test")
    model.eval()
    for i in range(5):
        ex = test_data[i]
        prompt = f"Question: {ex['question']}\nAnswer:"
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(input_ids, max_new_tokens=80, temperature=0.3)
        pred = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        pred = pred[len(prompt):].strip()
        expected = ex["answer"].strip()
        print(f"\nQ: {ex['question'][:80]}...")
        print(f"Expected: {expected[:80]}...")
        print(f"Model:   {pred[:80]}...")
        print("-" * 40)

    print(f"\nAnswer-Contains Accuracy ({num_math} problems):")
    correct_exact = 0
    correct_contains = 0
    total = 0

    test_subset = test_data.select(range(num_math))
    for ex in test_subset:
        prompt = f"Question: {ex['question']}\nAnswer:"
        expected_answer = extract_answer(ex["answer"])
        if expected_answer is None:
            continue

        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(input_ids, max_new_tokens=100, temperature=0.3)
        generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        predicted_answer = extract_answer(generated)

        if predicted_answer == expected_answer:
            correct_exact += 1
        if predicted_answer and expected_answer in generated:
            correct_contains += 1
        total += 1

    exact_acc = correct_exact / total * 100
    contains_acc = correct_contains / total * 100
    print(f"  Exact answer match:     {correct_exact}/{total} = {exact_acc:.1f}%")
    print(f"  Answer in generated text: {correct_contains}/{total} = {contains_acc:.1f}%")
    results.append(("Exact match", f"{correct_exact}/{total} = {exact_acc:.1f}%"))
    results.append(("Contains answer", f"{correct_contains}/{total} = {contains_acc:.1f}%"))

    if shakespeare_data is not None:
        ppl_shake, loss_shake = compute_perplexity(model, shakespeare_data, num_samples=50)
        print(f"\nShakespeare Perplexity: {ppl_shake:.2f}")
        print(f"Cross-Entropy Loss: {loss_shake:.4f}")
        results.append(("Shakespeare Perplexity", f"{ppl_shake:.2f}"))

    print(f"\n{'=' * 50}")
    print(f"SUMMARY: Contains-answer rate: {contains_acc:.1f}%")
    for name, val in results:
        print(f"  {name}: {val}")
    print(f"{'=' * 50}")

    with open("outputs/accuracy_report.txt", "w") as f:
        f.write("TinyGPT Evaluation Report\n")
        f.write("=" * 40 + "\n\n")
        for name, val in results:
            f.write(f"{name}: {val}\n")
        f.write(f"\nContains-answer rate: {contains_acc:.1f}%")

    return results


def launch_gradio(model, tokenizer):
    try:
        import gradio as gr
    except ImportError:
        print("gradio not installed. Run: pip install gradio")
        return

    def generate_shakespeare(prompt, max_tokens, temperature):
        output = generate(model, tokenizer, prompt, max_new_tokens=int(max_tokens), temperature=temperature)
        return output

    def solve_math(problem):
        prompt = f"Question: {problem}\nAnswer:"
        output = generate(model, tokenizer, prompt, max_new_tokens=150, temperature=0.3)
        answer = output[len(prompt):].strip()
        return answer

    with gr.Blocks(title="TinyGPT", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# TinyGPT")
        gr.Markdown("GPT-2 (124M) fine-tuned on Shakespeare + Math")

        with gr.Tab("Shakespeare"):
            gr.Markdown("Give it a prompt and it will generate Shakespeare-style text.")
            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="To be, or not to be",
                value="To be, or not to be",
            )
            with gr.Row():
                max_tokens = gr.Slider(minimum=50, maximum=500, value=150, step=10, label="Max tokens")
                temperature = gr.Slider(minimum=0.1, maximum=1.5, value=0.8, step=0.1, label="Temperature")
            generate_btn = gr.Button("Generate")
            output_text = gr.Textbox(label="Generated Text", lines=10)

            generate_btn.click(
                generate_shakespeare,
                inputs=[prompt_input, max_tokens, temperature],
                outputs=output_text,
            )

        with gr.Tab("Math Solver"):
            gr.Markdown("Type a math word problem and get the model's answer.")
            problem_input = gr.Textbox(
                label="Math Problem",
                placeholder="Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning...",
                lines=3,
            )
            solve_btn = gr.Button("Solve")
            answer_output = gr.Textbox(label="Answer", lines=5)

            solve_btn.click(solve_math, inputs=problem_input, outputs=answer_output)

        gr.Markdown("---")
        gr.Markdown("Built from scratch in PyTorch. 12 layers, 12 heads, 768 hidden dim.")

    demo.launch()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/tinygpt_shake_gsm.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--eval", action="store_true", help="Run evaluation")
    parser.add_argument("--gradio", action="store_true", help="Launch Gradio demo")
    parser.add_argument("--shakespeare-samples", action="store_true", help="Generate Shakespeare samples")
    args = parser.parse_args()

    if not args.eval and not args.gradio and not args.shakespeare_samples:
        args.eval = True
        args.gradio = True

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model from {args.checkpoint}...")
    model = GPT(config).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print("Model loaded successfully.")

    shakespeare_data = None
    gsm8k_data = None

    if args.eval or args.shakespeare_samples:
        from train import TextDataset, GSM8KDataset
        import requests

        print("Loading datasets for evaluation...")
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        response = requests.get(url)
        chunks = [c.strip() for c in response.text.split("\n\n") if len(c.strip()) > 50]
        shakespeare_data = TextDataset(chunks, tokenizer, config["block_size"])

        if args.eval:
            gsm8k = load_dataset("gsm8k", "main", split="train")
            gsm8k = gsm8k.select(range(5000))
            gsm8k_data = GSM8KDataset(gsm8k, tokenizer, config["block_size"])

    if args.shakespeare_samples:
        print("\n" + "=" * 50)
        print("Shakespeare Samples")
        print("=" * 50)
        prompts = [
            "To be, or not to be",
            "Romeo, Romeo, wherefore art thou",
            "All that glitters is not gold",
        ]
        for prompt in prompts:
            output = generate(model, tokenizer, prompt, max_new_tokens=150, temperature=0.8)
            print(f"\nPrompt: {prompt}")
            print(f"{output[:300]}")
            print("\n" + "-" * 40)

    if args.eval:
        run_evaluation(model, tokenizer, shakespeare_data, gsm8k_data)

    if args.gradio:
        launch_gradio(model, tokenizer)


if __name__ == "__main__":
    main()
